"""LanceDB vector store operations for code chunk storage and retrieval."""

from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path
from typing import Any

import lancedb  # type: ignore[import-untyped]
from pydantic import BaseModel, Field


class CodeChunk(BaseModel):
    """A chunk of code extracted from a source file."""

    file_path: str
    start_line: int
    end_line: int
    content: str
    language: str
    symbol: str | None = None
    embedding: list[float] | None = Field(default=None, exclude=True)


class SearchResult(BaseModel):
    """A code chunk with its similarity score."""

    chunk: CodeChunk
    score: float


class IndexStore:
    """LanceDB-backed vector store for code chunks."""

    TABLE_NAME = "code_chunks"

    def __init__(self, index_dir: Path) -> None:
        self._index_dir = index_dir
        self._db_path = index_dir / "lancedb"

    def _connect(self) -> lancedb.DBConnection:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        return lancedb.connect(str(self._db_path))

    async def create_or_replace(self, chunks: list[CodeChunk]) -> None:
        """Create or replace the index with the given chunks."""
        if not chunks:
            return

        def _sync() -> None:
            db = self._connect()
            records = _chunks_to_records(chunks)
            try:
                db.drop_table(self.TABLE_NAME)
            except (ValueError, FileNotFoundError):
                pass  # Table doesn't exist — that's fine
            db.create_table(self.TABLE_NAME, data=records)

        await asyncio.to_thread(_sync)

    async def search(self, query_embedding: list[float], top_k: int = 10) -> list[SearchResult]:
        """Vector similarity search."""

        def _sync() -> list[SearchResult]:
            db = self._connect()
            if self.TABLE_NAME not in db.table_names():
                return []
            table = db.open_table(self.TABLE_NAME)
            results = table.search(query_embedding).limit(top_k).to_pandas()

            search_results: list[SearchResult] = []
            for _, row in results.iterrows():
                distance = row.get("_distance", 0.0)
                score = 1.0 / (1.0 + distance)
                chunk = CodeChunk(
                    file_path=row["file_path"],
                    start_line=int(row["start_line"]),
                    end_line=int(row["end_line"]),
                    content=row["content"],
                    language=row["language"],
                    symbol=row.get("symbol"),
                )
                search_results.append(SearchResult(chunk=chunk, score=score))
            return search_results

        return await asyncio.to_thread(_sync)

    async def clear(self) -> None:
        """Delete the entire index."""

        def _sync() -> None:
            if self._db_path.exists():
                shutil.rmtree(self._db_path)

        await asyncio.to_thread(_sync)

    async def status(self) -> dict[str, Any]:
        """Return index statistics."""

        def _sync() -> dict[str, Any]:
            if not self._db_path.exists():
                return {"exists": False, "chunks": 0, "files": 0}
            db = self._connect()
            if self.TABLE_NAME not in db.table_names():
                return {"exists": False, "chunks": 0, "files": 0}
            table = db.open_table(self.TABLE_NAME)
            df = table.to_pandas()
            return {
                "exists": True,
                "chunks": len(df),
                "files": df["file_path"].nunique() if len(df) > 0 else 0,
            }

        return await asyncio.to_thread(_sync)

    def exists(self) -> bool:
        """Check if the index exists on disk."""
        if not self._db_path.exists():
            return False
        try:
            db = self._connect()
            return self.TABLE_NAME in db.table_names()
        except Exception:  # noqa: BLE001
            logging.getLogger(__name__).warning(
                "Index existence check failed for %s", self._db_path, exc_info=True
            )
            return False


def _chunks_to_records(chunks: list[CodeChunk]) -> list[dict[str, Any]]:
    """Convert CodeChunks (with embeddings) to LanceDB-compatible records."""
    records = []
    for chunk in chunks:
        if chunk.embedding is None:
            continue
        records.append(
            {
                "file_path": chunk.file_path,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "content": chunk.content,
                "language": chunk.language,
                "symbol": chunk.symbol or "",
                "vector": chunk.embedding,
            }
        )
    return records
