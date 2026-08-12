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
    # Schema-forward metadata (finding C6). symbol_path uses the shared-schema naming so
    # a Mita index stays export-compatible with Hearth AI; content_hash drives incremental
    # re-embedding; chunk_type enables filtered search. chunk_id is STABLE across line
    # shifts (it excludes start_line), so a chunk that only moves is not re-embedded.
    symbol_path: str = ""
    chunk_type: str = ""
    content_hash: str = ""
    chunk_id: str = ""
    occurrence_ordinal: int = 0
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

    INDEX_VERSION = 1

    @property
    def _meta_path(self) -> Path:
        return self._index_dir / "meta.json"

    def read_meta(self) -> dict[str, Any]:
        """Read the index metadata sidecar (version, embed model, build time)."""
        import json

        try:
            return dict(json.loads(self._meta_path.read_text()))
        except (OSError, ValueError):
            return {}

    def _write_meta(self, embed_model: str, built_at: float) -> None:
        import json

        from mita.config.toml_writer import atomic_write_text

        atomic_write_text(
            self._meta_path,
            json.dumps(
                {
                    "index_version": self.INDEX_VERSION,
                    "embed_model": embed_model,
                    "built_at": built_at,
                }
            ),
        )

    def needs_full_rebuild(self, embed_model: str) -> bool:
        """True if the on-disk index is missing, a different version, or a different model.

        A changed embedding model means a different vector dimension, which cannot be
        merged into the fixed-width column — it must be a full rebuild (finding C6).
        """
        if not self.exists():
            return True
        meta = self.read_meta()
        return meta.get("index_version") != self.INDEX_VERSION or meta.get("embed_model") != (
            embed_model
        )

    async def create_or_replace(
        self, chunks: list[CodeChunk], embed_model: str = "", built_at: float = 0.0
    ) -> None:
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
            table = db.create_table(self.TABLE_NAME, data=records)
            self._ensure_fts(table)
            self._write_meta(embed_model, built_at)

        await asyncio.to_thread(_sync)

    def _ensure_fts(self, table: Any) -> None:
        """(Re)build the full-text index on `content` for hybrid search.

        use_tantivy=False selects LanceDB's native FTS (no `tantivy` dependency). FTS is
        optional — hybrid search falls back to vector-only if this fails (finding C6/B2).
        """
        try:
            table.create_fts_index("content", use_tantivy=False, replace=True)
        except Exception:  # noqa: BLE001 - FTS is a best-effort enhancement
            logging.getLogger(__name__).debug("FTS index build failed", exc_info=True)

    async def load_content_hashes(self) -> dict[str, str]:
        """Return {chunk_id: content_hash} for the current index (empty if none)."""

        def _sync() -> dict[str, str]:
            db = self._connect()
            if self.TABLE_NAME not in db.table_names():
                return {}
            table = db.open_table(self.TABLE_NAME)
            df = table.to_pandas()
            if "chunk_id" not in df.columns or "content_hash" not in df.columns:
                return {}
            return dict(zip(df["chunk_id"], df["content_hash"], strict=False))

        return await asyncio.to_thread(_sync)

    async def apply_incremental(
        self,
        changed: list[CodeChunk],
        present_ids: set[str],
        embed_model: str,
        built_at: float,
    ) -> None:
        """Upsert changed chunks (by chunk_id) and delete chunks no longer present.

        Upsert BEFORE delete so the index stays queryable throughout (finding C6/M6).
        """

        def _sync() -> None:
            db = self._connect()
            table = db.open_table(self.TABLE_NAME)

            records = _chunks_to_records(changed)
            if records:
                (
                    table.merge_insert("chunk_id")
                    .when_matched_update_all()
                    .when_not_matched_insert_all()
                    .execute(records)
                )

            existing_ids = set(table.to_pandas()["chunk_id"]) if table.count_rows() else set()
            removed = existing_ids - present_ids
            for batch in _batched(sorted(removed), 500):
                quoted = ", ".join(f"'{cid}'" for cid in batch)
                table.delete(f"chunk_id IN ({quoted})")

            self._ensure_fts(table)
            self._write_meta(embed_model, built_at)

        await asyncio.to_thread(_sync)

    async def hybrid_search(
        self,
        query_vector: list[float],
        query_text: str,
        top_k: int = 10,
        floor: float = 0.0,
    ) -> list[SearchResult]:
        """Hybrid (vector + BM25) search with RRF fusion, falling back to vector-only."""

        def _sync() -> list[SearchResult]:
            db = self._connect()
            if self.TABLE_NAME not in db.table_names():
                return []
            table = db.open_table(self.TABLE_NAME)
            score_col: str | None
            try:
                results = (
                    table.search(query_type="hybrid")
                    .vector(query_vector)
                    .text(query_text)
                    .limit(top_k)
                    .to_pandas()
                )
                score_col = "_relevance_score"
            except Exception:  # noqa: BLE001 - no FTS index / hybrid unsupported
                results = table.search(query_vector).limit(top_k).to_pandas()
                score_col = None

            out: list[SearchResult] = []
            for _, row in results.iterrows():
                if score_col is not None and score_col in row:
                    score = float(row[score_col])
                else:
                    score = 1.0 / (1.0 + float(row.get("_distance", 0.0)))
                if score < floor:
                    continue
                out.append(SearchResult(chunk=_row_to_chunk(row), score=score))
            return out

        return await asyncio.to_thread(_sync)

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
                search_results.append(SearchResult(chunk=_row_to_chunk(row), score=score))
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
            count = table.count_rows()
            files = 0
            if count:
                files = int(table.to_pandas()["file_path"].nunique())
            meta = self.read_meta()
            return {
                "exists": True,
                "chunks": count,
                "files": files,
                "embed_model": meta.get("embed_model", ""),
                "built_at": meta.get("built_at", 0.0),
            }

        return await asyncio.to_thread(_sync)

    def is_stale(self, root: Path) -> bool:
        """True if a source file changed since the index was built.

        Bounded and cheap: in a git repo it stats only the changed/untracked files that
        `git status` reports, comparing their mtime to the index build time. Outside a
        git repo it can't tell cheaply, so it reports not-stale (finding C6).
        """
        import subprocess

        built_at = float(self.read_meta().get("built_at", 0.0) or 0.0)
        if not built_at or not (root / ".git").exists():
            return not built_at
        try:
            proc = subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        for line in proc.stdout.splitlines():
            rel = line[3:].strip().strip('"')
            if " -> " in rel:  # rename: take the destination
                rel = rel.split(" -> ", 1)[1]
            try:
                if (root / rel).stat().st_mtime > built_at:
                    return True
            except OSError:
                continue
        return False

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


def _batched(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _row_to_chunk(row: Any) -> CodeChunk:
    """Build a CodeChunk from a LanceDB result row (tolerant of missing columns)."""

    def get(key: str, default: Any = "") -> Any:
        try:
            value = row[key]
        except (KeyError, IndexError):
            return default
        return default if value is None else value

    return CodeChunk(
        file_path=get("file_path"),
        start_line=int(get("start_line", 0) or 0),
        end_line=int(get("end_line", 0) or 0),
        content=get("content"),
        language=get("language"),
        symbol=get("symbol") or None,
        symbol_path=get("symbol_path"),
        chunk_type=get("chunk_type"),
        content_hash=get("content_hash"),
        chunk_id=get("chunk_id"),
    )


def _chunks_to_records(chunks: list[CodeChunk]) -> list[dict[str, Any]]:
    """Convert CodeChunks (with embeddings) to LanceDB-compatible records."""
    records = []
    for chunk in chunks:
        if chunk.embedding is None:
            continue
        records.append(
            {
                "chunk_id": chunk.chunk_id,
                "file_path": chunk.file_path,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "content": chunk.content,
                "language": chunk.language,
                "symbol": chunk.symbol or "",
                "symbol_path": chunk.symbol_path,
                "chunk_type": chunk.chunk_type,
                "content_hash": chunk.content_hash,
                "vector": chunk.embedding,
            }
        )
    return records
