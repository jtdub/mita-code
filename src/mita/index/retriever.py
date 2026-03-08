"""RAG retrieval pipeline: vector search over indexed code chunks."""

from __future__ import annotations

from pathlib import Path

from mita.config.schema import MitaConfig
from mita.index.embeddings import EmbeddingClient
from mita.index.store import IndexStore, SearchResult


def _default_index_dir() -> Path:
    """Determine the index directory for the current project."""
    return Path.cwd() / ".mita" / "index"


class Retriever:
    """High-level retrieval combining embeddings + vector search."""

    def __init__(self, config: MitaConfig, index_dir: Path | None = None) -> None:
        self._config = config
        self._index_dir = index_dir or _default_index_dir()
        self._store = IndexStore(self._index_dir)
        self._embedder = EmbeddingClient(config)

    async def retrieve(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        """Retrieve relevant code chunks for a query."""
        k = top_k or self._config.index.top_k
        query_embedding = await self._embedder.embed_single(query)
        return await self._store.search(query_embedding, top_k=k)

    async def retrieve_formatted(self, query: str, top_k: int | None = None) -> str:
        """Retrieve and format results as a context string for the LLM."""
        results = await self.retrieve(query, top_k=top_k)
        if not results:
            return ""
        return format_results(results)

    def is_available(self) -> bool:
        """Check if the index exists and is ready to query."""
        return self._store.exists()


def format_results(results: list[SearchResult]) -> str:
    """Format search results as a readable context block."""
    parts: list[str] = []
    for r in results:
        c = r.chunk
        header = f"## {c.file_path}:{c.start_line}-{c.end_line}"
        if c.symbol:
            header += f" ({c.symbol})"
        parts.append(f"{header}\n```{c.language}\n{c.content}\n```")
    return "\n\n".join(parts)
