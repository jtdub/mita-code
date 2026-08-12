"""RAG retrieval pipeline: vector search over indexed code chunks."""

from __future__ import annotations

from pathlib import Path

from mita.config.schema import MitaConfig
from mita.index import get_index_dir
from mita.index.embeddings import EmbeddingClient
from mita.index.store import IndexStore, SearchResult


class Retriever:
    """High-level retrieval combining embeddings + vector search."""

    def __init__(self, config: MitaConfig, index_dir: Path | None = None) -> None:
        self._config = config
        self._index_dir = index_dir or get_index_dir()
        self._store = IndexStore(self._index_dir)
        self._embedder = EmbeddingClient(config)

    async def retrieve(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        """Retrieve relevant code chunks for a query (hybrid vector + BM25)."""
        k = top_k or self._config.index.top_k
        query_embedding = await self._embedder.embed_single(query)
        return await self._store.hybrid_search(
            query_embedding,
            query,
            top_k=k,
            floor=self._config.index.relevance_floor,
        )

    async def retrieve_formatted(self, query: str, top_k: int | None = None) -> str:
        """Retrieve and format results as a context string, capped at the token budget."""
        results = await self.retrieve(query, top_k=top_k)
        if not results:
            return ""
        return format_results(results, token_budget=self._config.index.context_token_budget)

    def is_available(self) -> bool:
        """Check if the index exists and is ready to query."""
        return self._store.exists()


def format_results(results: list[SearchResult], token_budget: int = 0) -> str:
    """Format search results as a readable context block.

    If token_budget > 0, stop adding chunks once the estimate (~4 chars/token) would
    exceed it, so retrieval can't blow up the model's context window (finding C6/S3).
    """
    parts: list[str] = []
    used_chars = 0
    char_budget = token_budget * 4 if token_budget > 0 else 0
    for r in results:
        c = r.chunk
        header = f"## {c.file_path}:{c.start_line}-{c.end_line}"
        if c.symbol:
            header += f" ({c.symbol})"
        block = f"{header}\n```{c.language}\n{c.content}\n```"
        if char_budget and parts and used_chars + len(block) > char_budget:
            break
        parts.append(block)
        used_chars += len(block)
    return "\n\n".join(parts)
