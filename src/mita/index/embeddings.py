"""Embedding generation via Ollama for code chunks."""

from __future__ import annotations

import asyncio
import logging

import ollama

from mita.config.schema import MitaConfig

BATCH_SIZE = 32
EMBED_TIMEOUT = 120  # seconds per batch


class EmbeddingClient:
    """Generate embeddings using Ollama's embedding endpoint."""

    def __init__(self, config: MitaConfig) -> None:
        self._model = config.model.embedding
        self._client = ollama.AsyncClient(host=config.ollama.host)
        self._timeout = config.ollama.timeout

    @property
    def model(self) -> str:
        return self._model

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts."""
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            response = await asyncio.wait_for(
                self._client.embed(model=self._model, input=batch),
                timeout=self._timeout,
            )
            all_embeddings.extend(list(e) for e in response.embeddings)
        return all_embeddings

    async def embed_single(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        response = await asyncio.wait_for(
            self._client.embed(model=self._model, input=[text]),
            timeout=self._timeout,
        )
        return list(response.embeddings[0])

    async def is_model_available(self) -> bool:
        """Check if the embedding model is pulled in Ollama."""
        try:
            models = await self._client.list()
            model_names = [m.model for m in models.models]
            # Check both exact match and base name (without tag)
            base_name = self._model.split(":")[0]
            return any(
                m == self._model or (m is not None and m.startswith(base_name + ":"))
                for m in model_names
            )
        except (ollama.ResponseError, ConnectionError, OSError) as e:
            logging.getLogger(__name__).warning("Embedding model check failed: %s", e)
            return False
