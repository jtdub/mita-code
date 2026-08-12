"""Embedding generation for code chunks, routed to the configured backend (finding C5)."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import litellm
import ollama

from mita.config.schema import MitaConfig
from mita.llm.providers import resolve_backend

BATCH_SIZE = 32

_logger = logging.getLogger(__name__)


class EmbeddingClient:
    """Generate embeddings via Ollama (native) or an OpenAI-compatible /v1/embeddings."""

    def __init__(self, config: MitaConfig) -> None:
        self._model = config.model.embedding
        self._backend = resolve_backend(config)
        self._timeout = config.ollama.timeout
        # Ollama uses its native client; every other backend serves /v1/embeddings, for
        # which the universal LiteLLM route is `openai/<model>` + api_base.
        self._ollama = ollama.AsyncClient(host=config.ollama.host) if self.is_ollama else None

    @property
    def model(self) -> str:
        return self._model

    @property
    def is_ollama(self) -> bool:
        return self._backend.is_ollama

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts."""
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            all_embeddings.extend(await self._embed_batch(batch))
        return all_embeddings

    async def embed_single(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        result = await self._embed_batch([text])
        return result[0]

    async def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        if self._ollama is not None:
            ollama_resp = await asyncio.wait_for(
                self._ollama.embed(model=self._model, input=batch),
                timeout=self._timeout,
            )
            return [list(e) for e in ollama_resp.embeddings]

        # OpenAI-compatible path. LiteLLM returns the OpenAI shape
        # (response.data[i]["embedding"]), which differs from Ollama's .embeddings.
        litellm_resp: Any = await asyncio.wait_for(
            litellm.aembedding(
                model=f"openai/{self._model}",
                input=batch,
                api_base=self._backend.api_base,
                api_key=self._backend.api_key,
            ),
            timeout=self._timeout,
        )
        return [_extract_embedding(item) for item in litellm_resp.data]

    async def is_model_available(self) -> bool:
        """Check the embedding model is available.

        Only Ollama exposes a model registry; for other backends we cannot verify, so
        we assume the operator loaded an embedding model and report available.
        """
        if self._ollama is None:
            return True
        try:
            models = await self._ollama.list()
            model_names = [m.model for m in models.models]
            base_name = self._model.split(":")[0]
            return any(
                m == self._model or (m is not None and m.startswith(base_name + ":"))
                for m in model_names
            )
        except (ollama.ResponseError, ConnectionError, OSError) as e:
            _logger.warning("Embedding model check failed: %s", e)
            return False


def _extract_embedding(item: Any) -> list[float]:
    """Read the vector from a LiteLLM/OpenAI embedding item (dict or object)."""
    vec = item["embedding"] if isinstance(item, dict) else item.embedding
    return [float(x) for x in vec]
