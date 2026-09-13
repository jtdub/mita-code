"""Tests for embedding generation via Ollama."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_openai import OpenAIEmbeddings

from mita.config.schema import LLMProvider, MitaConfig
from mita.index.embeddings import BATCH_SIZE, EmbeddingClient


def _mock_embedder(return_value: object) -> AsyncMock:
    embedder = AsyncMock()
    embedder.aembed_documents = AsyncMock(return_value=return_value)
    return embedder


class TestEmbedTexts:
    @pytest.mark.asyncio()
    async def test_single_batch(self) -> None:
        config = MitaConfig()
        client = EmbeddingClient(config)
        client._embeddings = _mock_embedder([[0.1, 0.2], [0.3, 0.4]])

        result = await client.embed_texts(["text1", "text2"])
        assert result == [[0.1, 0.2], [0.3, 0.4]]
        client._embeddings.aembed_documents.assert_called_once()

    @pytest.mark.asyncio()
    async def test_multiple_batches(self) -> None:
        config = MitaConfig()
        client = EmbeddingClient(config)
        texts = [f"text{i}" for i in range(BATCH_SIZE + 5)]
        batch1_embs = [[float(i)] for i in range(BATCH_SIZE)]
        batch2_embs = [[float(i)] for i in range(BATCH_SIZE, BATCH_SIZE + 5)]
        client._embeddings = AsyncMock()
        client._embeddings.aembed_documents = AsyncMock(side_effect=[batch1_embs, batch2_embs])

        result = await client.embed_texts(texts)
        assert len(result) == BATCH_SIZE + 5
        assert client._embeddings.aembed_documents.call_count == 2


class TestEmbedSingle:
    @pytest.mark.asyncio()
    async def test_returns_single_embedding(self) -> None:
        config = MitaConfig()
        client = EmbeddingClient(config)
        client._embeddings = _mock_embedder([[0.5, 0.6, 0.7]])

        result = await client.embed_single("hello world")
        assert result == [0.5, 0.6, 0.7]


class TestModelAvailability:
    @pytest.mark.asyncio()
    async def test_available(self) -> None:
        config = MitaConfig()
        client = EmbeddingClient(config)

        model = MagicMock()
        model.model = "nomic-embed-text:latest"
        models_resp = MagicMock()
        models_resp.models = [model]
        client._ollama.list = AsyncMock(return_value=models_resp)

        assert await client.is_model_available()

    @pytest.mark.asyncio()
    async def test_not_available(self) -> None:
        config = MitaConfig()
        client = EmbeddingClient(config)

        model = MagicMock()
        model.model = "llama3:latest"
        models_resp = MagicMock()
        models_resp.models = [model]
        client._ollama.list = AsyncMock(return_value=models_resp)

        assert not await client.is_model_available()

    @pytest.mark.asyncio()
    async def test_connection_error(self) -> None:
        config = MitaConfig()
        client = EmbeddingClient(config)
        client._ollama.list = AsyncMock(side_effect=ConnectionError("no server"))

        assert not await client.is_model_available()

    @pytest.mark.asyncio()
    async def test_non_ollama_available_without_registry(self) -> None:
        config = MitaConfig()
        config.llm.provider = LLMProvider.VLLM
        client = EmbeddingClient(config)
        assert client._ollama is None
        assert await client.is_model_available()


class TestNonOllamaEmbedding:
    @pytest.mark.asyncio()
    async def test_uses_openai_compatible_embeddings(self) -> None:
        config = MitaConfig()
        config.llm.provider = LLMProvider.LLAMACPP
        client = EmbeddingClient(config)
        assert isinstance(client._embeddings, OpenAIEmbeddings)
        assert client._embeddings.openai_api_base == "http://localhost:8080/v1"

        client._embeddings = _mock_embedder([[0.1, 0.2], [0.3, 0.4]])
        result = await client.embed_texts(["a", "b"])
        assert result == [[0.1, 0.2], [0.3, 0.4]]
