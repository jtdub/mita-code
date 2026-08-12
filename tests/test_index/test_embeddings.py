"""Tests for embedding generation via Ollama."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from mita.config.schema import MitaConfig
from mita.index.embeddings import BATCH_SIZE, EmbeddingClient


def _mock_embed_response(embeddings: list[list[float]]) -> MagicMock:
    resp = MagicMock()
    resp.embeddings = embeddings
    return resp


class TestEmbedTexts:
    @pytest.mark.asyncio()
    async def test_single_batch(self) -> None:
        config = MitaConfig()
        client = EmbeddingClient(config)
        mock_embed = AsyncMock(return_value=_mock_embed_response([[0.1, 0.2], [0.3, 0.4]]))
        client._ollama.embed = mock_embed

        result = await client.embed_texts(["text1", "text2"])
        assert len(result) == 2
        assert result[0] == [0.1, 0.2]
        mock_embed.assert_called_once()

    @pytest.mark.asyncio()
    async def test_multiple_batches(self) -> None:
        config = MitaConfig()
        client = EmbeddingClient(config)
        texts = [f"text{i}" for i in range(BATCH_SIZE + 5)]

        batch1_embs = [[float(i)] for i in range(BATCH_SIZE)]
        batch2_embs = [[float(i)] for i in range(BATCH_SIZE, BATCH_SIZE + 5)]

        mock_embed = AsyncMock(
            side_effect=[
                _mock_embed_response(batch1_embs),
                _mock_embed_response(batch2_embs),
            ]
        )
        client._ollama.embed = mock_embed

        result = await client.embed_texts(texts)
        assert len(result) == BATCH_SIZE + 5
        assert mock_embed.call_count == 2


class TestEmbedSingle:
    @pytest.mark.asyncio()
    async def test_returns_single_embedding(self) -> None:
        config = MitaConfig()
        client = EmbeddingClient(config)
        mock_embed = AsyncMock(return_value=_mock_embed_response([[0.5, 0.6, 0.7]]))
        client._ollama.embed = mock_embed

        result = await client.embed_single("hello world")
        assert result == [0.5, 0.6, 0.7]
        mock_embed.assert_called_once()


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
        from mita.config.schema import LLMProvider

        config = MitaConfig()
        config.llm.provider = LLMProvider.VLLM
        client = EmbeddingClient(config)
        assert client._ollama is None
        # No registry to check; must report available rather than blocking.
        assert await client.is_model_available()


class TestNonOllamaEmbedding:
    @pytest.mark.asyncio()
    async def test_litellm_path_normalizes_openai_shape(self) -> None:
        from unittest.mock import patch

        from mita.config.schema import LLMProvider

        config = MitaConfig()
        config.llm.provider = LLMProvider.LLAMACPP
        client = EmbeddingClient(config)

        # OpenAI/LiteLLM shape: response.data[i]["embedding"].
        resp = MagicMock()
        resp.data = [{"embedding": [0.1, 0.2]}, {"embedding": [0.3, 0.4]}]
        with patch(
            "mita.index.embeddings.litellm.aembedding", new_callable=AsyncMock, return_value=resp
        ) as mock_embed:
            result = await client.embed_texts(["a", "b"])
            assert result == [[0.1, 0.2], [0.3, 0.4]]
            kwargs = mock_embed.call_args.kwargs
            assert kwargs["model"] == "openai/nomic-embed-text"
            assert kwargs["api_base"] == "http://localhost:8080/v1"
