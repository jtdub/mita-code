"""Extended tests for LLM client."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from mita.config.schema import MitaConfig
from mita.llm.client import LLMClient, get_client


class TestLLMClient:
    def test_init(self) -> None:
        config = MitaConfig()
        client = LLMClient(config)
        assert "ollama/" in client.model
        assert config.model.default in client.model

    def test_model_property(self) -> None:
        config = MitaConfig()
        client = LLMClient(config)
        assert client.model == f"ollama/{config.model.default}"

    @pytest.mark.asyncio()
    async def test_chat(self) -> None:
        config = MitaConfig()
        client = LLMClient(config)

        mock_response = {"choices": [{"message": {"content": "hello"}}]}
        with patch("mita.llm.client.litellm.acompletion", new_callable=AsyncMock) as mock_comp:
            mock_comp.return_value = mock_response
            result = await client.chat([{"role": "user", "content": "hi"}])
            assert result == mock_response
            mock_comp.assert_called_once()

    @pytest.mark.asyncio()
    async def test_chat_with_tools(self) -> None:
        config = MitaConfig()
        client = LLMClient(config)

        tools = [{"type": "function", "function": {"name": "test"}}]
        with patch("mita.llm.client.litellm.acompletion", new_callable=AsyncMock) as mock_comp:
            mock_comp.return_value = {}
            await client.chat([{"role": "user", "content": "hi"}], tools=tools)
            call_kwargs = mock_comp.call_args.kwargs
            assert "tools" in call_kwargs

    @pytest.mark.asyncio()
    async def test_chat_with_ollama_options(self) -> None:
        config = MitaConfig()
        config.model.ollama_options.num_gpu = 1
        client = LLMClient(config)

        with patch("mita.llm.client.litellm.acompletion", new_callable=AsyncMock) as mock_comp:
            mock_comp.return_value = {}
            await client.chat([{"role": "user", "content": "hi"}])
            call_kwargs = mock_comp.call_args.kwargs
            assert "extra_body" in call_kwargs

    @pytest.mark.asyncio()
    async def test_stream_chat(self) -> None:
        config = MitaConfig()
        client = LLMClient(config)

        class FakeIterator:
            def __init__(self) -> None:
                self.items = [{"choices": [{"delta": {"content": "hi"}}]}]
                self.idx = 0

            def __aiter__(self) -> FakeIterator:
                return self

            async def __anext__(self) -> dict:
                if self.idx >= len(self.items):
                    raise StopAsyncIteration
                item = self.items[self.idx]
                self.idx += 1
                return item

        with patch("mita.llm.client.litellm.acompletion", new_callable=AsyncMock) as mock_comp:
            mock_comp.return_value = FakeIterator()
            chunks = []
            async for chunk in client.stream_chat([{"role": "user", "content": "hi"}]):
                chunks.append(chunk)
            assert len(chunks) == 1

    @pytest.mark.asyncio()
    async def test_stream_chat_with_tools(self) -> None:
        config = MitaConfig()
        client = LLMClient(config)

        class FakeIterator:
            def __aiter__(self) -> FakeIterator:
                return self

            async def __anext__(self) -> dict:
                raise StopAsyncIteration

        tools = [{"type": "function", "function": {"name": "test"}}]
        with patch("mita.llm.client.litellm.acompletion", new_callable=AsyncMock) as mock_comp:
            mock_comp.return_value = FakeIterator()
            async for _ in client.stream_chat([{"role": "user", "content": "hi"}], tools=tools):
                pass
            call_kwargs = mock_comp.call_args.kwargs
            assert "tools" in call_kwargs


class TestGetClient:
    def test_returns_client(self) -> None:
        config = MitaConfig()
        client = get_client(config)
        assert isinstance(client, LLMClient)
