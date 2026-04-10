"""Extended tests for streaming module."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from mita.llm.streaming import stream_to_terminal


class TestStreamToTerminal:
    @pytest.mark.asyncio()
    async def test_basic_streaming(self) -> None:
        async def fake_stream(*args, **kwargs):  # type: ignore[no-untyped-def]
            for content in ["Hello", " ", "world"]:
                yield {"choices": [{"delta": {"content": content}}]}

        mock_client = MagicMock()
        mock_client.stream_chat = fake_stream

        tokens: list[str] = []
        result = await stream_to_terminal(
            mock_client,
            [{"role": "user", "content": "hi"}],
            on_token=lambda t: tokens.append(t),
        )
        assert result == "Hello world"
        assert tokens == ["Hello", " ", "world"]

    @pytest.mark.asyncio()
    async def test_on_complete_called(self) -> None:
        async def fake_stream(*args, **kwargs):  # type: ignore[no-untyped-def]
            yield {"choices": [{"delta": {"content": "done"}}]}

        mock_client = MagicMock()
        mock_client.stream_chat = fake_stream

        complete_texts: list[str] = []
        result = await stream_to_terminal(
            mock_client,
            [{"role": "user", "content": "hi"}],
            on_complete=lambda t: complete_texts.append(t),
        )
        assert result == "done"
        assert complete_texts == ["done"]

    @pytest.mark.asyncio()
    async def test_on_token_error_suppressed(self) -> None:
        async def fake_stream(*args, **kwargs):  # type: ignore[no-untyped-def]
            yield {"choices": [{"delta": {"content": "hi"}}]}

        mock_client = MagicMock()
        mock_client.stream_chat = fake_stream

        def bad_callback(token: str) -> None:
            raise ValueError("callback error")

        result = await stream_to_terminal(
            mock_client,
            [{"role": "user", "content": "hi"}],
            on_token=bad_callback,
        )
        assert result == "hi"

    @pytest.mark.asyncio()
    async def test_on_complete_error_suppressed(self) -> None:
        async def fake_stream(*args, **kwargs):  # type: ignore[no-untyped-def]
            yield {"choices": [{"delta": {"content": "hi"}}]}

        mock_client = MagicMock()
        mock_client.stream_chat = fake_stream

        def bad_complete(text: str) -> None:
            raise ValueError("complete error")

        result = await stream_to_terminal(
            mock_client,
            [{"role": "user", "content": "hi"}],
            on_complete=bad_complete,
        )
        assert result == "hi"

    @pytest.mark.asyncio()
    async def test_empty_stream(self) -> None:
        async def fake_stream(*args, **kwargs):  # type: ignore[no-untyped-def]
            return
            yield  # noqa: RET504 - make it a generator

        mock_client = MagicMock()
        mock_client.stream_chat = fake_stream

        result = await stream_to_terminal(
            mock_client,
            [{"role": "user", "content": "hi"}],
        )
        assert result == ""

    @pytest.mark.asyncio()
    async def test_with_tools(self) -> None:
        async def fake_stream(*args, **kwargs):  # type: ignore[no-untyped-def]
            yield {"choices": [{"delta": {"content": "ok"}}]}

        mock_client = MagicMock()
        mock_client.stream_chat = fake_stream

        tools = [{"type": "function", "function": {"name": "test"}}]
        result = await stream_to_terminal(
            mock_client,
            [{"role": "user", "content": "hi"}],
            tools=tools,
        )
        assert result == "ok"

    @pytest.mark.asyncio()
    async def test_none_content_ignored(self) -> None:
        async def fake_stream(*args, **kwargs):  # type: ignore[no-untyped-def]
            yield {"choices": [{"delta": {"content": None}}]}
            yield {"choices": [{"delta": {"content": "text"}}]}

        mock_client = MagicMock()
        mock_client.stream_chat = fake_stream

        result = await stream_to_terminal(
            mock_client,
            [{"role": "user", "content": "hi"}],
        )
        assert result == "text"
