"""Tests for the core agent loop."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mita.agent.conversation import Conversation, Message, Role
from mita.agent.loop import MAX_ITERATIONS, _extract_delta, _parse_response, run_agent
from mita.config.schema import MitaConfig
from mita.tools.registry import create_default_registry


class TestExtractDelta:
    def test_object_chunk(self) -> None:
        class Delta:
            content = "hello"

        class Choice:
            delta = Delta()

        class Chunk:
            choices = [Choice()]

        assert _extract_delta(Chunk()) == "hello"

    def test_dict_chunk(self) -> None:
        chunk = {"choices": [{"delta": {"content": "world"}}]}
        assert _extract_delta(chunk) == "world"

    def test_empty(self) -> None:
        assert _extract_delta({}) == ""
        assert _extract_delta({"choices": []}) == ""


class TestParseResponse:
    def test_text_response(self) -> None:
        class Msg:
            content = "Hello!"
            tool_calls = None

        class Choice:
            message = Msg()

        class Response:
            choices = [Choice()]

        text, calls = _parse_response(Response())
        assert text == "Hello!"
        assert calls == []

    def test_empty_response(self) -> None:
        text, calls = _parse_response({})
        assert text == ""
        assert calls == []


class TestRunAgent:
    @pytest.mark.asyncio()
    async def test_basic_text_response(self) -> None:
        """Agent returns text with no tool calls."""
        config = MitaConfig()
        config.ui.stream = False
        console = MagicMock()
        console.print = MagicMock()

        # Mock LLM client
        mock_client = AsyncMock()

        class Msg:
            content = "Here is the answer."
            tool_calls = None

        class Choice:
            message = Msg()

        class Response:
            choices = [Choice()]

        mock_client.chat = AsyncMock(return_value=Response())

        registry = create_default_registry()

        with patch("mita.agent.loop.assemble_context"):
            conv = Conversation()
            conv.add(Message(role=Role.SYSTEM, content="system"))
            result = await run_agent(
                "test prompt",
                config,
                console,
                conversation=conv,
                registry=registry,
                llm_client=mock_client,
            )

        assert len(result.messages) >= 3  # system + user + assistant
        # User message was added
        user_msgs = [m for m in result.messages if m.role == Role.USER]
        assert len(user_msgs) == 1
        assert user_msgs[0].content == "test prompt"

    @pytest.mark.asyncio()
    async def test_max_iterations_constant(self) -> None:
        assert MAX_ITERATIONS == 25
