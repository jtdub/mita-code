"""Tests for the core agent loop."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mita.agent.conversation import Conversation, Message, Role
from mita.agent.loop import (
    MAX_ITERATIONS,
    _accumulate_tool_call_deltas,
    _extract_delta,
    _extract_tool_calls_from_text,
    _parse_response,
    run_agent,
)
from mita.config.schema import MitaConfig
from mita.tools.registry import ToolRegistry, create_default_registry
from mita.tools.schema import ToolDefinition, ToolParameter, ToolResult


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


class TestAccumulateToolCallDeltas:
    def test_accumulate_object_style(self) -> None:
        """Accumulate tool call deltas from object-style streaming chunks."""

        class Func:
            name = "file_read"
            arguments = '{"path":'

        class ToolCallDelta:
            index = 0
            id = "tc_1"
            function = Func()

        class Delta:
            content = None
            tool_calls = [ToolCallDelta()]

        class Choice:
            delta = Delta()

        class Chunk:
            choices = [Choice()]

        acc: dict[int, dict[str, Any]] = {}
        _accumulate_tool_call_deltas(Chunk(), acc)

        assert 0 in acc
        assert acc[0]["id"] == "tc_1"
        assert acc[0]["function"]["name"] == "file_read"
        assert acc[0]["function"]["arguments"] == '{"path":'

        # Second chunk appends arguments
        class Func2:
            name = None
            arguments = '"/tmp"}'

        class ToolCallDelta2:
            index = 0
            id = None
            function = Func2()

        class Delta2:
            content = None
            tool_calls = [ToolCallDelta2()]

        class Choice2:
            delta = Delta2()

        class Chunk2:
            choices = [Choice2()]

        _accumulate_tool_call_deltas(Chunk2(), acc)
        assert acc[0]["function"]["arguments"] == '{"path":"/tmp"}'

    def test_accumulate_dict_style(self) -> None:
        """Accumulate tool call deltas from dict-style streaming chunks."""
        acc: dict[int, dict[str, Any]] = {}
        chunk = {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "tc_2",
                                "function": {"name": "shell", "arguments": '{"cmd":'},
                            }
                        ]
                    }
                }
            ]
        }
        _accumulate_tool_call_deltas(chunk, acc)
        assert acc[0]["function"]["name"] == "shell"

    def test_empty_chunk_no_error(self) -> None:
        """Empty chunks should be silently ignored."""
        acc: dict[int, dict[str, Any]] = {}
        _accumulate_tool_call_deltas({}, acc)
        assert acc == {}
        _accumulate_tool_call_deltas({"choices": []}, acc)
        assert acc == {}


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


def _make_registry_with_tool(name: str) -> ToolRegistry:
    """Create a registry with a single dummy tool for testing text extraction."""
    registry = ToolRegistry()

    async def dummy_handler(args: dict[str, Any]) -> ToolResult:
        return ToolResult(tool_call_id="test", success=True, output="ok")

    registry.register(
        ToolDefinition(
            name=name,
            description="test tool",
            parameters=[
                ToolParameter(name="path", type="string", description="file path"),
            ],
        ),
        dummy_handler,
    )
    return registry


class TestExtractToolCallsFromText:
    def test_bare_json_tool_call(self) -> None:
        registry = _make_registry_with_tool("file_write")
        text = '{"name": "file_write", "arguments": {"path": "hello.py"}}'
        calls, remaining = _extract_tool_calls_from_text(text, registry)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "file_write"
        assert remaining == ""

    def test_fenced_json_tool_call(self) -> None:
        registry = _make_registry_with_tool("shell")
        text = '```json\n{"name": "shell", "arguments": {"command": "ls"}}\n```'
        calls, remaining = _extract_tool_calls_from_text(text, registry)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "shell"

    def test_text_with_tool_call(self) -> None:
        registry = _make_registry_with_tool("file_write")
        text = (
            "I'll create the file for you.\n"
            '{"name": "file_write", "arguments": {"path": "test.py"}}\n'
        )
        calls, remaining = _extract_tool_calls_from_text(text, registry)
        assert len(calls) == 1
        assert "create the file" in remaining

    def test_unknown_tool_not_extracted(self) -> None:
        registry = _make_registry_with_tool("file_write")
        text = '{"name": "unknown_tool", "arguments": {"foo": "bar"}}'
        calls, remaining = _extract_tool_calls_from_text(text, registry)
        assert len(calls) == 0
        assert remaining == text

    def test_plain_text_no_extraction(self) -> None:
        registry = _make_registry_with_tool("file_write")
        text = "Here is how to create a file."
        calls, remaining = _extract_tool_calls_from_text(text, registry)
        assert len(calls) == 0
        assert remaining == text

    def test_invalid_json_ignored(self) -> None:
        registry = _make_registry_with_tool("file_write")
        text = '{"name": "file_write", "arguments": {invalid}}'
        calls, remaining = _extract_tool_calls_from_text(text, registry)
        assert len(calls) == 0

    def test_empty_text(self) -> None:
        registry = _make_registry_with_tool("file_write")
        calls, remaining = _extract_tool_calls_from_text("", registry)
        assert len(calls) == 0
        assert remaining == ""
