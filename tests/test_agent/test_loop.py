"""Tests for the core agent loop (LangGraph)."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk

from mita.agent.conversation import Conversation, Message, Role
from mita.agent.loop import _extract_tool_calls_from_text, run_agent
from mita.config.schema import MitaConfig
from mita.tools.registry import ToolRegistry
from mita.tools.schema import ToolDefinition, ToolParameter, ToolResult


def _config() -> MitaConfig:
    config = MitaConfig()
    config.ui.stream = False
    config.llm.context_probe = "off"
    config.index.enabled = False
    return config


def _fake_model(
    ainvoke_side_effect: list[Any] | None = None, stream_chunks: list[Any] | None = None
) -> MagicMock:
    """Return a mock LangChain model with the loop's expected surface."""
    model = MagicMock()
    model.bind_tools = MagicMock(side_effect=lambda tools: model)
    if stream_chunks is not None:

        async def _astream(_messages: object) -> Any:
            for chunk in stream_chunks:
                yield chunk

        model.astream = _astream
    else:
        model.ainvoke = AsyncMock(return_value=AIMessage(content="Here is the answer."))
        if ainvoke_side_effect is not None:
            model.ainvoke = AsyncMock(side_effect=ainvoke_side_effect)
    return model


class TestRunAgent:
    @pytest.mark.asyncio()
    async def test_basic_text_response(self) -> None:
        config = _config()
        console = MagicMock()

        with patch("mita.agent.loop.build_chat_model", return_value=_fake_model()):
            conv = Conversation()
            conv.add(Message(role=Role.SYSTEM, content="system"))
            result = await run_agent("test prompt", config, console, conversation=conv)

        user_msgs = [m for m in result.messages if m.role == Role.USER]
        assert len(user_msgs) == 1
        assert user_msgs[0].content == "test prompt"
        assistant_msgs = [m for m in result.messages if m.role == Role.ASSISTANT]
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0].content == "Here is the answer."

    @pytest.mark.asyncio()
    async def test_tool_call_flow(self) -> None:
        config = _config()
        console = MagicMock()

        tool_response = AIMessage(
            content="",
            tool_calls=[{"name": "file_read", "args": {"path": "/tmp/none"}, "id": "call-1"}],
        )
        model = _fake_model(ainvoke_side_effect=[tool_response, AIMessage(content="done")])
        with patch("mita.agent.loop.build_chat_model", return_value=model):
            conv = Conversation()
            conv.add(Message(role=Role.SYSTEM, content="system"))
            result = await run_agent("read a file", config, console, conversation=conv)

        tool_msgs = [m for m in result.messages if m.role == Role.TOOL]
        assert len(tool_msgs) == 1
        assert tool_msgs[0].tool_call_id == "call-1"
        assistant_msgs = [m for m in result.messages if m.role == Role.ASSISTANT]
        assert assistant_msgs[-1].content == "done"

    @pytest.mark.asyncio()
    async def test_streaming_mode(self) -> None:
        config = _config()
        config.ui.stream = True
        console = MagicMock()

        model = _fake_model(
            stream_chunks=[AIMessageChunk(content="streamed "), AIMessageChunk(content="answer")]
        )
        with patch("mita.agent.loop.build_chat_model", return_value=model):
            conv = Conversation()
            conv.add(Message(role=Role.SYSTEM, content="system"))
            result = await run_agent("test", config, console, conversation=conv)

        assistant_msgs = [m for m in result.messages if m.role == Role.ASSISTANT]
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0].content == "streamed answer"

    @pytest.mark.asyncio()
    async def test_streaming_tool_call_deltas(self) -> None:
        config = _config()
        config.ui.stream = True
        console = MagicMock()

        model = _fake_model(
            stream_chunks=[
                AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {
                            "name": "file_",
                            "args": '{"path"',
                            "id": "call-1",
                            "index": 0,
                            "type": "function",
                        }
                    ],
                ),
                AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {"name": "read", "args": ': "/tmp/none"}', "index": 0, "type": "function"}
                    ],
                ),
            ]
        )
        with patch("mita.agent.loop.build_chat_model", return_value=model):
            conv = Conversation()
            conv.add(Message(role=Role.SYSTEM, content="system"))
            result = await run_agent("read", config, console, conversation=conv)

        tool_msgs = [m for m in result.messages if m.role == Role.TOOL]
        assert len(tool_msgs) == 1

    @pytest.mark.asyncio()
    async def test_streaming_multiple_tool_calls_in_one_chunk(self) -> None:
        """Two tool calls in one chunk (index unset) must stay separate (finding #1)."""
        config = _config()
        config.ui.stream = True
        console = MagicMock()

        model = _fake_model(
            stream_chunks=[
                AIMessageChunk(
                    content="",
                    tool_calls=[
                        {"name": "file_read", "args": {"path": "/tmp/none"}, "id": "call-1"},
                        {"name": "file_read", "args": {"path": "/tmp/none2"}, "id": "call-2"},
                    ],
                )
            ]
        )
        with patch("mita.agent.loop.build_chat_model", return_value=model):
            conv = Conversation()
            conv.add(Message(role=Role.SYSTEM, content="system"))
            result = await run_agent("read two", config, console, conversation=conv)

        tool_msgs = [m for m in result.messages if m.role == Role.TOOL]
        assert len(tool_msgs) == 2
        assert {m.tool_call_id for m in tool_msgs} == {"call-1", "call-2"}

    @pytest.mark.asyncio()
    async def test_text_json_fallback(self) -> None:
        config = _config()
        console = MagicMock()

        model = _fake_model(
            ainvoke_side_effect=[
                AIMessage(content='{"name": "file_read", "arguments": {"path": "/tmp/none"}}'),
                AIMessage(content="done"),
            ]
        )
        with patch("mita.agent.loop.build_chat_model", return_value=model):
            conv = Conversation()
            conv.add(Message(role=Role.SYSTEM, content="system"))
            result = await run_agent("read", config, console, conversation=conv)

        tool_msgs = [m for m in result.messages if m.role == Role.TOOL]
        assert len(tool_msgs) == 1

    @pytest.mark.asyncio()
    async def test_max_iterations_reached(self) -> None:
        from mita.ui.sink import RecordingSink

        config = _config()
        config.max_iterations = 1
        console = MagicMock()
        sink = RecordingSink()

        model = _fake_model(
            ainvoke_side_effect=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "file_read", "args": {"path": "/tmp/none"}, "id": "call-1"}
                    ],
                )
            ]
        )
        with patch("mita.agent.loop.build_chat_model", return_value=model):
            conv = Conversation()
            conv.add(Message(role=Role.SYSTEM, content="system"))
            result = await run_agent("test", config, console, conversation=conv, sink=sink)

        assert isinstance(result, Conversation)
        tool_msgs = [m for m in result.messages if m.role == Role.TOOL]
        assert len(tool_msgs) == 1
        assert any(
            kind == "error" and "maximum iterations" in str(message)
            for kind, message in sink.events
        )

    @pytest.mark.asyncio()
    async def test_max_iterations_plain_answer_no_message(self) -> None:
        """A plain answer on the last allowed iteration is a normal end."""
        from mita.ui.sink import RecordingSink

        config = _config()
        config.max_iterations = 1
        console = MagicMock()
        sink = RecordingSink()

        with patch("mita.agent.loop.build_chat_model", return_value=_fake_model()):
            conv = Conversation()
            conv.add(Message(role=Role.SYSTEM, content="system"))
            result = await run_agent("test", config, console, conversation=conv, sink=sink)

        assistant_msgs = [m for m in result.messages if m.role == Role.ASSISTANT]
        assert len(assistant_msgs) == 1
        assert not any(
            kind == "error" and "maximum iterations" in str(message)
            for kind, message in sink.events
        )

    @pytest.mark.asyncio()
    async def test_repeated_tool_calls_detected(self) -> None:
        config = _config()
        config.max_iterations = 5
        console = MagicMock()

        tool_response = AIMessage(
            content="",
            tool_calls=[{"name": "file_read", "args": {"path": "/tmp/none"}, "id": "call-1"}],
        )
        model = _fake_model(
            ainvoke_side_effect=[
                tool_response,
                tool_response,
                AIMessage(content="done"),
            ]
        )
        with patch("mita.agent.loop.build_chat_model", return_value=model):
            conv = Conversation()
            conv.add(Message(role=Role.SYSTEM, content="system"))
            result = await run_agent("test", config, console, conversation=conv)

        # The repeated batch must not run a second time (only one tool result).
        tool_msgs = [m for m in result.messages if m.role == Role.TOOL]
        assert len(tool_msgs) == 1

    @pytest.mark.asyncio()
    async def test_connection_error_handling(self) -> None:
        config = _config()
        console = MagicMock()

        model = _fake_model()
        model.ainvoke = AsyncMock(side_effect=ConnectionError("offline"))
        with patch("mita.agent.loop.build_chat_model", return_value=model):
            conv = Conversation()
            conv.add(Message(role=Role.SYSTEM, content="system"))
            result = await run_agent("test", config, console, conversation=conv)

        assert isinstance(result, Conversation)

    @pytest.mark.asyncio()
    async def test_json_decode_error(self) -> None:
        config = _config()
        console = MagicMock()

        model = _fake_model()
        model.ainvoke = AsyncMock(side_effect=json.JSONDecodeError("err", "", 0))
        with patch("mita.agent.loop.build_chat_model", return_value=model):
            conv = Conversation()
            conv.add(Message(role=Role.SYSTEM, content="system"))
            result = await run_agent("test", config, console, conversation=conv)

        assert isinstance(result, Conversation)

    @pytest.mark.asyncio()
    async def test_rag_context_injection(self) -> None:
        config = _config()
        config.index.enabled = True
        console = MagicMock()

        mock_retriever = MagicMock()
        mock_retriever.is_available.return_value = True
        mock_retriever.retrieve_formatted = AsyncMock(return_value="relevant code")

        with (
            patch("mita.agent.loop.build_chat_model", return_value=_fake_model()),
            patch("mita.index.retriever.Retriever", return_value=mock_retriever),
        ):
            conv = Conversation()
            conv.add(Message(role=Role.SYSTEM, content="system"))
            result = await run_agent("test", config, console, conversation=conv)

        rag_msgs = [
            m for m in result.messages if m.role == Role.SYSTEM and "Relevant code" in m.content
        ]
        assert len(rag_msgs) == 1

    @pytest.mark.asyncio()
    async def test_hooks_called(self) -> None:
        from mita.config.schema import HookDefinition

        config = _config()
        config.hooks = [HookDefinition(event="session_start", command="echo start")]
        console = MagicMock()

        with (
            patch("mita.agent.loop.build_chat_model", return_value=_fake_model()),
            patch("mita.hooks.runner.run_hooks", new_callable=AsyncMock) as mock_hooks,
        ):
            conv = Conversation()
            conv.add(Message(role=Role.SYSTEM, content="system"))
            await run_agent("test", config, console, conversation=conv)

        assert mock_hooks.call_count >= 2


class TestKeyboardInterrupt:
    @pytest.mark.asyncio()
    async def test_interrupt_during_llm_call(self) -> None:
        config = _config()
        console = MagicMock()

        model = _fake_model()
        model.ainvoke = AsyncMock(side_effect=KeyboardInterrupt)
        with patch("mita.agent.loop.build_chat_model", return_value=model):
            conv = Conversation()
            conv.add(Message(role=Role.SYSTEM, content="system"))
            result = await run_agent("test", config, console, conversation=conv)

        user_msgs = [m for m in result.messages if m.role == Role.USER]
        assert len(user_msgs) == 1

    @pytest.mark.asyncio()
    async def test_interrupt_preserves_conversation(self) -> None:
        config = _config()
        console = MagicMock()

        model = _fake_model()
        model.ainvoke = AsyncMock(side_effect=KeyboardInterrupt)
        with patch("mita.agent.loop.build_chat_model", return_value=model):
            conv = Conversation()
            conv.add(Message(role=Role.SYSTEM, content="system"))
            result = await run_agent("test prompt", config, console, conversation=conv)

        assert isinstance(result, Conversation)
        assert len(result.messages) >= 2


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
