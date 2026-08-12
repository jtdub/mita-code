"""Extended tests for the core agent loop."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mita.agent.conversation import Conversation, Message, Role
from mita.agent.loop import (
    _extract_usage,
    _process_tool_calls,
    _ResponseStats,
    _try_parse_json_object,
    run_agent,
)
from mita.config.schema import MitaConfig
from mita.tools.registry import create_default_registry


class TestExtractUsage:
    def test_object_with_usage(self) -> None:
        class Usage:
            prompt_tokens = 10
            completion_tokens = 20

        class Chunk:
            usage = Usage()

        result = _extract_usage(Chunk())
        assert result is not None
        assert result["prompt_tokens"] == 10
        assert result["completion_tokens"] == 20

    def test_dict_with_usage(self) -> None:
        chunk = {"usage": {"prompt_tokens": 5, "completion_tokens": 15}}
        result = _extract_usage(chunk)
        assert result is not None
        assert result["prompt_tokens"] == 5

    def test_none_usage(self) -> None:
        class Chunk:
            usage = None

        assert _extract_usage(Chunk()) is None

    def test_no_usage(self) -> None:
        assert _extract_usage({}) is None


class TestTryParseJsonObject:
    def test_valid_json(self) -> None:
        text = '{"key": "value"}'
        obj, end = _try_parse_json_object(text, 0)
        assert obj == {"key": "value"}
        assert end == len(text)

    def test_nested_json(self) -> None:
        text = '{"outer": {"inner": 1}}'
        obj, end = _try_parse_json_object(text, 0)
        assert obj is not None
        assert obj["outer"]["inner"] == 1

    def test_invalid_json(self) -> None:
        text = '{"key": invalid}'
        obj, end = _try_parse_json_object(text, 0)
        assert obj is None

    def test_no_closing_brace(self) -> None:
        text = '{"key": "value"'
        obj, end = _try_parse_json_object(text, 0)
        assert obj is None

    def test_json_with_offset(self) -> None:
        text = 'prefix {"key": "value"}'
        obj, end = _try_parse_json_object(text, 7)
        assert obj == {"key": "value"}

    def test_string_with_braces(self) -> None:
        text = '{"key": "val{ue}"}'
        obj, end = _try_parse_json_object(text, 0)
        assert obj == {"key": "val{ue}"}

    def test_escaped_quotes(self) -> None:
        text = '{"key": "val\\"ue"}'
        obj, end = _try_parse_json_object(text, 0)
        assert obj is not None


class TestResponseStats:
    def test_defaults(self) -> None:
        stats = _ResponseStats()
        assert stats.prompt_tokens == 0
        assert stats.completion_tokens == 0
        assert stats.total_time == 0.0
        assert stats.ttft is None


class TestRunAgentExtended:
    @pytest.mark.asyncio()
    async def test_connection_error_handling(self) -> None:
        """ConnectionError during LLM call is caught."""
        config = MitaConfig()
        config.ui.stream = False
        mock_console = MagicMock()
        mock_console.print = MagicMock()
        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(side_effect=ConnectionError("offline"))
        registry = create_default_registry()

        with patch("mita.agent.loop.assemble_context"):
            conv = Conversation()
            conv.add(Message(role=Role.SYSTEM, content="system"))
            result = await run_agent(
                "test",
                config,
                mock_console,
                conversation=conv,
                registry=registry,
                llm_client=mock_client,
            )
        assert isinstance(result, Conversation)

    @pytest.mark.asyncio()
    async def test_json_decode_error(self) -> None:
        """JSONDecodeError during response parsing is caught."""
        config = MitaConfig()
        config.ui.stream = False
        mock_console = MagicMock()
        mock_console.print = MagicMock()
        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(side_effect=json.JSONDecodeError("err", "", 0))
        registry = create_default_registry()

        with patch("mita.agent.loop.assemble_context"):
            conv = Conversation()
            conv.add(Message(role=Role.SYSTEM, content="system"))
            result = await run_agent(
                "test",
                config,
                mock_console,
                conversation=conv,
                registry=registry,
                llm_client=mock_client,
            )
        assert isinstance(result, Conversation)

    @pytest.mark.asyncio()
    async def test_max_iterations_reached(self) -> None:
        """When max iterations is reached, loop exits."""
        config = MitaConfig()
        config.ui.stream = False
        config.max_iterations = 1
        mock_console = MagicMock()
        mock_console.print = MagicMock()

        # Return a response with tool calls so it wants to continue
        class Func:
            name = "file_read"
            arguments = '{"path": "/tmp/test"}'

        class TC:
            id = "tc1"
            function = Func()

        class Msg:
            content = ""
            tool_calls = [TC()]

        class Choice:
            message = Msg()

        class Response:
            choices = [Choice()]

        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(return_value=Response())
        registry = create_default_registry()

        with (
            patch("mita.agent.loop.assemble_context"),
            patch("mita.agent.loop._process_tool_calls", new_callable=AsyncMock),
        ):
            conv = Conversation()
            conv.add(Message(role=Role.SYSTEM, content="system"))
            result = await run_agent(
                "test",
                config,
                mock_console,
                conversation=conv,
                registry=registry,
                llm_client=mock_client,
            )
        assert isinstance(result, Conversation)

    @pytest.mark.asyncio()
    async def test_repeated_tool_calls_detected(self) -> None:
        """Repeated identical tool calls should trigger abort."""
        config = MitaConfig()
        config.ui.stream = False
        config.max_iterations = 5
        mock_console = MagicMock()
        mock_console.print = MagicMock()

        class Func:
            name = "file_read"
            arguments = '{"path": "/tmp/test"}'

        class TC:
            id = "tc1"
            function = Func()

        class Msg:
            content = "thinking..."
            tool_calls = [TC()]

        class Choice:
            message = Msg()

        class Response:
            choices = [Choice()]

        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(return_value=Response())
        registry = create_default_registry()

        call_count = 0

        async def mock_process(*args: Any, **kwargs: Any) -> None:
            nonlocal call_count
            call_count += 1

        with (
            patch("mita.agent.loop.assemble_context"),
            patch("mita.agent.loop._process_tool_calls", side_effect=mock_process),
        ):
            conv = Conversation()
            conv.add(Message(role=Role.SYSTEM, content="system"))
            await run_agent(
                "test",
                config,
                mock_console,
                conversation=conv,
                registry=registry,
                llm_client=mock_client,
            )
        # Should have detected the repeat after first tool call
        assert call_count <= 2

    @pytest.mark.asyncio()
    async def test_streaming_mode(self) -> None:
        """Streaming mode uses _stream_response."""
        config = MitaConfig()
        config.ui.stream = True
        mock_console = MagicMock()
        mock_console.print = MagicMock()
        mock_client = AsyncMock()

        stats = _ResponseStats(prompt_tokens=5, completion_tokens=10, total_time=1.0)

        with (
            patch("mita.agent.loop.assemble_context"),
            patch(
                "mita.agent.loop._stream_response",
                new_callable=AsyncMock,
                return_value=("streamed answer", [], stats),
            ),
        ):
            conv = Conversation()
            conv.add(Message(role=Role.SYSTEM, content="system"))
            result = await run_agent(
                "test",
                config,
                mock_console,
                conversation=conv,
                registry=create_default_registry(),
                llm_client=mock_client,
            )
        assistant_msgs = [m for m in result.messages if m.role == Role.ASSISTANT]
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0].content == "streamed answer"

    @pytest.mark.asyncio()
    async def test_rag_context_injection(self) -> None:
        """RAG context is injected when index is available."""
        config = MitaConfig()
        config.ui.stream = False
        config.index.enabled = True
        mock_console = MagicMock()
        mock_console.print = MagicMock()

        class Msg:
            content = "answer"
            tool_calls = None

        class Choice:
            message = Msg()

        class Response:
            choices = [Choice()]

        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(return_value=Response())

        mock_retriever = MagicMock()
        mock_retriever.is_available.return_value = True
        mock_retriever.retrieve_formatted = AsyncMock(return_value="relevant code")

        with (
            patch("mita.agent.loop.assemble_context"),
            patch("mita.index.retriever.Retriever", return_value=mock_retriever),
        ):
            conv = Conversation()
            conv.add(Message(role=Role.SYSTEM, content="system"))
            result = await run_agent(
                "test",
                config,
                mock_console,
                conversation=conv,
                registry=create_default_registry(),
                llm_client=mock_client,
            )
        assert isinstance(result, Conversation)

    @pytest.mark.asyncio()
    async def test_hooks_called(self) -> None:
        """Session hooks are called when configured."""
        from mita.config.schema import HookDefinition

        config = MitaConfig()
        config.ui.stream = False
        config.hooks = [HookDefinition(event="session_start", command="echo start")]

        mock_console = MagicMock()
        mock_console.print = MagicMock()

        class Msg:
            content = "answer"
            tool_calls = None

        class Choice:
            message = Msg()

        class Response:
            choices = [Choice()]

        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(return_value=Response())

        with (
            patch("mita.agent.loop.assemble_context"),
            patch("mita.hooks.runner.run_hooks", new_callable=AsyncMock) as mock_hooks,
        ):
            conv = Conversation()
            conv.add(Message(role=Role.SYSTEM, content="system"))
            await run_agent(
                "test",
                config,
                mock_console,
                conversation=conv,
                registry=create_default_registry(),
                llm_client=mock_client,
            )
        # session_start and session_end should both be called
        assert mock_hooks.call_count >= 2


class TestProcessToolCalls:
    def _sink(self, decision: str = "once") -> object:
        from mita.ui.sink import ConfirmDecision, RecordingSink

        sink = RecordingSink()
        sink.confirm_decision = ConfirmDecision(decision)
        return sink

    @pytest.mark.asyncio()
    async def test_processes_tool_call(self) -> None:
        config = MitaConfig()
        conv = Conversation()
        registry = create_default_registry()
        tool_calls_raw = [
            {"id": "tc1", "function": {"name": "file_read", "arguments": '{"path": "/tmp/none"}'}}
        ]
        await _process_tool_calls(tool_calls_raw, conv, registry, config, self._sink())
        tool_msgs = [m for m in conv.messages if m.role == Role.TOOL]
        assert len(tool_msgs) == 1

    @pytest.mark.asyncio()
    async def test_dict_arguments(self) -> None:
        config = MitaConfig()
        conv = Conversation()
        registry = create_default_registry()
        tool_calls_raw = [
            {"id": "tc1", "function": {"name": "file_read", "arguments": {"path": "/tmp/none"}}}
        ]
        await _process_tool_calls(tool_calls_raw, conv, registry, config, self._sink())
        assert len([m for m in conv.messages if m.role == Role.TOOL]) == 1

    @pytest.mark.asyncio()
    async def test_invalid_json_arguments(self) -> None:
        config = MitaConfig()
        conv = Conversation()
        registry = create_default_registry()
        tool_calls_raw = [
            {"id": "tc1", "function": {"name": "file_read", "arguments": "not valid json"}}
        ]
        await _process_tool_calls(tool_calls_raw, conv, registry, config, self._sink())
        assert len([m for m in conv.messages if m.role == Role.TOOL]) == 1


class TestConfirmationAndInterrupt:
    """Audit should-fix: allow-for-session, auto_confirm, interrupt backfill."""

    def _destructive_call(self) -> list[dict]:
        return [
            {
                "id": "tc1",
                "function": {"name": "shell", "arguments": '{"command": "shred /tmp/x"}'},
            }
        ]

    @pytest.mark.asyncio()
    async def test_always_adds_to_session_approved(self) -> None:
        from mita.ui.sink import ConfirmDecision, RecordingSink

        config = MitaConfig()
        conv = Conversation()
        registry = create_default_registry()
        session: set[str] = set()
        sink = RecordingSink()
        sink.confirm_decision = ConfirmDecision.ALLOW_SESSION

        await _process_tool_calls(self._destructive_call(), conv, registry, config, sink, session)
        assert "shell" in session

    @pytest.mark.asyncio()
    async def test_auto_confirm_skips_prompt(self) -> None:
        from mita.ui.sink import RecordingSink

        config = MitaConfig()
        conv = Conversation()
        registry = create_default_registry()
        sink = RecordingSink()  # default decision is DENY

        await _process_tool_calls(
            self._destructive_call(), conv, registry, config, sink, None, auto_confirm=True
        )
        # auto_confirm means the sink is never asked to confirm.
        assert not any(kind == "confirm" for kind, _ in sink.events)

    @pytest.mark.asyncio()
    async def test_interrupt_backfills_tool_results(self) -> None:
        """If a tool call raises mid-batch, every tool_call still gets a TOOL result."""
        from mita.ui.sink import RecordingSink

        config = MitaConfig()
        conv = Conversation()
        registry = create_default_registry()
        calls = [
            {"id": "tc1", "function": {"name": "file_read", "arguments": '{"path": "/tmp/none"}'}},
            {"id": "tc2", "function": {"name": "file_read", "arguments": '{"path": "/tmp/none2"}'}},
        ]

        with patch(
            "mita.agent.loop.execute_tool", new_callable=AsyncMock, side_effect=KeyboardInterrupt
        ):
            with pytest.raises(KeyboardInterrupt):
                await _process_tool_calls(calls, conv, registry, config, RecordingSink())

        tool_ids = {m.tool_call_id for m in conv.messages if m.role == Role.TOOL}
        assert tool_ids == {"tc1", "tc2"}
