"""Extended tests for the core agent loop."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from mita.agent.conversation import Conversation, Role
from mita.agent.loop import (
    _process_tool_calls,
    _tool_calls_to_openai,
    _try_parse_json_object,
    _usage_metadata,
)
from mita.config.schema import MitaConfig
from mita.tools.registry import create_default_registry


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


class TestUsageMetadata:
    def test_extracts_tokens(self) -> None:
        class Msg:
            usage_metadata = {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}

        result = _usage_metadata(Msg())
        assert result == {"prompt_tokens": 10, "completion_tokens": 20}

    def test_none_metadata(self) -> None:
        class Msg:
            usage_metadata = None

        assert _usage_metadata(Msg()) is None


class TestToolCallsToOpenai:
    def test_converts_langchain_shape(self) -> None:
        calls = [{"name": "file_read", "args": {"path": "/tmp/x"}, "id": "call-1"}]
        result = _tool_calls_to_openai(calls)  # type: ignore[arg-type]
        assert result[0]["id"] == "call-1"
        assert result[0]["function"]["name"] == "file_read"
        assert result[0]["function"]["arguments"] == '{"path": "/tmp/x"}'


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
