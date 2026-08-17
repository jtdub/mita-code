"""Tests for the UISink abstraction (TUI enablement)."""

from __future__ import annotations

import pytest
from rich.console import Console

from mita.tools.schema import ToolCall, ToolResult
from mita.ui.sink import (
    ConfirmDecision,
    ConfirmRequest,
    RecordingSink,
    RichConsoleSink,
    UISink,
)


class TestRichConsoleSink:
    def test_is_a_uisink(self) -> None:
        sink = RichConsoleSink(Console())
        assert isinstance(sink, UISink)

    def test_render_methods_do_not_raise(self) -> None:
        sink = RichConsoleSink(Console(force_terminal=True, width=80))
        sink.notice("hi")
        sink.error("bad [/x]")  # markup-unsafe content must not crash
        sink.assistant_message("**bold**")
        sink.tool_call(ToolCall(id="1", name="file_read", arguments={"path": "x"}))
        sink.tool_result(ToolResult(tool_call_id="1", success=True, output="ok"))
        sink.response_stats(10, 20, 1.5, 0.3)
        with sink.busy("working"):
            pass

    @pytest.mark.asyncio()
    async def test_confirm_maps_decision(self) -> None:
        from unittest.mock import AsyncMock, patch

        sink = RichConsoleSink(Console())
        with patch(
            "mita.ui.display.prompt_user_confirm_decision",
            new_callable=AsyncMock,
            return_value="always",
        ):
            decision = await sink.confirm(ConfirmRequest(tool_name="shell", summary="run?"))
        assert decision is ConfirmDecision.ALLOW_SESSION


class TestRecordingSink:
    def test_is_a_uisink(self) -> None:
        assert isinstance(RecordingSink(), UISink)

    def test_records_events(self) -> None:
        sink = RecordingSink()
        sink.notice("n")
        sink.tool_call(ToolCall(id="1", name="grep", arguments={}))
        kinds = [k for k, _ in sink.events]
        assert kinds == ["notice", "tool_call"]

    @pytest.mark.asyncio()
    async def test_confirm_returns_configured_decision(self) -> None:
        sink = RecordingSink()
        sink.confirm_decision = ConfirmDecision.ALLOW_ONCE
        decision = await sink.confirm(ConfirmRequest(tool_name="shell", summary="?"))
        assert decision is ConfirmDecision.ALLOW_ONCE
        assert sink.events[-1][0] == "confirm"
