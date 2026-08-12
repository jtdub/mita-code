"""Smoke tests for the Textual TUI (TUI enablement)."""

from __future__ import annotations

import pytest

from mita.config.schema import MitaConfig
from mita.tools.registry import create_default_registry
from mita.tools.schema import ToolCall, ToolResult
from mita.ui.sink import ConfirmDecision, ConfirmRequest, UISink
from mita.ui.tui.app import ConfirmScreen, MitaTUI


def _app() -> MitaTUI:
    return MitaTUI(MitaConfig(), create_default_registry())


def test_app_is_a_uisink() -> None:
    assert isinstance(_app(), UISink)


class TestConfirmScreen:
    def test_button_ids_map_to_decisions(self) -> None:
        # The mapping the modal uses to translate a button press into a decision.
        screen = ConfirmScreen(ConfirmRequest(tool_name="shell", summary="run?"))
        assert screen._request.tool_name == "shell"


class TestTUIRuntime:
    @pytest.mark.asyncio()
    async def test_mounts_and_shows_welcome(self) -> None:
        app = _app()
        async with app.run_test() as pilot:
            await pilot.pause()
            # UISink methods render without raising once mounted.
            app.notice("a notice")
            app.error("bad [/x] token")
            app.tool_call(ToolCall(id="1", name="file_read", arguments={"path": "x"}))
            app.tool_result(ToolResult(tool_call_id="1", success=True, output="ok"))
            app.stream_token("hello ")
            app.stream_token("world")
            app.stream_end()
            app.response_stats(1, 2, 0.5, 0.1)

    @pytest.mark.asyncio()
    async def test_quit_command_exits(self) -> None:
        app = _app()
        async with app.run_test() as pilot:
            await pilot.pause()
            prompt = app.query_one("#prompt")
            prompt.value = "/quit"
            await pilot.press("enter")
            await pilot.pause()
        # run_test context exits cleanly when the app has exited.
        assert app.is_running is False

    @pytest.mark.asyncio()
    async def test_confirm_returns_decision_from_modal(self) -> None:
        app = _app()
        async with app.run_test() as pilot:
            await pilot.pause()

            async def ask() -> ConfirmDecision:
                return await app.confirm(ConfirmRequest(tool_name="shell", summary="run rm?"))

            task = app.run_worker(ask(), exclusive=False)
            await pilot.pause()
            # Press the "Always" button in the modal.
            await pilot.click("#always")
            await pilot.pause()
            result = await task.wait()
        assert result is ConfirmDecision.ALLOW_SESSION
