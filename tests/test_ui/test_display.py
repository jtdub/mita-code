"""Tests for UI display functions."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from rich.console import Console

from mita.tools.schema import ToolCall, ToolResult
from mita.ui.display import (
    display_error,
    display_goodbye,
    display_tool_call,
    display_tool_result,
    display_warning,
    display_welcome,
    get_console,
    prompt_user_confirm,
)
from mita.ui.theme import MITA_THEME


def _make_console() -> Console:
    return Console(force_terminal=True, width=80, theme=MITA_THEME)


class TestGetConsole:
    def test_returns_console(self) -> None:
        console = get_console()
        assert console is not None

    def test_returns_same_instance(self) -> None:
        c1 = get_console()
        c2 = get_console()
        assert c1 is c2


class TestDisplayFunctions:
    def test_display_welcome(self, capsys: pytest.CaptureFixture[str]) -> None:
        console = _make_console()
        display_welcome(console)
        output = capsys.readouterr().out
        assert "mita" in output

    def test_display_goodbye(self, capsys: pytest.CaptureFixture[str]) -> None:
        console = _make_console()
        display_goodbye(console)
        output = capsys.readouterr().out
        assert "Goodbye" in output

    def test_display_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        console = _make_console()
        display_error(console, "something broke")
        output = capsys.readouterr().out
        assert "something broke" in output

    def test_display_warning(self, capsys: pytest.CaptureFixture[str]) -> None:
        console = _make_console()
        display_warning(console, "be careful")
        output = capsys.readouterr().out
        assert "be careful" in output

    def test_display_tool_call(self, capsys: pytest.CaptureFixture[str]) -> None:
        console = _make_console()
        tc = ToolCall(id="1", name="file_read", arguments={"path": "/tmp/test"})
        display_tool_call(console, tc)
        output = capsys.readouterr().out
        assert "file_read" in output

    def test_display_tool_result_success(self, capsys: pytest.CaptureFixture[str]) -> None:
        console = _make_console()
        result = ToolResult(tool_call_id="1", success=True, output="done")
        display_tool_result(console, result)
        output = capsys.readouterr().out
        assert "done" in output

    def test_display_tool_result_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        console = _make_console()
        result = ToolResult(tool_call_id="1", success=False, error="failed")
        display_tool_result(console, result)
        output = capsys.readouterr().out
        assert "failed" in output


class TestPromptUserConfirm:
    @pytest.mark.asyncio()
    async def test_approve(self) -> None:
        console = MagicMock()
        console.input = MagicMock(return_value="y")
        result = await prompt_user_confirm(console, "Allow?")
        assert result is True

    @pytest.mark.asyncio()
    async def test_deny(self) -> None:
        console = MagicMock()
        console.input = MagicMock(return_value="n")
        result = await prompt_user_confirm(console, "Allow?")
        assert result is False

    @pytest.mark.asyncio()
    async def test_default_deny(self) -> None:
        console = MagicMock()
        console.input = MagicMock(return_value="")
        result = await prompt_user_confirm(console, "Allow?")
        assert result is False

    @pytest.mark.asyncio()
    async def test_eof_denies(self) -> None:
        console = MagicMock()
        console.input = MagicMock(side_effect=EOFError)
        console.print = MagicMock()
        result = await prompt_user_confirm(console, "Allow?")
        assert result is False


class TestMarkupInjection:
    """Audit finding: model/tool output containing '[/x]' must not crash rendering."""

    def test_tool_result_output_with_markup(self) -> None:
        console = Console(force_terminal=True, width=80, theme=MITA_THEME)
        result = ToolResult(tool_call_id="1", success=True, output="see [/dim] and [bold]x")
        display_tool_result(console, result)  # must not raise MarkupError

    def test_tool_result_error_with_markup(self) -> None:
        console = Console(force_terminal=True, width=80, theme=MITA_THEME)
        result = ToolResult(tool_call_id="1", success=False, error="bad [/red] token")
        display_tool_result(console, result)

    def test_tool_call_args_with_markup(self) -> None:
        console = Console(force_terminal=True, width=80, theme=MITA_THEME)
        call = ToolCall(id="1", name="mcp:srv/do", arguments={"pattern": "[/x]abc["})
        display_tool_call(console, call)

    def test_error_and_warning_with_markup(self) -> None:
        console = Console(force_terminal=True, width=80, theme=MITA_THEME)
        display_error(console, "failed on [/tag]")
        display_warning(console, "watch [/out]")
