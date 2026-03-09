"""Tests for the mita ask command (non-interactive mode)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

from typer.testing import CliRunner

from mita.agent.conversation import Conversation, Message, Role
from mita.cli import OutputFormat, app

runner = CliRunner()


class TestOutputFormatEnum:
    def test_values(self) -> None:
        assert OutputFormat.RICH == "rich"
        assert OutputFormat.TEXT == "text"
        assert OutputFormat.JSON == "json"


def _make_conv(response: str) -> Conversation:
    conv = Conversation()
    conv.add(Message(role=Role.SYSTEM, content="system"))
    conv.add(Message(role=Role.USER, content="test"))
    conv.add(Message(role=Role.ASSISTANT, content=response))
    return conv


class TestAskStdinDetection:
    def test_stdin_pipe(self) -> None:
        """When stdin is not a TTY, read prompt from stdin."""
        conv = _make_conv("piped response")

        with (
            patch("mita.cli.sys") as mock_sys,
            patch(
                "mita.agent.loop.run_agent",
                new_callable=AsyncMock,
                return_value=conv,
            ),
            patch("mita.models.server.ensure_server", return_value=True),
            patch("mita.models.server.ensure_model", return_value=True),
        ):
            mock_sys.stdin.isatty.return_value = False
            mock_sys.stdin.read.return_value = "hello from pipe"
            mock_sys.stdout.isatty.return_value = True
            result = runner.invoke(app, ["ask", "--output", "text"])
            assert result.exit_code == 0

    def test_no_input_error(self) -> None:
        """When no prompt and stdin is a TTY, show error."""
        with patch("mita.cli.sys") as mock_sys:
            mock_sys.stdin.isatty.return_value = True
            mock_sys.stdout.isatty.return_value = True
            result = runner.invoke(app, ["ask"])
            assert result.exit_code != 0
            assert "No prompt" in result.output

    def test_arg_and_stdin_concatenated(self) -> None:
        """When both arg and stdin are provided, they are concatenated."""
        conv = _make_conv("combined")

        with (
            patch("mita.cli.sys") as mock_sys,
            patch(
                "mita.agent.loop.run_agent",
                new_callable=AsyncMock,
                return_value=conv,
            ),
            patch("mita.models.server.ensure_server", return_value=True),
            patch("mita.models.server.ensure_model", return_value=True),
        ):
            mock_sys.stdin.isatty.return_value = False
            mock_sys.stdin.read.return_value = "stdin part"
            mock_sys.stdout.isatty.return_value = True
            result = runner.invoke(app, ["ask", "arg part", "--output", "text"])
            assert result.exit_code == 0


class TestAskOutputModes:
    def test_output_text(self) -> None:
        """--output text prints plain text to stdout."""
        conv = _make_conv("plain text answer")

        with (
            patch("mita.cli.sys") as mock_sys,
            patch(
                "mita.agent.loop.run_agent",
                new_callable=AsyncMock,
                return_value=conv,
            ),
            patch("mita.models.server.ensure_server", return_value=True),
            patch("mita.models.server.ensure_model", return_value=True),
        ):
            mock_sys.stdin.isatty.return_value = True
            mock_sys.stdout.isatty.return_value = True
            result = runner.invoke(app, ["ask", "test prompt", "--output", "text"])
            assert result.exit_code == 0
            assert "plain text answer" in result.output

    def test_output_json(self) -> None:
        """--output json prints JSON with response and model."""
        conv = _make_conv("json answer")

        with (
            patch("mita.cli.sys") as mock_sys,
            patch(
                "mita.agent.loop.run_agent",
                new_callable=AsyncMock,
                return_value=conv,
            ),
            patch("mita.models.server.ensure_server", return_value=True),
            patch("mita.models.server.ensure_model", return_value=True),
        ):
            mock_sys.stdin.isatty.return_value = True
            mock_sys.stdout.isatty.return_value = True
            result = runner.invoke(app, ["ask", "test prompt", "--output", "json"])
            assert result.exit_code == 0
            data = json.loads(result.output.strip())
            assert data["response"] == "json answer"
            assert "model" in data


class TestAskNoTools:
    def test_no_tools_flag(self) -> None:
        """--no-tools passes empty registry."""
        conv = _make_conv("no tools")

        captured_registry = None

        async def mock_run_agent(
            prompt: str, cfg: object, console: object, **kwargs: object
        ) -> Conversation:
            nonlocal captured_registry
            captured_registry = kwargs.get("registry")
            return conv

        with (
            patch("mita.cli.sys") as mock_sys,
            patch("mita.agent.loop.run_agent", side_effect=mock_run_agent),
            patch("mita.models.server.ensure_server", return_value=True),
            patch("mita.models.server.ensure_model", return_value=True),
        ):
            mock_sys.stdin.isatty.return_value = True
            mock_sys.stdout.isatty.return_value = True
            result = runner.invoke(app, ["ask", "hello", "--no-tools", "--output", "text"])
            assert result.exit_code == 0
            assert captured_registry is not None
            assert len(captured_registry.tool_names) == 0
