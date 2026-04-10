"""Tests for the git built-in tool."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mita.tools.builtins.git import TOOL_DEF, execute, is_safe_git_command


class TestToolDef:
    def test_name(self) -> None:
        assert TOOL_DEF.name == "git"

    def test_not_destructive(self) -> None:
        assert TOOL_DEF.destructive is False


class TestIsSafeGitCommand:
    def test_safe_commands(self) -> None:
        for cmd in ["status", "diff", "log", "show", "branch", "remote", "tag"]:
            assert is_safe_git_command(cmd) is True

    def test_safe_with_args(self) -> None:
        assert is_safe_git_command("log --oneline -10") is True
        assert is_safe_git_command("diff HEAD~1") is True

    def test_unsafe_commands(self) -> None:
        assert is_safe_git_command("push origin main") is False
        assert is_safe_git_command("reset --hard HEAD") is False
        assert is_safe_git_command("commit -m 'test'") is False
        assert is_safe_git_command("add .") is False
        assert is_safe_git_command("checkout -b feature") is False

    def test_empty(self) -> None:
        assert is_safe_git_command("") is False


class TestGitExecute:
    @pytest.mark.asyncio()
    async def test_missing_subcommand(self) -> None:
        result = await execute({})
        assert result.success is False
        assert "Missing" in (result.error or "")

    @pytest.mark.asyncio()
    async def test_empty_subcommand(self) -> None:
        result = await execute({"subcommand": ""})
        assert result.success is False

    @pytest.mark.asyncio()
    async def test_invalid_syntax(self) -> None:
        result = await execute({"subcommand": "log 'unclosed"})
        assert result.success is False
        assert "syntax" in (result.error or "").lower()

    @pytest.mark.asyncio()
    async def test_successful_command(self) -> None:
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"output here\n", b""))
        mock_proc.returncode = 0

        patch_target = "mita.tools.builtins.git.asyncio.create_subprocess_exec"
        with patch(patch_target, return_value=mock_proc):
            result = await execute({"subcommand": "status"})
            assert result.success is True
            assert "output here" in result.output

    @pytest.mark.asyncio()
    async def test_failed_command(self) -> None:
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b"error msg\n"))
        mock_proc.returncode = 1

        patch_target = "mita.tools.builtins.git.asyncio.create_subprocess_exec"
        with patch(patch_target, return_value=mock_proc):
            result = await execute({"subcommand": "log --bad-flag"})
            assert result.success is False

    @pytest.mark.asyncio()
    async def test_timeout(self) -> None:
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(side_effect=TimeoutError)
        mock_proc.terminate = MagicMock()
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock()
        mock_proc.returncode = None

        patch_target = "mita.tools.builtins.git.asyncio.create_subprocess_exec"
        with patch(patch_target, return_value=mock_proc):
            result = await execute({"subcommand": "log"})
            assert result.success is False
            assert "timed out" in (result.error or "").lower()

    @pytest.mark.asyncio()
    async def test_os_error(self) -> None:
        with patch(
            "mita.tools.builtins.git.asyncio.create_subprocess_exec",
            side_effect=OSError("git not found"),
        ):
            result = await execute({"subcommand": "status"})
            assert result.success is False
            assert "Failed" in (result.error or "")
