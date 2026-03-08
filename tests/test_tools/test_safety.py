"""Tests for the safety module."""

from __future__ import annotations

import pytest

from mita.config.schema import ToolSettings
from mita.tools.safety import (
    is_command_banned,
    is_command_destructive,
    is_git_command_destructive,
    needs_confirmation,
)
from mita.tools.schema import ToolCall, ToolDefinition


class TestIsCommandBanned:
    def test_banned_command_detected(self) -> None:
        assert is_command_banned("rm -rf /", ["rm -rf /"]) is True

    def test_banned_substring(self) -> None:
        assert is_command_banned("sudo rm -rf / --no-preserve-root", ["rm -rf /"]) is True

    def test_safe_command(self) -> None:
        assert is_command_banned("ls -la", ["rm -rf /", "mkfs"]) is False

    def test_empty_banned_list(self) -> None:
        assert is_command_banned("rm -rf /", []) is False


class TestIsCommandDestructive:
    @pytest.mark.parametrize(
        "cmd",
        [
            "rm -rf /tmp/test",
            "rm -f important.txt",
            "git push --force",
            "git reset --hard HEAD~1",
            "git clean -fd",
            "shred secret.txt",
        ],
    )
    def test_destructive_commands(self, cmd: str) -> None:
        assert is_command_destructive(cmd) is True

    @pytest.mark.parametrize(
        "cmd",
        [
            "ls -la",
            "git status",
            "cat file.txt",
            "echo hello",
            "python test.py",
            "git log --oneline",
        ],
    )
    def test_safe_commands(self, cmd: str) -> None:
        assert is_command_destructive(cmd) is False


class TestNeedsConfirmation:
    def _make_tool_def(self, destructive: bool = False) -> ToolDefinition:
        return ToolDefinition(
            name="test", description="test", parameters=[], destructive=destructive
        )

    def _make_call(self, name: str = "test", **kwargs: object) -> ToolCall:
        return ToolCall(id="1", name=name, arguments=dict(kwargs))

    def test_non_destructive_tool_no_confirm(self) -> None:
        settings = ToolSettings()
        assert needs_confirmation(self._make_call(), self._make_tool_def(), settings) is False

    def test_destructive_tool_needs_confirm(self) -> None:
        settings = ToolSettings()
        call = self._make_call(name="file_write")
        tool_def = self._make_tool_def(destructive=True)
        assert needs_confirmation(call, tool_def, settings) is True

    def test_auto_approved_tool_no_confirm(self) -> None:
        settings = ToolSettings(auto_approve=["file_write"])
        call = self._make_call(name="file_write")
        tool_def = self._make_tool_def(destructive=True)
        assert needs_confirmation(call, tool_def, settings) is False

    def test_confirm_disabled_globally(self) -> None:
        settings = ToolSettings(confirm_destructive=False)
        call = self._make_call(name="file_write")
        tool_def = self._make_tool_def(destructive=True)
        assert needs_confirmation(call, tool_def, settings) is False

    def test_shell_destructive_command_needs_confirm(self) -> None:
        settings = ToolSettings()
        call = ToolCall(id="1", name="shell", arguments={"command": "rm -rf /tmp"})
        tool_def = ToolDefinition(
            name="shell", description="shell", parameters=[], destructive=False
        )
        assert needs_confirmation(call, tool_def, settings) is True

    def test_git_destructive_subcommand_needs_confirm(self) -> None:
        settings = ToolSettings()
        call = ToolCall(id="1", name="git", arguments={"subcommand": "push --force"})
        tool_def = ToolDefinition(name="git", description="git", parameters=[], destructive=False)
        assert needs_confirmation(call, tool_def, settings) is True

    def test_git_safe_subcommand_no_confirm(self) -> None:
        settings = ToolSettings()
        call = ToolCall(id="1", name="git", arguments={"subcommand": "status"})
        tool_def = ToolDefinition(name="git", description="git", parameters=[], destructive=False)
        assert needs_confirmation(call, tool_def, settings) is False


class TestIsGitCommandDestructive:
    @pytest.mark.parametrize(
        "subcmd",
        [
            "push --force origin main",
            "push -f",
            "reset --hard HEAD~1",
            "clean -fd",
            "checkout -- .",
            "branch -D feature",
            "branch -d feature",
        ],
    )
    def test_destructive_git_commands(self, subcmd: str) -> None:
        assert is_git_command_destructive(subcmd) is True

    @pytest.mark.parametrize(
        "subcmd",
        [
            "status",
            "diff",
            "log --oneline -10",
            "add .",
            "commit -m 'test'",
            "push origin main",
            "branch feature",
        ],
    )
    def test_safe_git_commands(self, subcmd: str) -> None:
        assert is_git_command_destructive(subcmd) is False
