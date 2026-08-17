"""Tests for the safety module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from mita.config.schema import ToolSettings
from mita.tools.safety import (
    _extract_command_names,
    is_command_banned,
    is_command_destructive,
    is_git_command_destructive,
    needs_confirmation,
    validate_path_for_read,
    validate_path_for_write,
    validate_search_base,
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

    def test_piped_command_detected(self) -> None:
        """Banned command in a pipeline should be caught."""
        assert is_command_banned("find / -type f | xargs rm", ["rm"]) is True

    def test_chained_command_detected(self) -> None:
        """Banned command after && should be caught."""
        assert is_command_banned("echo hello && rm -rf /", ["rm"]) is True

    def test_sudo_prefix_detected(self) -> None:
        """Banned command after sudo should be caught."""
        assert is_command_banned("sudo rm -rf /", ["rm"]) is True

    def test_env_prefix_detected(self) -> None:
        """Banned command after env vars should be caught."""
        assert is_command_banned("FOO=bar rm -rf /", ["rm"]) is True

    def test_command_name_matching(self) -> None:
        """Banning 'mkfs' should catch mkfs with any flags."""
        assert is_command_banned("mkfs -t ext4 /dev/sda1", ["mkfs"]) is True

    def test_command_name_no_false_positive(self) -> None:
        """Banning 'rm' should not catch 'grep' just because 'rm' is a substring."""
        assert is_command_banned("grep 'rm -rf' logfile.txt", ["rm -rf /"]) is False


class TestExtractCommandNames:
    def test_simple_command(self) -> None:
        assert _extract_command_names("ls -la") == {"ls"}

    def test_piped_commands(self) -> None:
        assert _extract_command_names("cat file | grep pattern") == {"cat", "grep"}

    def test_chained_commands(self) -> None:
        assert _extract_command_names("cd /tmp && rm -rf *") == {"cd", "rm"}

    def test_sudo_prefix(self) -> None:
        assert _extract_command_names("sudo rm -rf /") == {"rm"}

    def test_env_var_prefix(self) -> None:
        assert _extract_command_names("FOO=bar python script.py") == {"python"}


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

    @pytest.mark.parametrize("subcmd", ["status", "diff", "log --oneline", "show", "branch --list"])
    def test_git_read_only_commands_skip_confirm(self, subcmd: str) -> None:
        """Read-only git subcommands should not require confirmation (#58)."""
        settings = ToolSettings()
        call = ToolCall(id="1", name="git", arguments={"subcommand": subcmd})
        tool_def = ToolDefinition(name="git", description="git", parameters=[], destructive=False)
        assert needs_confirmation(call, tool_def, settings) is False

    @pytest.mark.parametrize("subcmd", ["branch -D feature", "branch -d feature"])
    def test_git_destructive_branch_still_confirms(self, subcmd: str) -> None:
        """Destructive branch commands must require confirmation even though
        'branch' is in SAFE_SUBCOMMANDS — destructive check runs first."""
        settings = ToolSettings()
        call = ToolCall(id="1", name="git", arguments={"subcommand": subcmd})
        tool_def = ToolDefinition(name="git", description="git", parameters=[], destructive=False)
        assert needs_confirmation(call, tool_def, settings) is True

    def test_auto_edit_mode_approves_writes(self) -> None:
        """In auto_edit mode, file_write should not require confirmation."""
        from mita.config.schema import PermissionMode

        settings = ToolSettings(permission_mode=PermissionMode.AUTO_EDIT)
        call = self._make_call(name="file_write")
        tool_def = self._make_tool_def(destructive=True)
        assert needs_confirmation(call, tool_def, settings) is False

    def test_auto_edit_mode_still_confirms_shell(self) -> None:
        """In auto_edit mode, shell should still require confirmation."""
        from mita.config.schema import PermissionMode

        settings = ToolSettings(permission_mode=PermissionMode.AUTO_EDIT)
        call = ToolCall(id="1", name="shell", arguments={"command": "rm -rf /tmp"})
        tool_def = ToolDefinition(
            name="shell", description="shell", parameters=[], destructive=True
        )
        assert needs_confirmation(call, tool_def, settings) is True

    def test_trust_mode_still_confirms_destructive_shell(self) -> None:
        """Even in trust mode, a destructive shell command must still confirm.

        Auto-approving 'shell' covers ordinary commands, not `rm -rf`.
        """
        from mita.config.schema import PermissionMode

        settings = ToolSettings(permission_mode=PermissionMode.TRUST)
        call = ToolCall(id="1", name="shell", arguments={"command": "rm -rf /"})
        tool_def = ToolDefinition(
            name="shell", description="shell", parameters=[], destructive=True
        )
        assert needs_confirmation(call, tool_def, settings) is True

    def test_trust_mode_approves_safe_shell(self) -> None:
        """Trust mode still auto-approves ordinary (non-destructive) shell commands."""
        from mita.config.schema import PermissionMode

        settings = ToolSettings(permission_mode=PermissionMode.TRUST)
        call = ToolCall(id="1", name="shell", arguments={"command": "ls -la"})
        tool_def = ToolDefinition(
            name="shell", description="shell", parameters=[], destructive=True
        )
        assert needs_confirmation(call, tool_def, settings) is False

    @pytest.mark.parametrize("mode", ["auto_edit", "trust"])
    @pytest.mark.parametrize(
        "subcmd",
        ["reset --hard HEAD~1", "push --force origin main", "clean -fdx", "branch -D main"],
    )
    def test_destructive_git_confirms_even_when_auto_approved(self, mode: str, subcmd: str) -> None:
        """auto_edit/trust auto-approve 'git', but destructive subcommands still confirm."""
        from mita.config.schema import PermissionMode

        settings = ToolSettings(permission_mode=PermissionMode(mode))
        call = ToolCall(id="1", name="git", arguments={"subcommand": subcmd})
        tool_def = ToolDefinition(name="git", description="git", parameters=[], destructive=False)
        assert needs_confirmation(call, tool_def, settings) is True

    @pytest.mark.parametrize("mode", ["auto_edit", "trust"])
    def test_safe_git_auto_approved_in_permission_modes(self, mode: str) -> None:
        """A read-only git subcommand is still auto-approved in auto_edit/trust."""
        from mita.config.schema import PermissionMode

        settings = ToolSettings(permission_mode=PermissionMode(mode))
        call = ToolCall(id="1", name="git", arguments={"subcommand": "status"})
        tool_def = ToolDefinition(name="git", description="git", parameters=[], destructive=False)
        assert needs_confirmation(call, tool_def, settings) is False

    def test_session_approved_skips_confirm(self) -> None:
        """Tools in session_approved set should skip confirmation."""
        settings = ToolSettings()
        call = self._make_call(name="file_write")
        tool_def = self._make_tool_def(destructive=True)
        result = needs_confirmation(call, tool_def, settings, session_approved={"file_write"})
        assert result is False

    def test_session_approved_does_not_affect_other_tools(self) -> None:
        """Session approval for one tool should not approve another."""
        settings = ToolSettings()
        call = self._make_call(name="danger_tool")
        tool_def = self._make_tool_def(destructive=True)
        assert needs_confirmation(call, tool_def, settings, session_approved={"file_write"}) is True

    def test_session_approved_none_has_no_effect(self) -> None:
        """Passing session_approved=None should behave like no session approvals."""
        settings = ToolSettings()
        call = self._make_call(name="file_write")
        tool_def = self._make_tool_def(destructive=True)
        assert needs_confirmation(call, tool_def, settings, session_approved=None) is True


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


class TestValidatePathForRead:
    def test_path_inside_workspace(self, tmp_path: Path) -> None:
        target = tmp_path / "file.txt"
        target.touch()
        with patch("mita.tools.safety._workspace_root_override", tmp_path):
            assert validate_path_for_read(target.resolve()) is None

    def test_path_outside_workspace_blocked(self, tmp_path: Path) -> None:
        workspace = tmp_path / "project"
        workspace.mkdir()
        outside = tmp_path / "outside" / "secret.txt"
        outside.parent.mkdir()
        outside.touch()
        with patch("mita.tools.safety._workspace_root_override", workspace):
            error = validate_path_for_read(outside.resolve())
            assert error is not None
            assert "outside the workspace" in error

    def test_sensitive_path_blocked(self, tmp_path: Path) -> None:
        ssh_key = Path.home() / ".ssh" / "id_rsa"
        with patch("mita.tools.safety._workspace_root_override", tmp_path):
            error = validate_path_for_read(ssh_key)
            assert error is not None
            assert "sensitive" in error

    def test_etc_passwd_blocked(self, tmp_path: Path) -> None:
        with patch("mita.tools.safety._workspace_root_override", tmp_path):
            error = validate_path_for_read(Path("/etc/passwd"))
            assert error is not None
            assert "outside the workspace" in error


class TestValidatePathForWrite:
    def test_path_inside_workspace(self, tmp_path: Path) -> None:
        target = tmp_path / "new_file.txt"
        with patch("mita.tools.safety._workspace_root_override", tmp_path):
            assert validate_path_for_write(target.resolve()) is None

    def test_path_outside_workspace_blocked(self, tmp_path: Path) -> None:
        workspace = tmp_path / "project"
        workspace.mkdir()
        outside = tmp_path / "outside" / "file.txt"
        with patch("mita.tools.safety._workspace_root_override", workspace):
            error = validate_path_for_write(outside.resolve())
            assert error is not None
            assert "outside the workspace" in error

    def test_sensitive_path_blocked(self, tmp_path: Path) -> None:
        ssh_key = Path.home() / ".ssh" / "id_rsa"
        with patch("mita.tools.safety._workspace_root_override", tmp_path):
            error = validate_path_for_write(ssh_key)
            assert error is not None
            assert "sensitive" in error


class TestValidateSearchBase:
    def test_base_inside_workspace(self, tmp_path: Path) -> None:
        with patch("mita.tools.safety._workspace_root_override", tmp_path):
            assert validate_search_base(tmp_path.resolve()) is None

    def test_base_outside_workspace_blocked(self, tmp_path: Path) -> None:
        workspace = tmp_path / "project"
        workspace.mkdir()
        with patch("mita.tools.safety._workspace_root_override", workspace):
            error = validate_search_base(tmp_path.resolve())
            assert error is not None
            assert "outside the workspace" in error
