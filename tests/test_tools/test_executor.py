"""Tests for the tool executor."""

from __future__ import annotations

import pytest

from mita.config.schema import PermissionMode, ToolSettings
from mita.tools.executor import execute_tool
from mita.tools.registry import ToolRegistry
from mita.tools.schema import ToolCall, ToolDefinition, ToolResult


async def _echo_handler(args: dict[str, object]) -> ToolResult:
    return ToolResult(tool_call_id="", success=True, output=str(args))


@pytest.fixture()
def registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(
        ToolDefinition(name="safe_tool", description="safe", parameters=[], destructive=False),
        _echo_handler,
    )
    reg.register(
        ToolDefinition(name="danger_tool", description="danger", parameters=[], destructive=True),
        _echo_handler,
    )
    reg.register(
        ToolDefinition(name="shell", description="shell", parameters=[], destructive=True),
        _echo_handler,
    )
    reg.register(
        ToolDefinition(name="git", description="git", parameters=[], destructive=False),
        _echo_handler,
    )
    return reg


class TestExecuteTool:
    @pytest.mark.asyncio()
    async def test_safe_tool_executes(self, registry: ToolRegistry) -> None:
        call = ToolCall(id="1", name="safe_tool", arguments={"x": 1})
        result = await execute_tool(call, registry, ToolSettings())
        assert result.success is True

    @pytest.mark.asyncio()
    async def test_unknown_tool(self, registry: ToolRegistry) -> None:
        call = ToolCall(id="1", name="nonexistent", arguments={})
        result = await execute_tool(call, registry, ToolSettings())
        assert result.success is False
        assert "Unknown tool" in (result.error or "")

    @pytest.mark.asyncio()
    async def test_destructive_denied_without_confirm_fn(self, registry: ToolRegistry) -> None:
        call = ToolCall(id="1", name="danger_tool", arguments={})
        result = await execute_tool(call, registry, ToolSettings())
        assert result.success is False
        assert "confirmation" in (result.error or "").lower()

    @pytest.mark.asyncio()
    async def test_destructive_approved(self, registry: ToolRegistry) -> None:
        call = ToolCall(id="1", name="danger_tool", arguments={})

        async def approve(_prompt: str) -> bool:
            return True

        result = await execute_tool(call, registry, ToolSettings(), confirm_fn=approve)
        assert result.success is True

    @pytest.mark.asyncio()
    async def test_destructive_denied_by_user(self, registry: ToolRegistry) -> None:
        call = ToolCall(id="1", name="danger_tool", arguments={})

        async def deny(_prompt: str) -> bool:
            return False

        result = await execute_tool(call, registry, ToolSettings(), confirm_fn=deny)
        assert result.success is False
        assert "denied" in (result.error or "").lower()

    @pytest.mark.asyncio()
    async def test_banned_command_blocked(self, registry: ToolRegistry) -> None:
        call = ToolCall(id="1", name="shell", arguments={"command": "rm -rf /"})
        settings = ToolSettings(banned_commands=["rm -rf /"])
        result = await execute_tool(call, registry, settings)
        assert result.success is False
        assert "banned" in (result.error or "").lower()

    @pytest.mark.asyncio()
    async def test_auto_approve_skips_confirm(self, registry: ToolRegistry) -> None:
        call = ToolCall(id="1", name="danger_tool", arguments={})
        settings = ToolSettings(auto_approve=["danger_tool"])
        result = await execute_tool(call, registry, settings)
        assert result.success is True

    @pytest.mark.asyncio()
    async def test_git_banned_command_blocked(self, registry: ToolRegistry) -> None:
        call = ToolCall(id="1", name="git", arguments={"subcommand": "push --force"})
        settings = ToolSettings(banned_commands=["git push --force"])
        result = await execute_tool(call, registry, settings)
        assert result.success is False
        assert "banned" in (result.error or "").lower()

    @pytest.mark.asyncio()
    async def test_git_safe_command_executes(self, registry: ToolRegistry) -> None:
        call = ToolCall(id="1", name="git", arguments={"subcommand": "status"})
        result = await execute_tool(call, registry, ToolSettings())
        assert result.success is True

    @pytest.mark.asyncio()
    async def test_auto_edit_mode_approves_writes(self, registry: ToolRegistry) -> None:
        call = ToolCall(id="1", name="danger_tool", arguments={})
        settings = ToolSettings(
            permission_mode=PermissionMode.AUTO_EDIT,
            auto_approve=["danger_tool"],
        )
        result = await execute_tool(call, registry, settings)
        assert result.success is True

    @pytest.mark.asyncio()
    async def test_trust_mode_approves_shell(self, registry: ToolRegistry) -> None:
        call = ToolCall(id="1", name="shell", arguments={"command": "echo hello"})
        settings = ToolSettings(permission_mode=PermissionMode.TRUST)
        result = await execute_tool(call, registry, settings)
        assert result.success is True

    @pytest.mark.asyncio()
    async def test_session_approved_skips_confirm(self, registry: ToolRegistry) -> None:
        call = ToolCall(id="1", name="danger_tool", arguments={})
        result = await execute_tool(
            call, registry, ToolSettings(), session_approved={"danger_tool"}
        )
        assert result.success is True

    @pytest.mark.asyncio()
    async def test_glob_receives_config_max_results(self, registry: ToolRegistry) -> None:
        """Executor injects _max_results from config into glob args."""
        captured_args: dict[str, object] = {}

        async def capture_handler(args: dict[str, object]) -> ToolResult:
            captured_args.update(args)
            return ToolResult(tool_call_id="", success=True, output="ok")

        reg = ToolRegistry()
        reg.register(
            ToolDefinition(name="glob", description="glob", parameters=[], destructive=False),
            capture_handler,
        )
        call = ToolCall(id="1", name="glob", arguments={"pattern": "*.py"})
        settings = ToolSettings(glob_max_results=42)
        await execute_tool(call, reg, settings)
        assert captured_args.get("_max_results") == 42

    @pytest.mark.asyncio()
    async def test_grep_receives_config_max_matches(self, registry: ToolRegistry) -> None:
        """Executor injects _max_matches from config into grep args."""
        captured_args: dict[str, object] = {}

        async def capture_handler(args: dict[str, object]) -> ToolResult:
            captured_args.update(args)
            return ToolResult(tool_call_id="", success=True, output="ok")

        reg = ToolRegistry()
        reg.register(
            ToolDefinition(name="grep", description="grep", parameters=[], destructive=False),
            capture_handler,
        )
        call = ToolCall(id="1", name="grep", arguments={"pattern": "test"})
        settings = ToolSettings(grep_max_matches=99)
        await execute_tool(call, reg, settings)
        assert captured_args.get("_max_matches") == 99


class TestCoreBannedAndShellInjection:
    """Audit finding S11 + shell confinement."""

    @pytest.mark.asyncio()
    async def test_core_banned_survives_config_override(self, registry: ToolRegistry) -> None:
        # User replaces banned_commands, dropping the built-in guard from config.
        settings = ToolSettings(banned_commands=["harmless"])
        result = await execute_tool(
            ToolCall(id="1", name="shell", arguments={"command": "rm -rf /"}),
            registry,
            settings,
        )
        assert result.success is False
        assert "banned" in (result.error or "").lower()

    @pytest.mark.asyncio()
    async def test_shell_gets_cwd_and_timeout_cap(self, registry: ToolRegistry) -> None:
        # The echo handler returns str(args); assert the executor injected confinement.
        settings = ToolSettings(shell_timeout=42, permission_mode=PermissionMode.TRUST)
        call = ToolCall(id="1", name="shell", arguments={"command": "ls"})
        result = await execute_tool(call, registry, settings)
        assert result.success is True
        assert "_cwd" in call.arguments
        assert call.arguments["_timeout_cap"] == 42
