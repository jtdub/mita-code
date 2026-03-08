"""Tests for the tool executor."""

from __future__ import annotations

import pytest

from mita.config.schema import ToolSettings
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
