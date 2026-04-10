"""Tool dispatch: confirmation flow and execution."""

from __future__ import annotations

import logging
from collections.abc import Callable, Coroutine
from typing import Any

from mita.config.schema import ToolSettings
from mita.tools.registry import ToolRegistry
from mita.tools.safety import is_command_banned, needs_confirmation
from mita.tools.schema import ToolCall, ToolResult

_logger = logging.getLogger(__name__)

ConfirmFn = Callable[[str], Coroutine[Any, Any, bool]]


async def execute_tool(
    tool_call: ToolCall,
    registry: ToolRegistry,
    settings: ToolSettings,
    confirm_fn: ConfirmFn | None = None,
    session_approved: set[str] | None = None,
) -> ToolResult:
    """Execute a tool call with safety checks and optional confirmation.

    Args:
        tool_call: The parsed tool call from LLM output.
        registry: The tool registry to look up handlers.
        settings: Tool settings (auto_approve, banned_commands, etc.).
        confirm_fn: Optional async callable(str) -> bool for user confirmation.
                     If None, destructive tools are denied automatically.
        session_approved: Set of tool names approved for the current session.
    """
    tool_def = registry.get_definition(tool_call.name)
    if tool_def is None:
        return ToolResult(
            tool_call_id=tool_call.id,
            success=False,
            error=f"Unknown tool: {tool_call.name}",
        )

    # Check for banned commands (shell and git tools)
    if tool_call.name == "shell":
        command = tool_call.arguments.get("command", "")
        if is_command_banned(command, settings.banned_commands):
            return ToolResult(
                tool_call_id=tool_call.id,
                success=False,
                error=f"Command is banned by configuration: {command}",
            )
    if tool_call.name == "git":
        subcommand = tool_call.arguments.get("subcommand", "")
        if is_command_banned(f"git {subcommand}", settings.banned_commands):
            return ToolResult(
                tool_call_id=tool_call.id,
                success=False,
                error=f"Git command is banned by configuration: git {subcommand}",
            )

    # Confirmation flow
    if needs_confirmation(tool_call, tool_def, settings, session_approved=session_approved):
        if confirm_fn is None:
            return ToolResult(
                tool_call_id=tool_call.id,
                success=False,
                error="Destructive action requires confirmation, but no confirmation handler.",
            )

        prompt = f"Allow {tool_call.name}({_summarize_args(tool_call)})?"
        approved = await confirm_fn(prompt)
        if not approved:
            return ToolResult(
                tool_call_id=tool_call.id,
                success=False,
                error="User denied this action.",
            )

    if tool_call.name == "glob":
        tool_call.arguments["_max_results"] = settings.glob_max_results
    elif tool_call.name == "grep":
        tool_call.arguments["_max_matches"] = settings.grep_max_matches

    return await registry.execute(tool_call)


def _summarize_args(tool_call: ToolCall) -> str:
    """Create a short summary of tool call arguments for the confirmation prompt."""
    parts: list[str] = []
    for key, value in tool_call.arguments.items():
        if isinstance(value, str) and len(value) > 50:
            value = value[:50] + "..."
        parts.append(f"{key}={value!r}")
    return ", ".join(parts)
