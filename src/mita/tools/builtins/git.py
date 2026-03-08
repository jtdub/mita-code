"""Built-in tool: git operations (status, diff, commit, log)."""

from __future__ import annotations

import asyncio
from typing import Any

from mita.tools.schema import ToolDefinition, ToolParameter, ToolResult

TOOL_DEF = ToolDefinition(
    name="git",
    description=(
        "Run git commands. Supports: status, diff, log, add, commit, branch, checkout. "
        "Destructive operations (push, reset, clean) require confirmation."
    ),
    parameters=[
        ToolParameter(
            name="subcommand",
            type="string",
            description=(
                "The git subcommand to run "
                '(e.g. "status", "diff", "log --oneline -10", "add .", "commit -m msg").'
            ),
        ),
    ],
    destructive=True,  # Some git ops are destructive; safety module handles details
)

# Git subcommands that are always safe (read-only)
SAFE_SUBCOMMANDS = frozenset(
    {
        "status",
        "diff",
        "log",
        "show",
        "branch",
        "remote",
        "tag",
        "stash list",
        "ls-files",
        "rev-parse",
        "describe",
    }
)


def _is_safe_git_command(subcommand: str) -> bool:
    """Check if a git subcommand is read-only."""
    first_word = subcommand.strip().split()[0] if subcommand.strip() else ""
    return first_word in SAFE_SUBCOMMANDS


async def execute(args: dict[str, Any]) -> ToolResult:
    """Execute a git subcommand."""
    subcommand = str(args.get("subcommand", ""))

    if not subcommand:
        return ToolResult(
            tool_call_id="",
            success=False,
            error="Missing required parameter: subcommand",
        )

    full_command = f"git {subcommand}"

    try:
        proc = await asyncio.create_subprocess_shell(
            full_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=30)
    except TimeoutError:
        return ToolResult(
            tool_call_id="",
            success=False,
            error=f"Git command timed out: {full_command}",
        )
    except OSError as e:
        return ToolResult(tool_call_id="", success=False, error=f"Failed to run git: {e}")

    stdout = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
    stderr = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""

    output_parts: list[str] = []
    if stdout:
        output_parts.append(stdout)
    if stderr:
        output_parts.append(f"STDERR:\n{stderr}")

    output = "\n".join(output_parts) if output_parts else "(no output)"
    exit_code = proc.returncode or 0

    if exit_code != 0:
        output = f"[exit code {exit_code}]\n{output}"

    return ToolResult(
        tool_call_id="",
        success=exit_code == 0,
        output=output,
        error=f"Git command failed with exit code {exit_code}" if exit_code != 0 else None,
    )
