"""Built-in tool: shell command execution with timeout."""

from __future__ import annotations

import asyncio
from typing import Any

from mita.tools.schema import ToolDefinition, ToolParameter, ToolResult

TOOL_DEF = ToolDefinition(
    name="shell",
    description=(
        "Execute a shell command. Commands run in a bash subprocess with a timeout. "
        "Use for system commands, builds, tests, etc."
    ),
    parameters=[
        ToolParameter(
            name="command",
            type="string",
            description="The shell command to execute.",
        ),
        ToolParameter(
            name="timeout",
            type="integer",
            description="Timeout in seconds. Default: 120.",
            required=False,
            default=120,
        ),
    ],
    destructive=True,  # Shell commands can be destructive
)


async def execute(args: dict[str, Any]) -> ToolResult:
    """Execute a shell command."""
    command = str(args.get("command", ""))
    try:
        timeout = max(1, int(args.get("timeout", 120) or 120))
    except (ValueError, TypeError):
        timeout = 120

    # The executor injects a per-config cap; the effective timeout never exceeds it.
    try:
        cap = int(args.get("_timeout_cap", timeout))
        if cap > 0:
            timeout = min(timeout, cap)
    except (ValueError, TypeError):
        pass

    # The executor injects the workspace root so commands run inside the project.
    cwd = args.get("_cwd") or None

    if not command:
        return ToolResult(
            tool_call_id="", success=False, error="Missing required parameter: command"
        )

    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
        stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        try:
            proc.terminate()
            await asyncio.sleep(0.5)
            if proc.returncode is None:
                proc.kill()
            await proc.wait()
        except (OSError, ProcessLookupError):
            pass
        return ToolResult(
            tool_call_id="",
            success=False,
            error=f"Command timed out after {timeout}s: {command}",
        )
    except OSError as e:
        return ToolResult(tool_call_id="", success=False, error=f"Failed to run command: {e}")

    return ToolResult.from_subprocess(
        stdout_bytes,
        stderr_bytes,
        proc.returncode,
        empty_message="Command completed successfully (no output).",
    )
