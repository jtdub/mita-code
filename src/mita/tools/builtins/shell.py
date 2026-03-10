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

    if not command:
        return ToolResult(
            tool_call_id="", success=False, error="Missing required parameter: command"
        )

    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
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

    stdout = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
    stderr = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""

    output_parts: list[str] = []
    if stdout:
        output_parts.append(stdout)
    if stderr:
        output_parts.append(f"STDERR:\n{stderr}")

    output = (
        "\n".join(output_parts) if output_parts else "Command completed successfully (no output)."
    )
    exit_code = proc.returncode or 0

    if exit_code != 0:
        output = f"[exit code {exit_code}]\n{output}"

    return ToolResult(
        tool_call_id="",
        success=exit_code == 0,
        output=output,
        error=f"Command exited with code {exit_code}" if exit_code != 0 else None,
    )
