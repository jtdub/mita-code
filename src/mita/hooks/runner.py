"""Execute lifecycle hooks with context variable injection."""

from __future__ import annotations

import asyncio
import fnmatch
import subprocess
from typing import Any

from rich.console import Console

from mita.config.schema import HookDefinition

# Valid hook events
VALID_EVENTS = frozenset(
    {
        "session_start",
        "session_end",
        "pre_tool_call",
        "post_tool_call",
        "on_file_write",
    }
)

# Default timeout for hook commands (seconds)
HOOK_TIMEOUT = 30


def _matches(hook: HookDefinition, context: dict[str, Any] | None) -> bool:
    """Check if a hook's match pattern applies to the context."""
    if hook.match is None:
        return True
    if context is None:
        return True

    # Match against file_path for on_file_write
    file_path = context.get("file_path")
    if file_path and fnmatch.fnmatch(str(file_path), hook.match):
        return True

    # Match against tool name for pre/post_tool_call
    tool = context.get("tool")
    if tool and fnmatch.fnmatch(str(tool), hook.match):
        return True

    return False


def _render_command(command: str, context: dict[str, Any] | None) -> str:
    """Substitute context variables into a hook command string."""
    if context is None:
        return command

    rendered = command
    for key, value in context.items():
        rendered = rendered.replace(f"{{{key}}}", str(value))
    return rendered


async def run_hooks(
    event: str,
    hooks: list[HookDefinition],
    context: dict[str, Any] | None = None,
    console: Console | None = None,
    timeout: float = HOOK_TIMEOUT,
) -> list[dict[str, Any]]:
    """Execute all hooks matching the given event.

    Args:
        event: Hook event name (e.g., "session_start", "on_file_write").
        hooks: List of hook definitions from config.
        context: Optional context dict with vars like {file_path}, {tool}.
        console: Optional console for displaying hook output.
        timeout: Timeout in seconds for each hook command.

    Returns:
        List of result dicts with keys: event, command, returncode, stdout, stderr.
    """
    results: list[dict[str, Any]] = []

    matching = [h for h in hooks if h.event == event and _matches(h, context)]
    if not matching:
        return results

    for hook in matching:
        command = _render_command(hook.command, context)
        result = await _execute_hook(command, timeout=timeout)
        result["event"] = event
        results.append(result)

        if console:
            if result["returncode"] != 0:
                console.print(
                    f"  [yellow]Hook ({event}): '{command}' "
                    f"exited with code {result['returncode']}[/yellow]"
                )
                if result["stderr"]:
                    console.print(f"  [dim]{result['stderr'][:200]}[/dim]")
            elif result["stdout"]:
                # Show truncated output for successful hooks
                stdout = result["stdout"].strip()
                if stdout:
                    lines = stdout.split("\n")
                    preview = lines[0][:100]
                    if len(lines) > 1:
                        preview += f" (+{len(lines) - 1} more lines)"
                    console.print(f"  [dim]Hook ({event}): {preview}[/dim]")

    return results


async def _execute_hook(command: str, timeout: float = HOOK_TIMEOUT) -> dict[str, Any]:
    """Execute a single hook command asynchronously."""
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(
                subprocess.run,
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            ),
            timeout=timeout + 5,  # extra buffer for asyncio
        )
        return {
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except (TimeoutError, subprocess.TimeoutExpired):
        return {
            "command": command,
            "returncode": -1,
            "stdout": "",
            "stderr": f"Hook timed out after {timeout}s",
        }
    except OSError as e:
        return {
            "command": command,
            "returncode": -1,
            "stdout": "",
            "stderr": f"Hook execution failed: {e}",
        }
