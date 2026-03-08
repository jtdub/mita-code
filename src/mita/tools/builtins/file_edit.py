"""Built-in tool: string-replace edit in a file."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mita.tools.schema import ToolDefinition, ToolParameter, ToolResult

TOOL_DEF = ToolDefinition(
    name="file_edit",
    description=(
        "Edit a file by replacing an exact string match. "
        "The old_string must appear exactly once in the file."
    ),
    parameters=[
        ToolParameter(name="path", type="string", description="Path to the file to edit."),
        ToolParameter(
            name="old_string", type="string", description="The exact text to find and replace."
        ),
        ToolParameter(name="new_string", type="string", description="The replacement text."),
    ],
    destructive=True,
)


async def execute(args: dict[str, Any]) -> ToolResult:
    """Perform a string-replace edit on a file."""
    path_str = str(args.get("path", ""))
    old_string = str(args.get("old_string", ""))
    new_string = str(args.get("new_string", ""))

    if not path_str:
        return ToolResult(tool_call_id="", success=False, error="Missing required parameter: path")
    if not old_string:
        return ToolResult(
            tool_call_id="", success=False, error="Missing required parameter: old_string"
        )

    path = Path(path_str).expanduser().resolve()

    if not path.is_file():
        return ToolResult(tool_call_id="", success=False, error=f"File not found: {path}")

    try:
        content = path.read_text(encoding="utf-8")
    except OSError as e:
        return ToolResult(tool_call_id="", success=False, error=f"Cannot read file: {e}")

    count = content.count(old_string)
    if count == 0:
        return ToolResult(
            tool_call_id="",
            success=False,
            error="old_string not found in file. Make sure it matches exactly.",
        )
    if count > 1:
        return ToolResult(
            tool_call_id="",
            success=False,
            error=f"old_string found {count} times. It must be unique. Add more context.",
        )

    new_content = content.replace(old_string, new_string, 1)

    try:
        path.write_text(new_content, encoding="utf-8")
    except OSError as e:
        return ToolResult(tool_call_id="", success=False, error=f"Cannot write file: {e}")

    return ToolResult(tool_call_id="", success=True, output=f"Edited {path}")
