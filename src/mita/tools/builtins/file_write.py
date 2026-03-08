"""Built-in tool: write file contents."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mita.tools.schema import ToolDefinition, ToolParameter, ToolResult

TOOL_DEF = ToolDefinition(
    name="file_write",
    description="Write content to a file. Creates the file and any parent directories if needed.",
    parameters=[
        ToolParameter(name="path", type="string", description="Path to the file to write."),
        ToolParameter(name="content", type="string", description="Content to write to the file."),
    ],
    destructive=True,  # Overwrites existing files
)


async def execute(args: dict[str, Any]) -> ToolResult:
    """Write content to a file."""
    path_str = str(args.get("path", ""))
    content = str(args.get("content", ""))

    if not path_str:
        return ToolResult(tool_call_id="", success=False, error="Missing required parameter: path")

    path = Path(path_str).expanduser().resolve()

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        existed = path.exists()
        path.write_text(content, encoding="utf-8")
        action = "Updated" if existed else "Created"
        return ToolResult(
            tool_call_id="",
            success=True,
            output=f"{action} {path} ({len(content)} chars, {content.count(chr(10)) + 1} lines)",
        )
    except OSError as e:
        return ToolResult(tool_call_id="", success=False, error=f"Cannot write file: {e}")
