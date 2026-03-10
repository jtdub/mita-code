"""Built-in tool: read file contents."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mita.tools.schema import ToolDefinition, ToolParameter, ToolResult

TOOL_DEF = ToolDefinition(
    name="file_read",
    description="Read the contents of a file. Returns the file content with line numbers.",
    parameters=[
        ToolParameter(name="path", type="string", description="Path to the file to read."),
        ToolParameter(
            name="offset",
            type="integer",
            description="Line number to start reading from (1-based). Default: 1.",
            required=False,
            default=1,
        ),
        ToolParameter(
            name="limit",
            type="integer",
            description="Maximum number of lines to read. Default: 2000.",
            required=False,
            default=2000,
        ),
    ],
    destructive=False,
)


def _safe_int(value: object, default: int, minimum: int = 1) -> int:
    """Safely convert a value to int with bounds checking."""
    if value is None:
        return max(minimum, default)
    try:
        result = int(str(value))
    except (ValueError, TypeError):
        result = default
    return max(minimum, result)


async def execute(args: dict[str, Any]) -> ToolResult:
    """Read file contents with optional offset and limit."""
    path_str = str(args.get("path", ""))
    offset = _safe_int(args.get("offset", 1), default=1)
    limit = _safe_int(args.get("limit", 2000), default=2000)

    if not path_str:
        return ToolResult(tool_call_id="", success=False, error="Missing required parameter: path")

    path = Path(path_str).expanduser().resolve()

    from mita.tools.safety import validate_path_for_read

    path_error = validate_path_for_read(path)
    if path_error:
        return ToolResult(tool_call_id="", success=False, error=path_error)

    if not path.exists():
        return ToolResult(tool_call_id="", success=False, error=f"File not found: {path}")

    if not path.is_file():
        return ToolResult(tool_call_id="", success=False, error=f"Not a file: {path}")

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return ToolResult(tool_call_id="", success=False, error=f"Cannot read file: {e}")

    lines = text.splitlines()
    total_lines = len(lines)

    # Apply offset (1-based) and limit
    start = max(0, offset - 1)
    end = start + limit
    selected = lines[start:end]

    # Format with line numbers
    numbered = []
    for i, line in enumerate(selected, start=start + 1):
        numbered.append(f"{i:>6}\t{line}")

    output = "\n".join(numbered)

    if end < total_lines:
        output += f"\n\n... [{total_lines - end} more lines, {total_lines} total]"

    return ToolResult(tool_call_id="", success=True, output=output)
