"""Built-in tool: glob-based file search."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mita.tools.schema import ToolDefinition, ToolParameter, ToolResult

TOOL_DEF = ToolDefinition(
    name="glob",
    description="Search for files matching a glob pattern. Returns matching file paths.",
    parameters=[
        ToolParameter(
            name="pattern",
            type="string",
            description='Glob pattern (e.g. "**/*.py", "src/**/*.ts").',
        ),
        ToolParameter(
            name="path",
            type="string",
            description="Directory to search in. Defaults to current directory.",
            required=False,
            default=".",
        ),
    ],
    destructive=False,
)

DEFAULT_MAX_RESULTS = 500


async def execute(args: dict[str, Any]) -> ToolResult:
    """Search for files matching a glob pattern."""
    pattern = str(args.get("pattern", ""))
    base_path = str(args.get("path", ".") or ".")
    max_results = int(args.get("_max_results", DEFAULT_MAX_RESULTS))

    if not pattern:
        return ToolResult(
            tool_call_id="", success=False, error="Missing required parameter: pattern"
        )

    base = Path(base_path).expanduser().resolve()
    if not base.is_dir():
        return ToolResult(tool_call_id="", success=False, error=f"Not a directory: {base}")

    from mita.tools.safety import validate_search_base

    base_error = validate_search_base(base)
    if base_error:
        return ToolResult(tool_call_id="", success=False, error=base_error)

    try:
        matches = sorted(base.glob(pattern))
    except ValueError as e:
        return ToolResult(tool_call_id="", success=False, error=f"Invalid glob pattern: {e}")

    # Filter out directories, keep only files
    files = [str(m) for m in matches if m.is_file()]

    if not files:
        return ToolResult(tool_call_id="", success=True, output="No files matched.")

    truncated = len(files) > max_results
    output_files = files[:max_results]
    output = "\n".join(output_files)

    if truncated:
        output += f"\n\n... [{len(files)} total matches, showing first {max_results}]"
    else:
        output += f"\n\n[{len(files)} file(s) matched]"

    return ToolResult(tool_call_id="", success=True, output=output, truncated=truncated)
