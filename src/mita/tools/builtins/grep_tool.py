"""Built-in tool: content search (ripgrep-style)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from mita.tools.schema import ToolDefinition, ToolParameter, ToolResult

TOOL_DEF = ToolDefinition(
    name="grep",
    description="Search file contents for a regex pattern. Returns matching lines with context.",
    parameters=[
        ToolParameter(
            name="pattern",
            type="string",
            description="Regex pattern to search for.",
        ),
        ToolParameter(
            name="path",
            type="string",
            description="File or directory to search in. Defaults to current directory.",
            required=False,
            default=".",
        ),
        ToolParameter(
            name="include",
            type="string",
            description='Glob pattern to filter files (e.g. "*.py"). Only used for directories.',
            required=False,
        ),
    ],
    destructive=False,
)

DEFAULT_MAX_MATCHES = 200


async def execute(args: dict[str, Any]) -> ToolResult:
    """Search file contents for a regex pattern."""
    pattern_str = str(args.get("pattern", ""))
    path_str = str(args.get("path", ".") or ".")
    include = args.get("include")
    try:
        max_matches = max(1, int(args.get("_max_matches", DEFAULT_MAX_MATCHES)))
    except (TypeError, ValueError):
        max_matches = DEFAULT_MAX_MATCHES

    if not pattern_str:
        return ToolResult(
            tool_call_id="", success=False, error="Missing required parameter: pattern"
        )

    try:
        regex = re.compile(pattern_str)
    except re.error as e:
        return ToolResult(tool_call_id="", success=False, error=f"Invalid regex: {e}")

    target = Path(path_str).expanduser().resolve()

    from mita.tools.safety import validate_path_for_read, validate_search_base

    if target.is_file():
        read_error = validate_path_for_read(target)
        if read_error:
            return ToolResult(tool_call_id="", success=False, error=read_error)
    elif target.is_dir():
        base_error = validate_search_base(target)
        if base_error:
            return ToolResult(tool_call_id="", success=False, error=base_error)

    if target.is_file():
        files = [target]
    elif target.is_dir():
        glob_pattern = str(include) if include else "**/*"
        files = sorted(f for f in target.glob(glob_pattern) if f.is_file())
    else:
        return ToolResult(tool_call_id="", success=False, error=f"Path not found: {target}")

    matches: list[str] = []
    match_count = 0

    for file_path in files:
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        for line_num, line in enumerate(text.splitlines(), 1):
            if regex.search(line):
                matches.append(f"{file_path}:{line_num}: {line.rstrip()}")
                match_count += 1
                if match_count >= max_matches:
                    break
        if match_count >= max_matches:
            break

    if not matches:
        return ToolResult(tool_call_id="", success=True, output="No matches found.")

    output = "\n".join(matches)
    if match_count >= max_matches:
        output += f"\n\n... [showing first {max_matches} matches]"
    else:
        output += f"\n\n[{match_count} match(es)]"

    return ToolResult(tool_call_id="", success=True, output=output)
