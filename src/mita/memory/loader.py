"""Read, merge, and truncate MITA.md memory files."""

from __future__ import annotations

from pathlib import Path

from mita.config.schema import MemorySettings
from mita.memory.discovery import discover_memory_files


def _classify_scope(path: Path) -> str:
    """Classify a memory file as global, project, or directory scope."""
    home = Path.home()
    if str(path).startswith(str(home / ".config" / "mita")):
        return "global"
    # If the file is at a project root (parent has .git or .mita), it's project-level
    parent = path.parent
    if parent.name == ".mita":
        parent = parent.parent
    if (parent / ".git").exists() or (parent / ".mita").exists():
        return "project"
    return "directory"


def _read_and_truncate(path: Path, max_lines: int) -> tuple[str, bool]:
    """Read a file and truncate to max_lines. Returns (content, was_truncated)."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) <= max_lines:
        return "\n".join(lines), False
    truncated = lines[:max_lines]
    truncated.append(f"[... truncated at {max_lines} lines. Keep MITA.md concise.]")
    return "\n".join(truncated), True


def load_memory(
    cwd: Path | None = None,
    settings: MemorySettings | None = None,
) -> str:
    """Load and merge all discovered MITA.md files into a single string.

    Returns formatted memory content with source annotations, suitable
    for injection into the system prompt.
    """
    if settings is None:
        settings = MemorySettings()

    files = discover_memory_files(cwd)
    if not files:
        return ""

    sections: list[str] = []
    for path in files:
        scope = _classify_scope(path)
        content, _ = _read_and_truncate(path, settings.max_lines_per_file)
        if content.strip():
            sections.append(f"<!-- Source: {path} ({scope}) -->\n{content}")

    if not sections:
        return ""

    body = "\n\n".join(sections)

    # Cap the total injected memory so many MITA.md files can't blow up the context
    # window. ~4 chars/token, consistent with conversation token estimation (S5).
    max_chars = settings.max_total_tokens * 4
    if len(body) > max_chars:
        body = body[:max_chars].rstrip() + "\n<!-- memory truncated to fit max_total_tokens -->"

    return "<memory>\n" + body + "\n</memory>"


def load_memory_raw(cwd: Path | None = None) -> list[tuple[Path, str, str]]:
    """Load memory files and return as list of (path, scope, content) tuples.

    Useful for the `mita memory show` CLI command.
    """
    settings = MemorySettings()
    files = discover_memory_files(cwd)
    result: list[tuple[Path, str, str]] = []
    for path in files:
        scope = _classify_scope(path)
        content, _ = _read_and_truncate(path, settings.max_lines_per_file)
        result.append((path, scope, content))
    return result
