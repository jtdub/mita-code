"""Walk-up-tree discovery of MITA.md memory files."""

from __future__ import annotations

from pathlib import Path

from mita.config.defaults import PROJECT_CONFIG_DIR

MEMORY_FILENAME = "MITA.md"


def get_global_memory_path() -> Path:
    """Return the path to the global MITA.md file."""
    return Path.home() / ".config" / "mita" / MEMORY_FILENAME


def discover_memory_files(cwd: Path | None = None) -> list[Path]:
    """Discover MITA.md files by walking up from cwd to the project root.

    Lookup order (returned list, lowest priority first):
    1. Global: ~/.config/mita/MITA.md
    2. Project root: <project_root>/MITA.md or <project_root>/.mita/MITA.md
    3. Directory-level: <cwd>/MITA.md (if different from project root)

    Returns paths ordered from lowest to highest priority (global first).
    """
    if cwd is None:
        cwd = Path.cwd()
    cwd = cwd.resolve()

    found: list[Path] = []

    # Walk up from cwd, collecting MITA.md files
    current = cwd
    while True:
        candidate = current / MEMORY_FILENAME
        if candidate.is_file():
            found.append(candidate)

        # Check .mita/ subdirectory
        dotmita_candidate = current / PROJECT_CONFIG_DIR / MEMORY_FILENAME
        if dotmita_candidate.is_file() and dotmita_candidate not in found:
            found.append(dotmita_candidate)

        # Stop at project root (contains .git or .mita), or at the home directory.
        # Without the home boundary, running mita from a non-repo directory walked
        # all the way to '/', reading ~/MITA.md and even /MITA.md (finding S9).
        if (
            (current / ".git").exists()
            or (current / PROJECT_CONFIG_DIR).exists()
            or current == Path.home()
        ):
            break

        parent = current.parent
        if parent == current:
            break
        current = parent

    # Add global memory if it exists and isn't already found
    global_memory_path = get_global_memory_path()
    if global_memory_path.is_file() and global_memory_path not in found:
        found.append(global_memory_path)

    # Reverse so global (lowest priority) is first
    found.reverse()
    return found
