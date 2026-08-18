"""Chat session persistence (save/resume) under ``.mita/sessions/``."""

from __future__ import annotations

from pathlib import Path

from mita.config.defaults import PROJECT_CONFIG_DIR, get_project_root


def get_sessions_dir() -> Path:
    """Return the sessions directory for the current project.

    Anchored to the project root, not the current directory: autosave creates
    this directory, and a nested ``.mita/`` would otherwise become a false
    project root for every later command run from that subdirectory.
    """
    root = get_project_root() or Path.cwd()
    return root / PROJECT_CONFIG_DIR / "sessions"
