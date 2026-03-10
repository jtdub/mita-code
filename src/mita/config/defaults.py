"""Default configuration paths and values."""

from __future__ import annotations

from pathlib import Path

# Project-level config directory and file
PROJECT_CONFIG_DIR = ".mita"
PROJECT_CONFIG_FILE = "settings.toml"


def get_global_config_dir() -> Path:
    """Return the global config directory."""
    return Path.home() / ".config" / "mita"


def get_global_config_path() -> Path:
    """Return the path to the global config file."""
    return get_global_config_dir() / "config.toml"


def get_project_config_path(project_root: Path | None = None) -> Path | None:
    """Return the path to the project config file, or None if not found."""
    if project_root is None:
        project_root = _find_project_root(Path.cwd())
    if project_root is None:
        return None
    path = project_root / PROJECT_CONFIG_DIR / PROJECT_CONFIG_FILE
    return path if path.is_file() else None


def _find_project_root(start: Path) -> Path | None:
    """Walk up from start to find a project root (contains .git or .mita)."""
    current = start.resolve()
    while True:
        if (current / ".git").is_dir() or (current / PROJECT_CONFIG_DIR).is_dir():
            return current
        parent = current.parent
        if parent == current:
            return None
        current = parent
