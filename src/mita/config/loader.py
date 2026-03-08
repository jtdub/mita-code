"""TOML config loading with global → project layered merge."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from mita.config.defaults import get_global_config_path, get_project_config_path
from mita.config.schema import MitaConfig


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge override into base. Lists are appended, dicts are merged recursively."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        elif key in result and isinstance(result[key], list) and isinstance(value, list):
            result[key] = result[key] + value
        else:
            result[key] = value
    return result


def _load_toml(path: Path) -> dict[str, Any]:
    """Load a TOML file and return its contents as a dict."""
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_config(project_root: Path | None = None) -> MitaConfig:
    """Load and merge configuration from global and project TOML files.

    Merge strategy:
    - Scalar values: project overrides global
    - Dicts: deep merged recursively
    - Lists (hooks, plugins, skills_paths): project values appended to global
    """
    merged: dict[str, Any] = {}

    # Load global config
    global_path = get_global_config_path()
    if global_path.is_file():
        merged = _load_toml(global_path)

    # Load and merge project config
    project_path = get_project_config_path(project_root)
    if project_path is not None:
        project_data = _load_toml(project_path)
        merged = _deep_merge(merged, project_data)

    return MitaConfig.model_validate(merged)
