"""TOML config loading with global → project layered merge."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from mita.config.defaults import get_global_config_path, get_project_config_path
from mita.config.schema import MitaConfig


class ConfigError(Exception):
    """Raised when a config file is malformed or fails validation.

    Carries a human-readable message so the CLI can show a clear error instead of
    a raw traceback (which previously bricked every command, including doctor).
    """


def _dedup_preserve_order(items: list[Any]) -> list[Any]:
    """Return items with duplicates removed, preserving first-seen order.

    Handles unhashable entries (dicts, as in hooks/plugins) via equality, so an
    identical hook declared in both global and project config is not run twice (S10).
    """
    result: list[Any] = []
    for item in items:
        if item not in result:
            result.append(item)
    return result


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge override into base. Lists are appended, dicts are merged recursively."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        elif key in result and isinstance(result[key], list) and isinstance(value, list):
            result[key] = _dedup_preserve_order(result[key] + value)
        else:
            result[key] = value
    return result


def _load_toml(path: Path) -> dict[str, Any]:
    """Load a TOML file and return its contents as a dict."""
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"Malformed TOML in {path}: {e}") from e
    except OSError as e:
        raise ConfigError(f"Could not read config file {path}: {e}") from e


def load_config(
    project_root: Path | None = None, *, trust_project: bool | None = None
) -> MitaConfig:
    """Load and merge configuration from global and project TOML files.

    Merge strategy:
    - Scalar values: project overrides global
    - Dicts: deep merged recursively
    - Lists (hooks, plugins, skills_paths): project values appended to global

    Security: a project's ``.mita/settings.toml`` may only contribute
    security-relevant keys (hooks, plugins, tools.permission_mode,
    tools.auto_approve, tools.confirm_destructive) when its directory is trusted.
    ``trust_project`` overrides the trust-store lookup when set explicitly
    (True/False); when None (default) the trust store decides. See finding C1.
    """
    from mita.config.trust import filter_untrusted, is_trusted

    merged: dict[str, Any] = {}

    # Load global config
    global_path = get_global_config_path()
    if global_path.is_file():
        merged = _load_toml(global_path)

    # Load and merge project config
    project_path = get_project_config_path(project_root)
    if project_path is not None:
        project_data = _load_toml(project_path)
        trusted = (
            trust_project if trust_project is not None else is_trusted(project_path.parent.parent)
        )
        if not trusted:
            project_data, _ = filter_untrusted(project_data)
        merged = _deep_merge(merged, project_data)

    try:
        return MitaConfig.model_validate(merged)
    except ValidationError as e:
        raise ConfigError(f"Invalid configuration: {e}") from e
