"""Project-directory trust store and untrusted-config filtering.

A project's ``.mita/settings.toml`` can run code at session start (hooks, plugins)
and weaken the confirmation gate (permission_mode, auto_approve, confirm_destructive).
A freshly cloned repository must not be able to do any of that silently. These keys
are stripped from an untrusted project's config until the user explicitly trusts the
directory. See audit finding C1.
"""

from __future__ import annotations

import copy
import tomllib
from pathlib import Path
from typing import Any

from mita.config.defaults import get_global_config_dir
from mita.config.toml_writer import atomic_write_text, toml_value

# Config keys that let a project execute code or weaken the safety gate.
SECURITY_RELEVANT_KEYS: tuple[str, ...] = ("hooks", "plugins")
SECURITY_RELEVANT_TOOL_KEYS: tuple[str, ...] = (
    "permission_mode",
    "auto_approve",
    "confirm_destructive",
)


def get_trust_store_path() -> Path:
    """Return the path to the trusted-directories store."""
    return get_global_config_dir() / "trusted_dirs.toml"


def _load_trusted() -> set[str]:
    path = get_trust_store_path()
    if not path.is_file():
        return set()
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        return set()
    entries = data.get("trusted", [])
    return {str(e) for e in entries if isinstance(e, str)}


def is_trusted(project_root: Path) -> bool:
    """Return True if the project directory has been explicitly trusted."""
    return str(project_root.resolve()) in _load_trusted()


def add_trusted(project_root: Path) -> None:
    """Record a project directory as trusted."""
    trusted = _load_trusted()
    trusted.add(str(project_root.resolve()))
    items = ", ".join(toml_value(p) for p in sorted(trusted))
    atomic_write_text(get_trust_store_path(), f"trusted = [{items}]\n")


def security_relevant_keys(project_data: dict[str, Any]) -> list[str]:
    """List the security-relevant keys present in a project config dict."""
    found: list[str] = [k for k in SECURITY_RELEVANT_KEYS if k in project_data]
    tools = project_data.get("tools")
    if isinstance(tools, dict):
        found += [f"tools.{k}" for k in SECURITY_RELEVANT_TOOL_KEYS if k in tools]
    return found


def filter_untrusted(project_data: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Strip security-relevant keys from an untrusted project's config.

    Returns the cleaned dict and the list of stripped key labels. The input is not
    mutated. Non-security keys (model, ui, index, etc.) are preserved.
    """
    stripped = security_relevant_keys(project_data)
    if not stripped:
        return project_data, []
    cleaned = copy.deepcopy(project_data)
    for key in SECURITY_RELEVANT_KEYS:
        cleaned.pop(key, None)
    tools = cleaned.get("tools")
    if isinstance(tools, dict):
        for key in SECURITY_RELEVANT_TOOL_KEYS:
            tools.pop(key, None)
        if not tools:
            cleaned.pop("tools", None)
    return cleaned, stripped
