"""TOML writing utilities for config and project files."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def atomic_write_text(path: Path, text: str) -> None:
    """Write text to path atomically (temp file + fsync + os.replace).

    A crash or concurrent process can never observe a truncated file: readers see
    either the old contents or the complete new contents, never a partial write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def escape_toml_string(s: str) -> str:
    """Escape a string for safe inclusion in a TOML quoted value."""
    s = s.replace("\\", "\\\\")
    s = s.replace('"', '\\"')
    s = s.replace("\n", "\\n")
    s = s.replace("\r", "\\r")
    s = s.replace("\t", "\\t")
    return s


def toml_value(v: object) -> str:
    """Format a Python value as a TOML value string."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return f'"{escape_toml_string(v)}"'
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, list):
        items = ", ".join(toml_value(i) for i in v)
        return f"[{items}]"
    return repr(v)


def dict_to_toml(data: dict, lines: list[str], prefix: str) -> None:  # type: ignore[type-arg]
    """Recursively format a dict as TOML lines.

    TOML requires every bare ``key = value`` (including scalar arrays) to precede
    any ``[table]`` / ``[[array-of-table]]`` header at the same level, so scalars
    and scalar lists are emitted first, then tables, then arrays of tables. Keys
    whose value is ``None`` are skipped entirely — a bare ``None`` token is not
    valid TOML and would make the whole file unparseable.
    """
    scalars: dict[str, object] = {}
    scalar_lists: dict[str, list] = {}  # type: ignore[type-arg]
    table_lists: dict[str, list] = {}  # type: ignore[type-arg]
    dicts: dict[str, dict] = {}  # type: ignore[type-arg]
    for k, v in data.items():
        if v is None:
            continue
        if isinstance(v, dict):
            dicts[k] = v
        elif isinstance(v, list):
            if v and isinstance(v[0], dict):
                table_lists[k] = v
            else:
                scalar_lists[k] = v
        else:
            scalars[k] = v

    for k, v in scalars.items():
        lines.append(f"{k} = {toml_value(v)}")
    for k, v in scalar_lists.items():
        lines.append(f"{k} = {toml_value(v)}")

    for k, dv in dicts.items():
        if not dv:
            continue  # skip empty tables to avoid stray headers
        section = f"{prefix}{k}" if prefix else k
        lines.append(f"\n[{section}]")
        dict_to_toml(dv, lines, prefix=f"{section}.")

    for k, lv in table_lists.items():
        section = f"{prefix}{k}" if prefix else k
        for item in lv:
            lines.append(f"\n[[{section}]]")
            dict_to_toml(item, lines, prefix=f"{section}.")


def write_toml(path: Path, data: dict) -> None:  # type: ignore[type-arg]
    """Write a dict back to a TOML file atomically."""
    lines: list[str] = []
    dict_to_toml(data, lines, prefix="")
    atomic_write_text(path, "\n".join(lines) + "\n")


def config_to_toml(cfg: object) -> str:
    """Convert a MitaConfig to a TOML-formatted string for display."""
    from mita.config.schema import MitaConfig  # avoid circular import

    if not isinstance(cfg, MitaConfig):
        raise TypeError(f"Expected MitaConfig, got {type(cfg).__name__}")
    # mode="json" converts enums to their values and paths to strings, so the
    # output is real TOML rather than Python reprs.
    data = cfg.model_dump(mode="json")
    lines: list[str] = []
    dict_to_toml(data, lines, prefix="")
    return "\n".join(lines)
