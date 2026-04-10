"""TOML writing utilities for config and project files."""

from __future__ import annotations

from pathlib import Path


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
    """Recursively format a dict as TOML lines."""
    scalars = {k: v for k, v in data.items() if not isinstance(v, (dict, list))}
    dicts = {k: v for k, v in data.items() if isinstance(v, dict)}
    lists = {k: v for k, v in data.items() if isinstance(v, list)}

    for k, v in scalars.items():
        lines.append(f"{k} = {toml_value(v)}")

    for k, v in lists.items():
        if v and isinstance(v[0], dict):
            for item in v:
                section = f"{prefix}{k}" if prefix else k
                lines.append(f"\n[[{section}]]")
                dict_to_toml(item, lines, prefix=f"{section}.")
        else:
            lines.append(f"{k} = {toml_value(v)}")

    for k, v in dicts.items():
        section = f"{prefix}{k}" if prefix else k
        lines.append(f"\n[{section}]")
        dict_to_toml(v, lines, prefix=f"{section}.")


def write_toml(path: Path, data: dict) -> None:  # type: ignore[type-arg]
    """Write a dict back to a TOML file."""
    lines: list[str] = []
    dict_to_toml(data, lines, prefix="")
    path.write_text("\n".join(lines) + "\n")


def config_to_toml(cfg: object) -> str:
    """Convert a MitaConfig to a TOML-formatted string for display."""
    from mita.config.schema import MitaConfig

    assert isinstance(cfg, MitaConfig)
    data = cfg.model_dump()
    lines: list[str] = []
    dict_to_toml(data, lines, prefix="")
    return "\n".join(lines)
