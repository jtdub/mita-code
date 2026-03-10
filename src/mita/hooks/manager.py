"""Hook management — CLI helpers for listing and adding hooks."""

from __future__ import annotations

from rich.console import Console

from mita.config.loader import load_config
from mita.hooks.runner import VALID_EVENTS

console = Console()


def list_hooks() -> None:
    """List all configured hooks."""
    cfg = load_config()
    if not cfg.hooks:
        console.print("[dim]No hooks configured.[/dim]")
        return

    for hook in cfg.hooks:
        match_str = f" (match: {hook.match})" if hook.match else ""
        console.print(f"  [{hook.event}] {hook.command}{match_str}")


def add_hook(event: str, command: str, match: str | None = None) -> None:
    """Add a hook to the project config."""
    if event not in VALID_EVENTS:
        console.print(f"[red]Invalid event '{event}'.[/red]")
        console.print(f"Valid events: {', '.join(sorted(VALID_EVENTS))}")
        return

    from pathlib import Path

    from mita.config.defaults import (
        PROJECT_CONFIG_DIR,
        PROJECT_CONFIG_FILE,
        _find_project_root,
        get_project_config_path,
    )

    project_path = get_project_config_path()
    if not project_path:
        root = _find_project_root(Path.cwd())
        if root is None:
            console.print("[red]Not in a project directory (no .git or .mita/).[/red]")
            return
        config_dir = root / PROJECT_CONFIG_DIR
        config_dir.mkdir(exist_ok=True)
        project_path = config_dir / PROJECT_CONFIG_FILE
        project_path.touch()

    from mita.cli import _escape_toml_string

    esc = _escape_toml_string
    lines = [f'\n[[hooks]]\nevent = "{esc(event)}"\ncommand = "{esc(command)}"']
    if match:
        lines.append(f'match = "{esc(match)}"')

    block = "\n".join(lines) + "\n"

    with open(project_path, "a") as f:
        f.write(block)

    console.print(f"[green]Hook added: [{event}] {command}[/green]")


def remove_hooks(event: str) -> None:
    """Remove all hooks for an event from the project config."""
    import tomllib

    from mita.config.defaults import get_project_config_path

    project_path = get_project_config_path()
    if not project_path or not project_path.is_file():
        console.print("[red]No project config found.[/red]")
        return

    with open(project_path, "rb") as f:
        data = tomllib.load(f)

    hooks = data.get("hooks", [])
    new_hooks = [h for h in hooks if h.get("event") != event]
    if len(new_hooks) == len(hooks):
        console.print(f"[yellow]No hooks found for event '{event}'.[/yellow]")
        return

    removed = len(hooks) - len(new_hooks)
    data["hooks"] = new_hooks

    # Rewrite the TOML file
    from mita.cli import _dict_to_toml

    lines: list[str] = []
    _dict_to_toml(data, lines, prefix="")

    project_path.write_text("\n".join(lines) + "\n")
    console.print(f"[green]Removed {removed} hook(s) for event '{event}'.[/green]")
