"""CLI commands for MITA.md memory management."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from mita.config.defaults import _find_project_root
from mita.memory.discovery import MEMORY_FILENAME, discover_memory_files, get_global_memory_path
from mita.memory.loader import load_memory_raw

console = Console()


def _ensure_memory_file(path: Path, title: str) -> None:
    """Create a memory file with a header if it doesn't exist."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {title}\n\n", encoding="utf-8")


def show_memory() -> None:
    """Display all discovered memory content with source annotations."""
    entries = load_memory_raw()
    if not entries:
        console.print("[dim]No MITA.md files found.[/dim]")
        return

    for path, scope, content in entries:
        console.print(
            Panel(
                Markdown(content) if content.strip() else "[dim]Empty[/dim]",
                title=f"[bold]{path}[/bold]",
                subtitle=f"[dim]{scope}[/dim]",
                border_style="blue",
            )
        )


def show_memory_paths() -> None:
    """Display the paths of all discovered MITA.md files."""
    files = discover_memory_files()
    if not files:
        console.print("[dim]No MITA.md files found.[/dim]")
        return

    for path in files:
        console.print(f"  {path}")


def edit_memory(scope: str | None = None) -> None:
    """Open a MITA.md file in $EDITOR.

    Args:
        scope: "global", "project", or None (nearest/project default).
    """
    editor = os.environ.get("EDITOR", "vi")

    if scope == "global":
        path = get_global_memory_path()
        _ensure_memory_file(path, "Global Mita Memory")
    elif scope == "project":
        project_root = _find_project_root(Path.cwd())
        if project_root is None:
            console.print("[red]Not in a project (no .git or .mita directory found).[/red]")
            return
        path = project_root / MEMORY_FILENAME
        _ensure_memory_file(path, f"{project_root.name} — Mita Memory")
    else:
        # Find nearest existing MITA.md, or create at project root
        files = discover_memory_files()
        # Filter out global — prefer project/directory level
        local_files = [f for f in files if "/.config/mita/" not in str(f)]
        if local_files:
            path = local_files[-1]  # highest priority (most specific)
        else:
            project_root = _find_project_root(Path.cwd())
            if project_root is None:
                console.print(
                    "[red]Not in a project (no .git or .mita directory found). "
                    "Use --global to edit global memory.[/red]"
                )
                return
            path = project_root / MEMORY_FILENAME
            _ensure_memory_file(path, f"{project_root.name} — Mita Memory")

    console.print(f"[dim]Opening {path} in {editor}...[/dim]")
    subprocess.run([editor, str(path)])


def add_memory(text: str, scope: str | None = None) -> None:
    """Append a line of text to a MITA.md file.

    Args:
        text: Text to append.
        scope: "global", "project", or None (defaults to project).
    """
    if scope == "global":
        path = get_global_memory_path()
        _ensure_memory_file(path, "Global Mita Memory")
    else:
        project_root = _find_project_root(Path.cwd())
        if project_root is None:
            console.print("[red]Not in a project (no .git or .mita directory found).[/red]")
            return
        path = project_root / MEMORY_FILENAME
        _ensure_memory_file(path, f"{project_root.name} — Mita Memory")

    with open(path, "a", encoding="utf-8") as f:
        f.write(f"- {text}\n")

    console.print(f"[green]Added to {path}[/green]")
