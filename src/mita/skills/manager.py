"""CLI command handlers for skills management."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from mita.config.loader import load_config
from mita.skills.loader import discover_skills

console = Console()

SKILL_TEMPLATE = """---
name: {name}
description: "{description}"
args: []
---

{name} skill prompt goes here.

Use {{{{input}}}} to reference the user's input.
"""


def list_skills() -> None:
    """List all available skills."""
    config = load_config()
    skills = discover_skills(config.skills_paths)

    if not skills:
        console.print("[yellow]No skills found.[/yellow]")
        _show_paths(config.skills_paths)
        return

    table = Table(title="Available Skills")
    table.add_column("Name", style="bold")
    table.add_column("Description")
    table.add_column("Args")
    table.add_column("Source")

    for skill in skills:
        arg_names = ", ".join(a.name for a in skill.frontmatter.args) or "-"
        table.add_row(
            skill.frontmatter.name,
            skill.frontmatter.description,
            arg_names,
            str(skill.source_path),
        )

    console.print(table)


def show_skill(name: str) -> None:
    """Show details and template for a skill."""
    config = load_config()
    skills = discover_skills(config.skills_paths)

    match = None
    for s in skills:
        if s.frontmatter.name == name:
            match = s
            break

    if not match:
        console.print(f"[red]Skill '{name}' not found.[/red]")
        return

    console.print(f"[bold]Skill:[/bold] {match.frontmatter.name}")
    console.print(f"[bold]Description:[/bold] {match.frontmatter.description}")
    console.print(f"[bold]Source:[/bold] {match.source_path}")

    if match.frontmatter.args:
        console.print("[bold]Arguments:[/bold]")
        for arg in match.frontmatter.args:
            req = "required" if arg.required else f"optional, default={arg.default}"
            console.print(f"  - {arg.name} ({arg.type}): {arg.description} [{req}]")

    if match.frontmatter.trigger:
        console.print(f"[bold]Trigger:[/bold] {match.frontmatter.trigger}")

    console.print("\n[bold]Template:[/bold]")
    console.print(Markdown(match.template))


def create_skill(name: str) -> None:
    """Create a new skill file from a template."""
    config = load_config()

    # Use the first writable path (prefer project-local)
    target_dir = None
    for raw_path in reversed(config.skills_paths):
        p = Path(raw_path).expanduser()
        target_dir = p
        break

    if target_dir is None:
        console.print("[red]No skills path configured.[/red]")
        return

    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / f"{name}.md"

    if target_file.exists():
        console.print(f"[red]Skill '{name}' already exists at {target_file}.[/red]")
        return

    content = SKILL_TEMPLATE.format(name=name, description=f"A {name} skill.")
    target_file.write_text(content, encoding="utf-8")
    console.print(f"[green]Created skill '{name}' at {target_file}.[/green]")
    console.print("Edit the file to customize the prompt template.")


def show_paths() -> None:
    """Show skill search paths."""
    config = load_config()
    _show_paths(config.skills_paths)


def _show_paths(paths: list[str]) -> None:
    """Display skill search paths."""
    console.print("[bold]Skill search paths:[/bold]")
    for raw_path in paths:
        p = Path(raw_path).expanduser()
        exists = p.is_dir()
        style = "bold" if exists else "dim"
        console.print(f"  {p}", style=style)
