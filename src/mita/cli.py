"""Typer CLI app — top-level command routing for Mita Code."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console
from rich.syntax import Syntax

import mita
from mita.config.loader import load_config
from mita.memory.manager import add_memory, edit_memory, show_memory, show_memory_paths
from mita.models.manager import (
    list_models,
    pull_model,
    remove_model,
    set_default_model,
    show_hardware,
    show_model_info,
    show_recommendations,
)

console = Console()

app = typer.Typer(
    name="mita",
    help="Local-first agentic coding assistant powered by Ollama.",
    no_args_is_help=True,
    add_completion=True,
)


# ── Version ───────────────────────────────────────────────────────


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"mita {mita.__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            "-v",
            help="Show version and exit.",
            callback=_version_callback,
            is_eager=True,
        ),
    ] = None,
) -> None:
    """Mita Code — local-first agentic coding assistant."""


# ── Config commands ───────────────────────────────────────────────

config_app = typer.Typer(help="Configuration management.")
app.add_typer(config_app, name="config")


@config_app.command("show")
def config_show() -> None:
    """Show the merged configuration."""
    cfg = load_config()
    toml_str = _config_to_toml(cfg)
    console.print(Syntax(toml_str, "toml", theme="monokai"))


@config_app.command("path")
def config_path() -> None:
    """Show config file paths."""
    from mita.config.defaults import get_global_config_path, get_project_config_path

    global_path = get_global_config_path()
    project_path = get_project_config_path()

    console.print(f"  Global:  {global_path}", style="bold" if global_path.is_file() else "dim")
    if project_path:
        console.print(f"  Project: {project_path}", style="bold")
    else:
        console.print("  Project: [dim]not found[/dim]")


# ── Memory commands ───────────────────────────────────────────────

memory_app = typer.Typer(help="MITA.md memory management.")
app.add_typer(memory_app, name="memory")


@memory_app.command("show")
def memory_show() -> None:
    """Show all discovered memory content."""
    show_memory()


@memory_app.command("path")
def memory_path() -> None:
    """Show discovered memory file paths."""
    show_memory_paths()


@memory_app.command("edit")
def memory_edit(
    is_global: Annotated[bool, typer.Option("--global", "-g", help="Edit global memory.")] = False,
    project: Annotated[bool, typer.Option("--project", "-p", help="Edit project memory.")] = False,
) -> None:
    """Open a MITA.md file in $EDITOR."""
    if is_global:
        edit_memory("global")
    elif project:
        edit_memory("project")
    else:
        edit_memory()


@memory_app.command("add")
def memory_add(
    text: Annotated[str, typer.Argument(help="Text to add to memory.")],
    is_global: Annotated[
        bool, typer.Option("--global", "-g", help="Add to global memory.")
    ] = False,
    project: Annotated[
        bool, typer.Option("--project", "-p", help="Add to project memory.")
    ] = False,
) -> None:
    """Append a line to a MITA.md file."""
    if is_global:
        add_memory(text, "global")
    elif project:
        add_memory(text, "project")
    else:
        add_memory(text, "project")


# ── Models commands ───────────────────────────────────────────────

models_app = typer.Typer(help="Model management.")
app.add_typer(models_app, name="models")


@models_app.command("list")
def models_list() -> None:
    """List installed Ollama models."""
    list_models()


@models_app.command("pull")
def models_pull(
    name: Annotated[str, typer.Argument(help="Model name to pull (e.g. qwen2.5-coder:7b).")],
) -> None:
    """Pull a model from the Ollama registry."""
    pull_model(name)


@models_app.command("remove")
def models_remove(
    name: Annotated[str, typer.Argument(help="Model name to remove.")],
) -> None:
    """Remove an installed model."""
    remove_model(name)


@models_app.command("recommend")
def models_recommend() -> None:
    """Recommend models for your hardware."""
    show_recommendations()


@models_app.command("info")
def models_info(
    name: Annotated[str, typer.Argument(help="Model name to show info for.")],
) -> None:
    """Show details about a model."""
    show_model_info(name)


@models_app.command("default")
def models_default(
    name: Annotated[str, typer.Argument(help="Model name to set as default.")],
) -> None:
    """Set the default model."""
    set_default_model(name)


@models_app.command("hardware")
def models_hardware() -> None:
    """Show detected hardware information."""
    show_hardware()


# ── Chat / Ask commands ───────────────────────────────────────────


@app.command("chat")
def chat_command() -> None:
    """Open an interactive chat session with the agent."""
    import asyncio

    from mita.agent.conversation import Conversation
    from mita.agent.loop import run_agent
    from mita.config.loader import load_config as _load_config
    from mita.ui.display import get_console
    from mita.ui.repl import repl_loop

    cfg = _load_config()
    chat_console = get_console()
    conversation = Conversation()

    async def on_input(user_input: str) -> None:
        nonlocal conversation
        conversation = await run_agent(user_input, cfg, chat_console, conversation=conversation)

    def on_clear() -> None:
        nonlocal conversation
        conversation.clear_non_system()

    asyncio.run(repl_loop(chat_console, on_input, on_clear=on_clear))


@app.command("ask")
def ask_command(
    prompt: Annotated[str, typer.Argument(help="The prompt to send to the agent.")],
) -> None:
    """Send a single prompt to the agent (non-interactive)."""
    import asyncio

    from mita.agent.loop import run_agent
    from mita.config.loader import load_config as _load_config
    from mita.ui.display import get_console

    cfg = _load_config()
    ask_console = get_console()
    asyncio.run(run_agent(prompt, cfg, ask_console))


# ── Helpers ───────────────────────────────────────────────────────


def _config_to_toml(cfg: object) -> str:
    """Convert a MitaConfig to a TOML-formatted string for display."""
    from mita.config.schema import MitaConfig

    assert isinstance(cfg, MitaConfig)
    data = cfg.model_dump()
    lines: list[str] = []
    _dict_to_toml(data, lines, prefix="")
    return "\n".join(lines)


def _dict_to_toml(data: dict, lines: list[str], prefix: str) -> None:  # type: ignore[type-arg]
    """Recursively format a dict as TOML."""
    scalars = {k: v for k, v in data.items() if not isinstance(v, (dict, list))}
    dicts = {k: v for k, v in data.items() if isinstance(v, dict)}
    lists = {k: v for k, v in data.items() if isinstance(v, list)}

    for k, v in scalars.items():
        lines.append(f"{k} = {_toml_value(v)}")

    for k, v in lists.items():
        if v and isinstance(v[0], dict):
            # Array of tables
            for item in v:
                section = f"{prefix}{k}" if prefix else k
                lines.append(f"\n[[{section}]]")
                _dict_to_toml(item, lines, prefix=f"{section}.")
        else:
            lines.append(f"{k} = {_toml_value(v)}")

    for k, v in dicts.items():
        section = f"{prefix}{k}" if prefix else k
        lines.append(f"\n[{section}]")
        _dict_to_toml(v, lines, prefix=f"{section}.")


def _toml_value(v: object) -> str:
    """Format a Python value as a TOML value string."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return f'"{v}"'
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, list):
        items = ", ".join(_toml_value(i) for i in v)
        return f"[{items}]"
    return repr(v)
