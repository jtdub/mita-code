"""Typer CLI app — top-level command routing for Mita Code."""

from __future__ import annotations

import enum
import logging
import shlex
import sys
from io import StringIO
from typing import TYPE_CHECKING, Annotated

import typer
from rich.console import Console
from rich.syntax import Syntax

import mita
from mita.config.loader import load_config
from mita.config.toml_writer import config_to_toml as _config_to_toml
from mita.config.toml_writer import (
    escape_toml_string,
    write_toml,
)
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

if TYPE_CHECKING:
    from mita.config.schema import MitaConfig

_logger = logging.getLogger(__name__)

console = Console()


def _safe_load_config() -> MitaConfig:
    """Load config, converting a ConfigError into a clean CLI error + exit."""
    from mita.config.loader import ConfigError
    from mita.config.loader import load_config as _load_config

    try:
        return _load_config()
    except ConfigError as e:
        console.print(f"[red]Configuration error:[/red] {e}")
        raise typer.Exit(1) from e


def _preflight_backend(cfg: MitaConfig, out_console: Console) -> bool:
    """Ensure the configured LLM backend is available before a chat/ask run.

    For Ollama this manages the daemon and pulls the model (existing behavior). For
    every other provider it does a generic reachability check and never touches Ollama
    (finding C5).
    """
    from mita.config.schema import LLMProvider

    if cfg.llm.provider == LLMProvider.OLLAMA:
        from mita.models.server import ensure_model, ensure_server

        if not ensure_server(
            host=cfg.ollama.host, auto_manage=cfg.ollama.auto_manage, console=out_console
        ):
            return False
        return ensure_model(
            cfg.model.default,
            host=cfg.ollama.host,
            timeout=cfg.ollama.timeout,
            console=out_console,
        )

    from mita.llm.health import backend_reachable
    from mita.llm.providers import resolve_backend

    backend = resolve_backend(cfg)
    if not backend_reachable(backend.api_base, backend.api_key):
        out_console.print(
            f"[red]Cannot reach the {cfg.llm.provider.value} backend at "
            f"{backend.api_base}.[/red]\nStart the server or fix [bold][llm] base_url[/bold]."
        )
        return False
    return True


app = typer.Typer(
    name="mita",
    help="Local-first agentic coding assistant powered by Ollama.",
    no_args_is_help=True,
    add_completion=True,
)


class OutputFormat(enum.StrEnum):
    """Output format for non-interactive ask command."""

    RICH = "rich"
    TEXT = "text"
    JSON = "json"


# ── Version ───────────────────────────────────────────────────────


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"mita {mita.__version__}")
        raise typer.Exit()


def _verbose_callback(value: bool) -> None:
    if value:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s %(name)s %(levelname)s %(message)s",
        )


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
    verbose: Annotated[
        bool | None,
        typer.Option(
            "--verbose",
            help="Enable verbose/debug logging.",
            callback=_verbose_callback,
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
    scope = "global" if is_global else "project"
    add_memory(text, scope)


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


# ── Index commands ────────────────────────────────────────────────

index_app = typer.Typer(help="Codebase index management.")
app.add_typer(index_app, name="index")


@index_app.command("build")
def index_build(
    force: Annotated[bool, typer.Option("--force", "-f", help="Force rebuild the index.")] = False,
) -> None:
    """Build the codebase index."""
    import asyncio

    from mita.index.manager import build_index

    async def _ensure_model(config: object) -> bool:
        """Pull the embedding model if missing. Lives in cli.py to respect dependency rules."""
        from mita.config.schema import MitaConfig
        from mita.index.embeddings import EmbeddingClient
        from mita.models.ollama_client import OllamaClient

        assert isinstance(config, MitaConfig)
        embedder = EmbeddingClient(config)
        console.print(f"[yellow]Embedding model '{embedder.model}' is not installed.[/yellow]")
        confirm = console.input(f"Pull '{embedder.model}' now? [Y/n] ").strip().lower()
        if confirm not in ("", "y", "yes"):
            console.print("[red]Cannot build index without embedding model.[/red]")
            return False
        client = OllamaClient(host=config.ollama.host)
        console.print(f"Pulling {embedder.model}...")
        for _progress in client.pull(embedder.model):
            pass
        console.print("[green]Model pulled successfully.[/green]")
        return True

    asyncio.run(build_index(force=force, pull_model_fn=_ensure_model))


@index_app.command("status")
def index_status() -> None:
    """Show index statistics."""
    import asyncio

    from mita.index.manager import show_index_status

    asyncio.run(show_index_status())


@index_app.command("search")
def index_search(
    query: Annotated[str, typer.Argument(help="Search query.")],
    top_k: Annotated[int, typer.Option("--top-k", "-k", help="Number of results.")] = 10,
) -> None:
    """Search the codebase index."""
    import asyncio

    from mita.index.manager import search_index

    asyncio.run(search_index(query, top_k=top_k))


@index_app.command("clear")
def index_clear() -> None:
    """Delete the codebase index."""
    import asyncio

    from mita.index.manager import clear_index

    asyncio.run(clear_index())


# ── Skills commands ───────────────────────────────────────────────

skills_app = typer.Typer(help="Skills management.")
app.add_typer(skills_app, name="skills")


@skills_app.command("list")
def skills_list() -> None:
    """List all available skills."""
    from mita.skills.manager import list_skills

    list_skills()


@skills_app.command("show")
def skills_show(
    name: Annotated[str, typer.Argument(help="Skill name to show.")],
) -> None:
    """Show details about a skill."""
    from mita.skills.manager import show_skill

    show_skill(name)


@skills_app.command("create")
def skills_create(
    name: Annotated[str, typer.Argument(help="Name for the new skill.")],
) -> None:
    """Create a new skill from a template."""
    from mita.skills.manager import create_skill

    create_skill(name)


@skills_app.command("path")
def skills_path() -> None:
    """Show skill search paths."""
    from mita.skills.manager import show_paths

    show_paths()


# ── Hook commands ─────────────────────────────────────────────────

hooks_app = typer.Typer(help="Lifecycle hooks management.")
app.add_typer(hooks_app, name="hooks")


@hooks_app.command("list")
def hooks_list() -> None:
    """List configured hooks."""
    from mita.hooks.manager import list_hooks

    list_hooks()


@hooks_app.command("add")
def hooks_add(
    event: Annotated[str, typer.Argument(help="Hook event (e.g. on_file_write).")],
    command: Annotated[str, typer.Argument(help="Shell command to run.")],
    match: Annotated[str | None, typer.Option("--match", "-m", help="Glob pattern filter.")] = None,
) -> None:
    """Add a hook to project config."""
    from mita.hooks.manager import add_hook

    add_hook(event, command, match)


@hooks_app.command("remove")
def hooks_remove(
    event: Annotated[str, typer.Argument(help="Hook event to remove.")],
) -> None:
    """Remove hooks for an event from project config."""
    from mita.hooks.manager import remove_hooks

    remove_hooks(event)


# ── Plugin commands ───────────────────────────────────────────────

plugins_app = typer.Typer(help="MCP plugin management.")
app.add_typer(plugins_app, name="plugins")


@plugins_app.command("list")
def plugins_list() -> None:
    """List configured plugins and their tools."""
    import asyncio

    from mita.plugins.manager import PluginManager

    cfg = load_config()
    if not cfg.plugins:
        console.print("[dim]No plugins configured.[/dim]")
        return

    async def _list() -> None:
        mgr = PluginManager(cfg.plugins)
        started = await mgr.start_all(console=console)
        try:
            for plugin in cfg.plugins:
                connected = plugin.name in started
                status = "[green]connected[/green]" if connected else "[red]not connected[/red]"
                transport = plugin.transport
                target = plugin.command or plugin.url or ""
                console.print(f"  {plugin.name} ({transport}) — {status}")
                console.print(f"    {target}")

                if connected:
                    tools_map = await mgr.list_tools(plugin.name)
                    tools = tools_map.get(plugin.name, [])
                    if tools:
                        for t in tools:
                            console.print(f"    - {t['name']}: {t['description']}")
                    else:
                        console.print("    [dim]No tools[/dim]")
        finally:
            await mgr.stop_all()

    asyncio.run(_list())


@plugins_app.command("add")
def plugins_add(
    name: Annotated[str, typer.Argument(help="Plugin name.")],
    command: Annotated[
        str | None, typer.Option("--command", "-c", help="Command for stdio.")
    ] = None,
    url: Annotated[str | None, typer.Option("--url", "-u", help="URL for SSE.")] = None,
) -> None:
    """Add an MCP plugin to project config."""
    if not command and not url:
        console.print("[red]Provide --command or --url.[/red]")
        raise typer.Exit(1)

    from mita.config.defaults import ensure_project_config_path

    project_path = ensure_project_config_path()
    if not project_path:
        console.print("[red]Not in a project directory (no .git or .mita/).[/red]")
        raise typer.Exit(1)

    transport = "stdio" if command else "sse"
    # Build TOML block with proper escaping
    esc = escape_toml_string
    lines = [f'\n[[plugins]]\nname = "{esc(name)}"\ntransport = "{esc(transport)}"']
    if command:
        # Split command into executable + args (respects quoted arguments)
        parts = shlex.split(command)
        lines.append(f'command = "{esc(parts[0])}"')
        if len(parts) > 1:
            args_toml = ", ".join(f'"{esc(a)}"' for a in parts[1:])
            lines.append(f"args = [{args_toml}]")
    if url:
        lines.append(f'url = "{esc(url)}"')

    block = "\n".join(lines) + "\n"

    with open(project_path, "a") as f:
        f.write(block)

    console.print(f"[green]Plugin '{name}' added to {project_path}[/green]")


@plugins_app.command("remove")
def plugins_remove(
    name: Annotated[str, typer.Argument(help="Plugin name to remove.")],
) -> None:
    """Remove an MCP plugin from project config."""
    from mita.config.defaults import get_project_config_path

    project_path = get_project_config_path()
    if not project_path or not project_path.is_file():
        console.print("[red]No project config found.[/red]")
        raise typer.Exit(1)

    import tomllib

    with open(project_path, "rb") as f:
        data = tomllib.load(f)

    plugins = data.get("plugins", [])
    new_plugins = [p for p in plugins if p.get("name") != name]
    if len(new_plugins) == len(plugins):
        console.print(f"[yellow]Plugin '{name}' not found in project config.[/yellow]")
        return

    data["plugins"] = new_plugins
    write_toml(project_path, data)
    console.print(f"[green]Plugin '{name}' removed.[/green]")


@plugins_app.command("test")
def plugins_test(
    name: Annotated[str, typer.Argument(help="Plugin name to test.")],
) -> None:
    """Test connectivity to an MCP plugin."""
    import asyncio

    from mita.plugins.manager import PluginManager

    cfg = load_config()
    plugin = next((p for p in cfg.plugins if p.name == name), None)
    if not plugin:
        console.print(f"[red]Plugin '{name}' not found in configuration.[/red]")
        raise typer.Exit(1)

    async def _test() -> None:
        mgr = PluginManager([plugin])
        started = await mgr.start_all(console=console)
        try:
            if name not in started:
                console.print(f"[red]Failed to connect to '{name}'.[/red]")
                raise typer.Exit(1)

            result = await mgr.test_plugin(name)
            if result.get("ping"):
                console.print(f"[green]Plugin '{name}' is healthy.[/green]")
                tool_names = result.get("tool_names", [])
                console.print(f"  Tools: {len(tool_names)}")
                for t in tool_names:
                    console.print(f"    - {t}")
            else:
                console.print(f"[red]Plugin '{name}' ping failed.[/red]")
        finally:
            await mgr.stop_all()

    asyncio.run(_test())


# ── Ollama server commands ────────────────────────────────────────

ollama_app = typer.Typer(help="Ollama server management.")
app.add_typer(ollama_app, name="ollama")


@ollama_app.command("start")
def ollama_start() -> None:
    """Start the Ollama server in the background."""
    from mita.models.server import is_server_running, start_server

    cfg = load_config()
    if is_server_running(cfg.ollama.host):
        console.print("[green]Ollama is already running.[/green]")
        return

    if not start_server(host=cfg.ollama.host, console=console):
        raise typer.Exit(1)


@ollama_app.command("stop")
def ollama_stop() -> None:
    """Stop the Ollama server (only if started by Mita)."""
    from mita.models.server import is_managed, stop_server

    if not is_managed():
        console.print("[yellow]Ollama was not started by Mita — not stopping.[/yellow]")
        return

    stop_server(console=console)


@ollama_app.command("status")
def ollama_status() -> None:
    """Show Ollama server status."""
    from mita.models.server import find_ollama_binary, is_managed, is_server_running

    cfg = load_config()

    binary = find_ollama_binary()
    console.print(f"  Binary: {binary or '[red]not found[/red]'}")
    console.print(f"  Host:   {cfg.ollama.host}")

    running = is_server_running(cfg.ollama.host)
    if running:
        managed = is_managed()
        label = "running (managed by Mita)" if managed else "running"
        console.print(f"  Status: [green]{label}[/green]")
    else:
        console.print("  Status: [red]not running[/red]")

    console.print(f"  Auto-manage: {'yes' if cfg.ollama.auto_manage else 'no'}")


# ── Chat / Ask commands ───────────────────────────────────────────


def _ensure_project_trust(interactive: bool) -> None:
    """Prompt to trust a project whose config can run code (audit finding C1).

    If the current project's .mita/settings.toml declares hooks, plugins, or
    gate-weakening tool settings and the directory is not yet trusted, ask the user
    (when interactive) whether to trust it. Untrusted security-relevant keys are
    ignored by load_config until the directory is trusted.
    """
    import tomllib

    from mita.config.defaults import get_project_config_path
    from mita.config.trust import add_trusted, is_trusted, security_relevant_keys

    project_path = get_project_config_path()
    if project_path is None:
        return
    project_root = project_path.parent.parent
    if is_trusted(project_root):
        return
    try:
        with open(project_path, "rb") as f:
            project_data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        return
    keys = security_relevant_keys(project_data)
    if not keys:
        return

    key_list = ", ".join(keys)
    console.print(
        f"[yellow]This project's .mita/settings.toml can run code or change safety "
        f"settings (keys: {key_list}).[/yellow]"
    )
    if not interactive:
        console.print(
            "[yellow]Running non-interactively; these settings are ignored. "
            "Run 'mita chat' here once to trust this directory.[/yellow]"
        )
        return
    answer = console.input(f"Trust {project_root} and apply these settings? [y/N] ").strip().lower()
    if answer in ("y", "yes"):
        add_trusted(project_root)
        console.print(f"[green]Trusted {project_root}.[/green]")
    else:
        console.print("[yellow]Not trusted. These settings are ignored this session.[/yellow]")


@app.command("chat")
def chat_command(
    no_tools: Annotated[
        bool,
        typer.Option("--no-tools", help="Disable all tools."),
    ] = False,
    permission: Annotated[
        str | None,
        typer.Option(
            "--permission",
            "-p",
            help="Permission mode: ask (default), auto_edit, or trust.",
        ),
    ] = None,
) -> None:
    """Open an interactive chat session with the agent."""
    import asyncio

    from mita.agent.conversation import Conversation
    from mita.agent.loop import run_agent
    from mita.config.schema import PermissionMode
    from mita.tools.registry import ToolRegistry, create_default_registry
    from mita.ui.display import get_console
    from mita.ui.repl import repl_loop

    _ensure_project_trust(interactive=True)
    cfg = _safe_load_config()

    if permission is not None:
        try:
            cfg.tools.permission_mode = PermissionMode(permission)
        except ValueError:
            console.print(f"[red]Invalid permission mode: {permission}[/red]")
            console.print("Valid modes: ask, auto_edit, trust")
            raise typer.Exit(1)

    chat_console = get_console()

    if not _preflight_backend(cfg, chat_console):
        raise typer.Exit(1)

    from mita.plugins.manager import PluginManager

    conversation = Conversation()
    registry: ToolRegistry = ToolRegistry() if no_tools else create_default_registry()
    # Tools the user approved "always" persist for the whole chat session.
    session_approved: set[str] = set()

    async def _run_chat() -> None:
        nonlocal conversation

        # Start MCP plugins at session level (not per-call)
        plugin_mgr: PluginManager | None = None
        if cfg.plugins and not no_tools:
            plugin_mgr = PluginManager(cfg.plugins)
            await plugin_mgr.start_all(console=chat_console)
            await plugin_mgr.register_tools(registry)

        try:

            async def on_input(user_input: str) -> None:
                nonlocal conversation
                conversation = await run_agent(
                    user_input,
                    cfg,
                    chat_console,
                    conversation=conversation,
                    registry=registry,
                    session_approved=session_approved,
                )

            def on_clear() -> None:
                nonlocal conversation
                conversation.clear_non_system()

            await repl_loop(
                chat_console, on_input, on_clear=on_clear, skills_paths=cfg.skills_paths
            )
        finally:
            if plugin_mgr is not None:
                await plugin_mgr.stop_all()

    asyncio.run(_run_chat())


@app.command("ask")
def ask_command(
    prompt: Annotated[
        str | None,
        typer.Argument(help="The prompt to send to the agent."),
    ] = None,
    output: Annotated[
        OutputFormat | None,
        typer.Option("--output", "-o", help="Output format: rich, text, or json."),
    ] = None,
    no_tools: Annotated[
        bool,
        typer.Option("--no-tools", help="Disable all tools."),
    ] = False,
    yes: Annotated[
        bool,
        typer.Option(
            "--yes",
            "-y",
            help="Approve destructive actions without prompting (non-interactive).",
        ),
    ] = False,
) -> None:
    """Send a single prompt to the agent (non-interactive)."""
    import asyncio
    import json

    from mita.agent.conversation import Conversation, Role
    from mita.agent.loop import run_agent
    from mita.tools.registry import ToolRegistry, create_default_registry
    from mita.ui.display import get_console

    # Assemble prompt from argument and/or stdin
    parts: list[str] = []
    if prompt is not None:
        parts.append(prompt)
    if not sys.stdin.isatty():
        stdin_text = sys.stdin.read().strip()
        if stdin_text:
            parts.append(stdin_text)
    if not parts:
        console.print("[red]No prompt provided. Pass a prompt argument or pipe via stdin.[/red]")
        raise typer.Exit(1)

    full_prompt = "\n".join(parts)

    # Resolve output format: explicit flag wins, else auto-detect
    effective_output = output
    if effective_output is None:
        effective_output = OutputFormat.TEXT if not sys.stdout.isatty() else OutputFormat.RICH

    _ensure_project_trust(interactive=False)
    cfg = _safe_load_config()

    # Build the console for this run
    if effective_output == OutputFormat.TEXT:
        ask_console = Console(no_color=True, highlight=False)
        cfg.ui.stream = False
    elif effective_output == OutputFormat.JSON:
        ask_console = Console(file=StringIO(), no_color=True, highlight=False)
        cfg.ui.stream = False
    else:
        ask_console = get_console()

    if not _preflight_backend(cfg, ask_console):
        raise typer.Exit(1)

    async def _run_ask() -> Conversation:
        from mita.plugins.manager import PluginManager

        registry: ToolRegistry = ToolRegistry() if no_tools else create_default_registry()

        plugin_mgr: PluginManager | None = None
        if cfg.plugins and not no_tools:
            plugin_mgr = PluginManager(cfg.plugins)
            await plugin_mgr.start_all(console=ask_console)
            await plugin_mgr.register_tools(registry)

        try:
            return await run_agent(
                full_prompt, cfg, ask_console, registry=registry, auto_confirm=yes
            )
        finally:
            if plugin_mgr is not None:
                await plugin_mgr.stop_all()

    conversation = asyncio.run(_run_ask())

    # Extract the last assistant message
    last_assistant = ""
    for msg in reversed(conversation.messages):
        if msg.role == Role.ASSISTANT and msg.content:
            last_assistant = msg.content
            break

    if effective_output == OutputFormat.TEXT:
        print(last_assistant)  # noqa: T201
    elif effective_output == OutputFormat.JSON:
        print(  # noqa: T201
            json.dumps({"response": last_assistant, "model": cfg.model.default})
        )


# ── Doctor command ────────────────────────────────────────────────


@app.command("doctor")
def doctor_command() -> None:
    """Check system health and configuration."""
    import platform

    from mita.ui.display import display_error_with_suggestion

    doc_console = Console()

    # 1. Python version
    py_ver = platform.python_version()
    py_tuple = tuple(int(x) for x in py_ver.split(".")[:2])
    if py_tuple >= (3, 11):
        doc_console.print(f"  [green]\u2713[/green] Python {py_ver}")
    else:
        doc_console.print(f"  [red]\u2717[/red] Python {py_ver}")
        display_error_with_suggestion(
            doc_console,
            "Python 3.11+ is required",
            "Install Python 3.11 or later",
        )

    cfg = _safe_load_config()

    from mita.config.schema import LLMProvider

    # 2-5. Backend health \u2014 Ollama gets the daemon/binary/model checks; every other
    # provider gets a generic reachability check and no Ollama assumptions (finding C5).
    if cfg.llm.provider == LLMProvider.OLLAMA:
        from mita.models.server import find_ollama_binary, is_server_running

        binary = find_ollama_binary()
        if binary:
            doc_console.print(f"  [green]\u2713[/green] Ollama binary found ({binary})")
        else:
            doc_console.print("  [red]\u2717[/red] Ollama binary not found")
            display_error_with_suggestion(
                doc_console, "Ollama is not installed", "Install from https://ollama.com"
            )

        if is_server_running(cfg.ollama.host):
            doc_console.print("  [green]\u2713[/green] Ollama server running")
        else:
            doc_console.print("  [red]\u2717[/red] Ollama not running")
            display_error_with_suggestion(
                doc_console,
                "Ollama server is not running",
                "Run 'mita ollama start' or 'ollama serve'",
            )

        _check_model_installed(doc_console, cfg.model.default, "Default model", cfg)
        _check_model_installed(doc_console, cfg.model.embedding, "Embedding model", cfg)
    else:
        from mita.llm.health import backend_reachable
        from mita.llm.providers import resolve_backend

        backend = resolve_backend(cfg)
        label = f"{cfg.llm.provider.value} backend at {backend.api_base}"
        if backend_reachable(backend.api_base, backend.api_key):
            doc_console.print(f"  [green]\u2713[/green] {label} reachable")
        else:
            doc_console.print(f"  [red]\u2717[/red] {label} unreachable")
            display_error_with_suggestion(
                doc_console,
                f"Cannot reach the {cfg.llm.provider.value} backend",
                "Start the server or fix [llm] base_url",
            )

    # 6. Config loads without error (already verified by the load_config() above)
    doc_console.print("  [green]\u2713[/green] Config loaded successfully")

    # 7. Memory files discoverable
    from mita.memory.discovery import discover_memory_files

    mem_files = discover_memory_files()
    if mem_files:
        doc_console.print(f"  [green]\u2713[/green] Memory files found ({len(mem_files)})")
    else:
        doc_console.print("  [yellow]![/yellow] No MITA.md memory files found")

    # 8. Index exists
    from pathlib import Path

    from mita.index.store import IndexStore

    index_dir = Path.cwd() / ".mita" / "index"
    store = IndexStore(index_dir)
    if store.exists():
        doc_console.print("  [green]\u2713[/green] Codebase index exists")
    else:
        doc_console.print("  [yellow]![/yellow] No codebase index")
        display_error_with_suggestion(
            doc_console,
            "Codebase index not built",
            "Run 'mita index build' to create it",
        )


def _check_model_installed(doc_console: Console, model_name: str, label: str, cfg: object) -> None:
    """Check if an Ollama model is installed and print status."""
    from mita.config.schema import MitaConfig
    from mita.models.ollama_client import OllamaClient
    from mita.models.server import is_server_running
    from mita.ui.display import display_error_with_suggestion

    assert isinstance(cfg, MitaConfig)

    if not is_server_running(cfg.ollama.host):
        doc_console.print(
            f"  [yellow]![/yellow] {label} ({model_name}) — cannot check (server not running)"
        )
        return

    try:
        client = OllamaClient(host=cfg.ollama.host, timeout=cfg.ollama.timeout)
        if client.is_model_installed(model_name):
            doc_console.print(f"  [green]\u2713[/green] {label} ({model_name}) installed")
        else:
            doc_console.print(f"  [red]\u2717[/red] {label} ({model_name}) not installed")
            display_error_with_suggestion(
                doc_console,
                f"{label} '{model_name}' is not installed",
                f"Run 'mita models pull {model_name}'",
            )
    except (ConnectionError, OSError):
        doc_console.print(f"  [yellow]![/yellow] {label} ({model_name}) — cannot check")


# ── Helpers ───────────────────────────────────────────────────────
