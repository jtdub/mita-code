"""CLI command handlers for model management."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.table import Table

from mita.config.loader import load_config
from mita.models.hardware import detect_hardware
from mita.models.ollama_client import OllamaClient
from mita.models.recommender import recommend_models
from mita.models.registry import find_model

console = Console()


def _get_client() -> OllamaClient:
    """Get an OllamaClient from the current config."""
    cfg = load_config()
    return OllamaClient(host=cfg.ollama.host)


def _check_ollama(client: OllamaClient) -> bool:
    """Check if Ollama is running, print error if not."""
    if not client.is_running():
        console.print(
            "[red]Cannot connect to Ollama.[/red]\n"
            "Make sure Ollama is running: [bold]ollama serve[/bold]"
        )
        return False
    return True


def list_models() -> None:
    """List all installed Ollama models."""
    client = _get_client()
    if not _check_ollama(client):
        return

    models = client.list_models()
    if not models:
        console.print(
            "[dim]No models installed. "
            "Run [bold]mita models pull <name>[/bold] to get started.[/dim]"
        )
        return

    table = Table(title="Installed Models")
    table.add_column("Name", style="bold")
    table.add_column("Size", justify="right")
    table.add_column("Parameters", justify="right")
    table.add_column("Quantization")
    table.add_column("Family")

    for m in models:
        table.add_row(
            m.name,
            f"{m.size_gb:.1f} GB",
            m.parameter_size,
            m.quantization,
            m.family,
        )

    console.print(table)


def pull_model(name: str) -> None:
    """Pull a model from the Ollama registry."""
    client = _get_client()
    if not _check_ollama(client):
        return

    console.print(f"Pulling [bold]{name}[/bold]...")

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("Downloading", total=100)

        for update in client.pull(name, stream=True):
            status = update.get("status", "")
            completed = update.get("completed", 0)
            total = update.get("total", 0)

            if total > 0:
                pct = (completed / total) * 100
                progress.update(task, completed=pct, description=status)
            else:
                progress.update(task, description=status)

    console.print(f"[green]Successfully pulled {name}[/green]")


def remove_model(name: str) -> None:
    """Remove an installed model."""
    client = _get_client()
    if not _check_ollama(client):
        return

    try:
        client.remove(name)
        console.print(f"[green]Removed {name}[/green]")
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]Failed to remove {name}: {e}[/red]")


def show_model_info(name: str) -> None:
    """Show details about a model."""
    # Check registry first
    card = find_model(name)
    if card:
        table = Table(title=f"Registry: {name}", show_header=False)
        table.add_column("Field", style="bold")
        table.add_column("Value")
        table.add_row("Family", card.family)
        table.add_row("Parameters", card.param_count)
        table.add_row("Context Window", f"{card.context_window:,}")
        table.add_row("Min RAM", f"{card.min_ram_gb:.0f} GB")
        table.add_row("Min VRAM", f"{card.min_vram_gb:.0f} GB")
        table.add_row("Tool Calls", "Yes" if card.tool_call_support else "No")
        table.add_row("Quantization", card.quantization)
        table.add_row("Description", card.description)
        table.add_row("Tags", ", ".join(card.tags))
        console.print(table)
        console.print()

    # Also show Ollama details if installed
    client = _get_client()
    if not _check_ollama(client):
        return

    try:
        info = client.show(name)
        details = info.get("details", {})
        if isinstance(details, dict) and details:
            table = Table(title=f"Ollama: {name}", show_header=False)
            table.add_column("Field", style="bold")
            table.add_column("Value")
            for k, v in details.items():
                table.add_row(str(k), str(v))
            console.print(table)
        elif not card:
            console.print(f"[dim]Model {name} is installed but not in the curated registry.[/dim]")
    except Exception as e:  # noqa: BLE001
        if not card:
            console.print(f"[red]Model {name} not found in registry or Ollama: {e}[/red]")


def show_recommendations() -> None:
    """Detect hardware and recommend models."""
    hw = detect_hardware()

    # Show hardware summary
    hw_panel = Panel(
        f"[bold]CPU:[/bold] {hw.cpu_name} ({hw.cpu_cores} cores)\n"
        f"[bold]RAM:[/bold] {hw.ram_gb:.1f} GB\n"
        f"[bold]GPU:[/bold] {hw.gpus[0].name if hw.gpus else 'None detected'}\n"
        f"[bold]VRAM:[/bold] {hw.available_vram_gb:.1f} GB effective\n"
        f"[bold]OS:[/bold] {hw.os}"
        + ("\n[bold]Unified Memory:[/bold] Yes" if hw.unified_memory else ""),
        title="Hardware Detected",
        border_style="blue",
    )
    console.print(hw_panel)
    console.print()

    # Show recommendations
    recs = recommend_models(hw)
    if not recs:
        console.print(
            "[yellow]No models in the registry fit your hardware.[/yellow]\n"
            "You may need more RAM/VRAM, or try a smaller model manually."
        )
        return

    table = Table(title="Recommended Models")
    table.add_column("Model", style="bold")
    table.add_column("Params", justify="right")
    table.add_column("Context", justify="right")
    table.add_column("Fit", justify="right")
    table.add_column("Notes")

    for rec in recs:
        # Color-code fit score
        score = rec.fit_score
        if score >= 0.8:
            fit_style = "green"
        elif score >= 0.6:
            fit_style = "yellow"
        else:
            fit_style = "red"

        table.add_row(
            rec.model.name,
            rec.model.param_count,
            f"{rec.model.context_window:,}",
            f"[{fit_style}]{score:.0%}[/{fit_style}]",
            rec.notes,
        )

    console.print(table)


def set_default_model(name: str) -> None:
    """Set the default model (prints instruction since config is TOML-based)."""
    card = find_model(name)
    if card:
        console.print(f"[green]Model [bold]{name}[/bold] is in the registry.[/green]")
    else:
        console.print(f"[yellow]Model [bold]{name}[/bold] is not in the curated registry.[/yellow]")

    console.print(
        f"\nTo set as default, add to your config:\n\n"
        f'  [bold]mita config set model.default "{name}"[/bold]\n\n'
        f"Or edit your config file directly:\n\n"
        f'  [dim][model]\n  default = "{name}"[/dim]'
    )


def show_hardware() -> None:
    """Show detected hardware information."""
    hw = detect_hardware()

    table = Table(title="Hardware Information", show_header=False)
    table.add_column("Field", style="bold")
    table.add_column("Value")
    table.add_row("OS", hw.os)
    table.add_row("CPU", hw.cpu_name)
    table.add_row("CPU Cores", str(hw.cpu_cores))
    table.add_row("RAM", f"{hw.ram_gb:.1f} GB")
    table.add_row("Apple Silicon", "Yes" if hw.apple_silicon else "No")
    table.add_row("Unified Memory", "Yes" if hw.unified_memory else "No")

    for i, gpu in enumerate(hw.gpus):
        prefix = f"GPU {i}" if len(hw.gpus) > 1 else "GPU"
        table.add_row(prefix, f"{gpu.name} ({gpu.vram_gb:.1f} GB, {gpu.vendor})")

    if not hw.gpus:
        table.add_row("GPU", "None detected")

    table.add_row("Effective VRAM", f"{hw.available_vram_gb:.1f} GB")

    console.print(table)
