"""CLI command handlers for codebase indexing."""

from __future__ import annotations

import time
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.table import Table

from mita.config.loader import load_config
from mita.index.embeddings import EmbeddingClient
from mita.index.parser import parse_codebase
from mita.index.retriever import Retriever
from mita.index.store import IndexStore

console = Console()


def _get_index_dir() -> Path:
    return Path.cwd() / ".mita" / "index"


async def build_index(force: bool = False, pull_model_fn: object | None = None) -> None:
    """Build the codebase index.

    Args:
        force: Rebuild even if index already exists.
        pull_model_fn: Async callback to pull the embedding model if missing.
            Signature: async (MitaConfig) -> bool. Called from the CLI layer
            so it can import from mita.models without violating dependency rules.
    """
    config = load_config()
    index_dir = _get_index_dir()
    store = IndexStore(index_dir)

    # Check if index already exists
    if store.exists() and not force:
        console.print("[yellow]Index already exists. Use --force to rebuild.[/yellow]")
        return

    # Check embedding model availability
    embedder = EmbeddingClient(config)
    if not await embedder.is_model_available():
        if pull_model_fn is not None:
            if not await pull_model_fn(config):  # type: ignore[operator]
                return
        else:
            console.print("[red]Embedding model not available.[/red]")
            return

    start_time = time.monotonic()

    # Parse codebase
    root = Path.cwd()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Parsing codebase...", total=None)
        chunks = parse_codebase(root, config.index)

    if not chunks:
        console.print("[yellow]No code chunks found to index.[/yellow]")
        return

    console.print(f"Found {len(chunks)} chunks from codebase.")

    # Generate embeddings
    texts = [c.content for c in chunks]
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("Generating embeddings...", total=len(texts))
        embeddings: list[list[float]] = []
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                batch_embeddings = await embedder.embed_texts(batch)
            except (ConnectionError, TimeoutError, OSError) as e:
                console.print(
                    f"\n[red]Embedding failed at batch {i // batch_size + 1}: {e}[/red]"
                )
                console.print("[red]Index build aborted to prevent data corruption.[/red]")
                return
            embeddings.extend(batch_embeddings)
            progress.update(task, completed=min(i + batch_size, len(texts)))

    # Verify alignment before attaching
    if len(embeddings) != len(chunks):
        console.print(
            f"[red]Embedding count mismatch ({len(embeddings)} vs {len(chunks)} chunks). "
            "Index build aborted.[/red]"
        )
        return

    # Attach embeddings to chunks
    for chunk, embedding in zip(chunks, embeddings):
        chunk.embedding = embedding

    # Store in LanceDB
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Writing index...", total=None)
        await store.create_or_replace(chunks)

    elapsed = time.monotonic() - start_time
    file_count = len({c.file_path for c in chunks})
    console.print(
        f"[green]Indexed {len(chunks)} chunks from {file_count} files in {elapsed:.1f}s.[/green]"
    )


async def show_index_status() -> None:
    """Show index statistics."""
    config = load_config()
    store = IndexStore(_get_index_dir())
    stats = await store.status()

    if not stats["exists"]:
        console.print("[yellow]No index found. Run 'mita index build' first.[/yellow]")
        return

    table = Table(title="Index Status")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("Chunks", str(stats["chunks"]))
    table.add_row("Files", str(stats["files"]))
    table.add_row("Location", str(_get_index_dir()))
    table.add_row("Embedding model", config.model.embedding)
    console.print(table)


async def search_index(query: str, top_k: int = 10) -> None:
    """Search the index and display results."""
    config = load_config()
    retriever = Retriever(config, index_dir=_get_index_dir())

    if not retriever.is_available():
        console.print("[yellow]No index found. Run 'mita index build' first.[/yellow]")
        return

    results = await retriever.retrieve(query, top_k=top_k)

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    console.print(f"[bold]Found {len(results)} results:[/bold]\n")
    for i, r in enumerate(results, 1):
        c = r.chunk
        header = f"{i}. {c.file_path}:{c.start_line}-{c.end_line}"
        if c.symbol:
            header += f" ({c.symbol})"
        header += f"  [dim]score: {r.score:.3f}[/dim]"
        console.print(header)
        console.print(Syntax(c.content, c.language, line_numbers=True, start_line=c.start_line))
        console.print()


async def clear_index() -> None:
    """Clear the index."""
    store = IndexStore(_get_index_dir())

    if not store.exists():
        console.print("[yellow]No index found.[/yellow]")
        return

    confirm = console.input("Delete the index? [y/N] ").strip().lower()
    if confirm in ("y", "yes"):
        await store.clear()
        console.print("[green]Index cleared.[/green]")
    else:
        console.print("Cancelled.")
