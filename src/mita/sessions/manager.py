"""CLI command handlers for chat session management."""

from __future__ import annotations

import time
from pathlib import Path

from rich.console import Console
from rich.table import Table

from mita.sessions import get_sessions_dir
from mita.sessions.store import SessionMeta, SessionStore

console = Console()

_CWD_FOOTNOTE = "[dim]* saved in a different working directory[/dim]"


def list_sessions_command() -> None:
    """Show saved sessions, newest activity first."""
    store = SessionStore(get_sessions_dir())
    metas = store.list_metas()

    if not metas:
        console.print("[yellow]No saved sessions. Start one with 'mita chat'.[/yellow]")
        return

    table, other_cwd = _sessions_table(metas, title="Saved Sessions")
    console.print(table)
    if other_cwd:
        console.print(_CWD_FOOTNOTE)


def pick_session(store: SessionStore, out: Console, limit: int = 10) -> str | None:
    """Show recent sessions and prompt for a pick; returns a session id or None."""
    metas = store.list_metas()[:limit]
    if not metas:
        out.print("[yellow]No saved sessions.[/yellow]")
        return None

    table, other_cwd = _sessions_table(metas, title="Recent Sessions", numbered=True)
    out.print(table)
    if other_cwd:
        out.print(_CWD_FOOTNOTE)

    try:
        pick = out.input(f"Resume which session? [1-{len(metas)}, blank to cancel] ").strip()
    except (EOFError, KeyboardInterrupt):
        return None
    if not pick:
        return None
    try:
        idx = int(pick)
    except ValueError:
        idx = 0
    if not 1 <= idx <= len(metas):
        out.print("[red]Invalid selection.[/red]")
        return None
    return metas[idx - 1].id


def clear_sessions_command(yes: bool = False) -> None:
    """Delete all saved sessions, with confirmation unless yes=True."""
    store = SessionStore(get_sessions_dir())
    count = sum(1 for _ in get_sessions_dir().glob("*.json"))

    if count == 0:
        console.print("[yellow]No saved sessions.[/yellow]")
        return

    if not yes:
        try:
            confirm = console.input(f"Delete {count} saved session(s)? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            confirm = ""
        if confirm not in ("y", "yes"):
            console.print("Cancelled.")
            return

    removed = store.clear()
    console.print(f"[green]Deleted {removed} session(s).[/green]")


def _sessions_table(
    metas: list[SessionMeta], *, title: str, numbered: bool = False
) -> tuple[Table, bool]:
    """Render sessions as a table; returns (table, any row saved in another cwd)."""
    cwd = str(Path.cwd())
    table = Table(title=title)
    if numbered:
        table.add_column("#", justify="right")
    table.add_column("ID", style="bold")
    table.add_column("Title")
    table.add_column("Last activity")
    table.add_column("Msgs", justify="right")
    table.add_column("≈Tokens", justify="right")
    table.add_column("Model")

    other_cwd = False
    for i, meta in enumerate(metas, 1):
        session_id = meta.id
        if meta.cwd != cwd:
            session_id += " *"
            other_cwd = True
        row = [
            session_id,
            meta.title or "(untitled)",
            _format_age(time.time() - meta.updated_at),
            str(meta.message_count),
            str(meta.total_tokens),
            meta.model,
        ]
        if numbered:
            row.insert(0, str(i))
        table.add_row(*row)
    return table, other_cwd


def _format_age(seconds: float) -> str:
    """Humanize an age in seconds (e.g. '5m ago', '3d ago')."""
    seconds = max(0.0, seconds)
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{int(seconds // 60)}m ago"
    if seconds < 86400:
        return f"{int(seconds // 3600)}h ago"
    return f"{int(seconds // 86400)}d ago"
