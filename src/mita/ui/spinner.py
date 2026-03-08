"""Thinking/loading indicators for the terminal."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner as RichSpinner
from rich.text import Text


@contextmanager
def thinking_spinner(
    console: Console,
    message: str = "Thinking...",
) -> Generator[Live, None, None]:
    """Display a spinner while the agent is thinking.

    Usage:
        with thinking_spinner(console):
            await long_operation()
    """
    spinner = RichSpinner("dots", text=Text(message, style="mita.dim"))
    with Live(spinner, console=console, transient=True, refresh_per_second=10) as live:
        yield live
