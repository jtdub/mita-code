"""Thinking/loading indicators for the terminal."""

from __future__ import annotations

import time
from collections.abc import Generator
from contextlib import contextmanager

from rich.console import Console, RenderableType
from rich.live import Live
from rich.spinner import Spinner as RichSpinner
from rich.table import Table
from rich.text import Text


class _TimedSpinner:
    """A spinner that shows elapsed time alongside the message."""

    def __init__(self, message: str = "Thinking...") -> None:
        self._message = message
        self._start = time.monotonic()
        self._spinner = RichSpinner("dots")

    def __rich__(self) -> RenderableType:
        elapsed = time.monotonic() - self._start
        table = Table.grid(padding=(0, 1))
        table.add_row(
            self._spinner,
            Text(self._message, style="mita.dim"),
            Text(f"({elapsed:.0f}s)", style="dim"),
        )
        return table


@contextmanager
def thinking_spinner(
    console: Console,
    message: str = "Thinking...",
) -> Generator[Live, None, None]:
    """Display a spinner with elapsed time while the agent is thinking.

    Usage:
        with thinking_spinner(console):
            await long_operation()
    """
    spinner = _TimedSpinner(message)
    with Live(spinner, console=console, transient=True, refresh_per_second=4) as live:
        yield live
