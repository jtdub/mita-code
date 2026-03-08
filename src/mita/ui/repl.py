"""Interactive REPL input loop with prompt_toolkit."""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from rich.console import Console

from mita.ui.display import display_goodbye, display_welcome


async def repl_loop(
    console: Console,
    on_input: Callable[[str], Coroutine[Any, Any, None]],
    on_clear: Callable[[], None] | None = None,
) -> None:
    """Run the interactive REPL.

    Args:
        console: Rich console for output.
        on_input: Async callback called with each user input line.
        on_clear: Callback invoked when the user types /clear.
    """
    display_welcome(console)

    session: PromptSession[str] = PromptSession(history=InMemoryHistory())

    while True:
        try:
            user_input = await session.prompt_async("mita> ")
        except (EOFError, KeyboardInterrupt):
            display_goodbye(console)
            break

        user_input = user_input.strip()
        if not user_input:
            continue

        if user_input.lower() in ("/quit", "/exit", "/q"):
            display_goodbye(console)
            break

        if user_input.lower() == "/clear":
            if on_clear is not None:
                on_clear()
            console.print("[mita.dim]Conversation cleared.[/mita.dim]")
            continue

        await on_input(user_input)
