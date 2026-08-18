"""Interactive REPL input loop with prompt_toolkit."""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from rich.console import Console

from mita.ui.display import display_goodbye, display_welcome

# Built-in REPL commands that should NOT be treated as skill invocations.
_BUILTIN_COMMANDS = frozenset({"/quit", "/exit", "/q", "/clear", "/reload", "/help", "/resume"})


async def repl_loop(
    console: Console,
    on_input: Callable[[str], Coroutine[Any, Any, None]],
    on_clear: Callable[[], None] | None = None,
    skills_paths: list[str] | None = None,
    on_reload: Callable[[], None] | None = None,
    on_resume: Callable[[str], Coroutine[Any, Any, None]] | None = None,
) -> None:
    """Run the interactive REPL.

    Args:
        console: Rich console for output.
        on_input: Async callback called with each user input line.
        on_clear: Callback invoked when the user types /clear.
        skills_paths: Skill search paths for ``/`` prefix invocation.
        on_reload: Callback invoked when the user types /reload (re-read memory/config).
        on_resume: Async callback invoked when the user types /resume [session_id];
            receives the session id or an empty string.
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

        # Parse once: builtin commands dispatch on the first word.
        cmd, _, arg = user_input.partition(" ")
        cmd = cmd.lower()

        if cmd in ("/quit", "/exit", "/q"):
            display_goodbye(console)
            break

        if cmd == "/help":
            console.print(
                "[mita.dim]Commands: /clear (reset conversation), /reload (re-read "
                "memory + config), /resume [id] (resume a saved session), /quit. "
                "Anything else is sent to the agent; /<name> runs a skill.[/mita.dim]"
            )
            continue

        if cmd == "/clear":
            if on_clear is not None:
                on_clear()
            console.print("[mita.dim]Conversation cleared.[/mita.dim]")
            continue

        if cmd == "/reload":
            if on_reload is not None:
                on_reload()
            console.print("[mita.dim]Reloaded memory and config.[/mita.dim]")
            continue

        if cmd == "/resume":
            if on_resume is not None:
                await on_resume(arg.strip())
            continue

        # Skill invocation: /name [args]
        if cmd.startswith("/") and cmd not in _BUILTIN_COMMANDS:
            rendered = _try_render_skill(user_input, skills_paths or [], console)
            if rendered is not None:
                try:
                    await on_input(rendered)
                except KeyboardInterrupt:
                    console.print("\n[Interrupted]")
                continue

        try:
            await on_input(user_input)
        except KeyboardInterrupt:
            console.print("\n[Interrupted]")


def _try_render_skill(user_input: str, skills_paths: list[str], console: Console) -> str | None:
    """Attempt to find and render a skill from user input.

    Returns the rendered prompt string, or ``None`` if no matching skill was found.
    """
    from mita.skills.executor import render_skill
    from mita.skills.loader import discover_skills, find_skill

    skills = discover_skills(skills_paths)
    skill = find_skill(skills, user_input)
    if skill is None:
        return None

    rendered = render_skill(skill, user_input)
    console.print(f"[dim]Running skill: {skill.frontmatter.name}[/dim]")
    return rendered
