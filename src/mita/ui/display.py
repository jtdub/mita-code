"""Rich terminal rendering: markdown, code blocks, diffs, tool calls."""

from __future__ import annotations

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from mita.tools.schema import ToolCall, ToolResult
from mita.ui.theme import MITA_THEME

# Module-level console with mita theme
_console: Console | None = None


def get_console() -> Console:
    """Get or create the themed console."""
    global _console  # noqa: PLW0603
    if _console is None:
        _console = Console(theme=MITA_THEME)
    return _console


def display_welcome(console: Console) -> None:
    """Display the welcome banner."""
    console.print(
        Panel(
            "[mita.prompt]mita[/mita.prompt] — local-first coding assistant\n"
            "[mita.dim]Type your prompt, or /quit to exit.[/mita.dim]",
            border_style="mita.info",
        )
    )


def display_goodbye(console: Console) -> None:
    """Display the exit message."""
    console.print("[mita.dim]Goodbye![/mita.dim]")


def display_markdown(console: Console, text: str) -> None:
    """Render markdown text in the terminal."""
    if not text.strip():
        return
    md = Markdown(text)
    console.print(md)


def display_streaming_token(console: Console, token: str) -> None:
    """Display a single streaming token (no newline)."""
    console.print(token, end="", highlight=False)


def display_streaming_end(console: Console) -> None:
    """End a streaming output block."""
    console.print()  # final newline


def display_tool_call(console: Console, tool_call: ToolCall) -> None:
    """Display a tool call that is about to be executed."""
    args_str = ", ".join(f"{k}={v!r}" for k, v in tool_call.arguments.items())
    # Truncate long args for display
    if len(args_str) > 200:
        args_str = args_str[:200] + "..."
    console.print(
        f"  [mita.tool_name]{tool_call.name}[/mita.tool_name]({args_str})",
    )


def display_tool_result(console: Console, result: ToolResult) -> None:
    """Display the result of a tool execution."""
    if result.success:
        if result.output:
            # Show truncated output
            output = result.output
            if len(output) > 500:
                output = output[:500] + "\n..."
            console.print(f"  [mita.dim]{output}[/mita.dim]")
    else:
        console.print(f"  [mita.tool_error]Error: {result.error}[/mita.tool_error]")


def display_error(console: Console, message: str) -> None:
    """Display an error message."""
    console.print(f"[mita.error]{message}[/mita.error]")


def display_warning(console: Console, message: str) -> None:
    """Display a warning message."""
    console.print(f"[mita.warning]{message}[/mita.warning]")


def display_token_usage(console: Console, total_tokens: int) -> None:
    """Display token usage after a response (estimated)."""
    console.print(f"[mita.token_count]({total_tokens:,} tokens)[/mita.token_count]")


def display_response_stats(
    console: Console,
    prompt_tokens: int,
    completion_tokens: int,
    total_time: float,
    ttft: float | None = None,
) -> None:
    """Display detailed response statistics after an LLM call."""
    parts: list[str] = []

    parts.append(f"prompt: {prompt_tokens:,}")
    parts.append(f"completion: {completion_tokens:,}")

    if ttft is not None:
        parts.append(f"ttft: {ttft:.1f}s")

    parts.append(f"total: {total_time:.1f}s")

    if completion_tokens > 0 and total_time > 0:
        tps = completion_tokens / total_time
        parts.append(f"{tps:.1f} tok/s")

    console.print(f"[mita.token_count]({' · '.join(parts)})[/mita.token_count]")


async def prompt_user_confirm(console: Console, question: str) -> bool:
    """Ask the user for confirmation before a destructive action.

    Returns True if the user approves, False otherwise.
    """
    console.print(f"[mita.warning]{question}[/mita.warning] ", end="")
    try:
        response = console.input("[y/N] ")
        return response.strip().lower() in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        console.print()
        return False
