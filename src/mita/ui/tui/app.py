"""Textual TUI app for Mita Code — a second UISink frontend (finding: TUI)."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Header, Input, Label, RichLog

from mita.agent.conversation import Conversation
from mita.tools.schema import ToolCall, ToolResult
from mita.ui.sink import AbstractBusy, ConfirmDecision, ConfirmRequest

if TYPE_CHECKING:
    from mita.config.schema import MitaConfig
    from mita.tools.registry import ToolRegistry

# NOTE: instance attributes are prefixed `_mita_` because Textual's App reserves several
# private names (e.g. _log, _registry); colliding with them breaks the framework.


class ConfirmScreen(ModalScreen[ConfirmDecision]):
    """Modal asking the user to approve a destructive tool call."""

    def __init__(self, request: ConfirmRequest) -> None:
        super().__init__()
        self._request = request

    def compose(self) -> ComposeResult:
        yield Label(f"Allow {self._request.tool_name}?\n\n{self._request.summary}", id="confirm-q")
        with Horizontal(id="confirm-buttons"):
            yield Button("Yes", variant="success", id="once")
            yield Button("Always", variant="primary", id="always")
            yield Button("No", variant="error", id="deny")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        mapping = {
            "once": ConfirmDecision.ALLOW_ONCE,
            "always": ConfirmDecision.ALLOW_SESSION,
            "deny": ConfirmDecision.DENY,
        }
        self.dismiss(mapping.get(event.button.id or "deny", ConfirmDecision.DENY))


class MitaTUI(App[None]):
    """Full-screen chat TUI. Implements the UISink protocol the agent loop calls."""

    CSS = """
    #log { border: round $panel; padding: 0 1; }
    #prompt { dock: bottom; }
    ConfirmScreen { align: center middle; }
    #confirm-q { padding: 1 2; }
    #confirm-buttons { height: auto; align: center middle; }
    #confirm-buttons Button { margin: 1 2; }
    """

    BINDINGS = [("ctrl+c", "quit", "Quit")]

    def __init__(
        self,
        config: MitaConfig,
        registry: ToolRegistry,
        session_approved: set[str] | None = None,
    ) -> None:
        super().__init__()
        self._mita_config = config
        self._mita_registry = registry
        self._mita_session_approved = session_approved if session_approved is not None else set()
        self._mita_conversation: Conversation | None = None
        self._mita_stream_buffer: list[str] = []
        self._mita_hooks_console = Console(quiet=True)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield RichLog(id="log", wrap=True, markup=False, highlight=False)
        yield Input(placeholder="Ask mita…  (/quit, /clear)", id="prompt")
        yield Footer()

    def on_mount(self) -> None:
        self._out.write(Text("mita — local-first coding assistant. Type a prompt to begin.", "dim"))
        self.query_one("#prompt", Input).focus()

    @property
    def _out(self) -> RichLog:
        return self.query_one("#log", RichLog)

    # ── UISink implementation ────────────────────────────────────────
    def notice(self, message: str) -> None:
        self._out.write(Text(message, "yellow"))

    def error(self, message: str) -> None:
        self._out.write(Text(message, "red"))

    def assistant_message(self, markdown: str) -> None:
        if markdown.strip():
            self._out.write(Markdown(markdown))

    def stream_token(self, token: str) -> None:
        self._mita_stream_buffer.append(token)

    def stream_end(self) -> None:
        text = "".join(self._mita_stream_buffer)
        self._mita_stream_buffer.clear()
        if text.strip():
            self._out.write(Markdown(text))

    def tool_call(self, tool_call: ToolCall) -> None:
        args = ", ".join(f"{k}={v!r}" for k, v in tool_call.arguments.items())
        self._out.write(Text(f"→ {tool_call.name}({args[:200]})", "cyan"))

    def tool_result(self, result: ToolResult) -> None:
        if result.success:
            out = (result.output or "")[:500]
            if out:
                self._out.write(Text(f"  {out}", "dim"))
        else:
            self._out.write(Text(f"  error: {result.error}", "red"))

    def response_stats(
        self, prompt_tokens: int, completion_tokens: int, total_time: float, ttft: float | None
    ) -> None:
        self._out.write(
            Text(
                f"  ({prompt_tokens} prompt · {completion_tokens} completion · {total_time:.1f}s)",
                "dim",
            )
        )

    def busy(self, message: str) -> AbstractBusy:
        # Tokens stream live; no blocking spinner needed in the TUI.
        return contextlib.nullcontext()

    async def confirm(self, request: ConfirmRequest) -> ConfirmDecision:
        decision = await self.push_screen_wait(ConfirmScreen(request))
        return decision or ConfirmDecision.DENY

    # ── Input handling ───────────────────────────────────────────────
    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        if not text:
            return
        if text.lower() in ("/quit", "/exit", "/q"):
            self.exit()
            return
        if text.lower() == "/clear":
            if self._mita_conversation is not None:
                self._mita_conversation.clear_non_system()
            self._out.clear()
            return

        self._out.write(Text(f"❯ {escape(text)}", "bold"))
        event.input.disabled = True
        self.run_worker(self._run_turn(text), exclusive=True)

    async def _run_turn(self, text: str) -> None:
        from mita.agent.loop import run_agent

        try:
            self._mita_conversation = await run_agent(
                text,
                self._mita_config,
                self._mita_hooks_console,
                conversation=self._mita_conversation,
                registry=self._mita_registry,
                session_approved=self._mita_session_approved,
                sink=self,
            )
        finally:
            prompt = self.query_one("#prompt", Input)
            prompt.disabled = False
            prompt.focus()


def run_tui(
    config: MitaConfig,
    registry: ToolRegistry,
    session_approved: set[str] | None = None,
) -> None:
    """Launch the Textual TUI (blocking)."""
    MitaTUI(config, registry, session_approved).run()
