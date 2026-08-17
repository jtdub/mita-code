"""UISink — the interface between the agent loop and any frontend (finding: TUI).

The agent loop emits UI events through this protocol instead of calling Rich display
functions directly, so a Textual TUI can be a second implementation with no loop changes.
``RichConsoleSink`` is the default adapter over the existing terminal renderers.
"""

from __future__ import annotations

import enum
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from rich.console import Console

from mita.tools.schema import ToolCall, ToolResult


class ConfirmDecision(enum.Enum):
    """Outcome of a destructive-action confirmation."""

    ALLOW_ONCE = "once"
    ALLOW_SESSION = "always"
    DENY = "deny"


@dataclass(frozen=True)
class ConfirmRequest:
    """A request to confirm a destructive tool call."""

    tool_name: str
    summary: str


@runtime_checkable
class UISink(Protocol):
    """Everything the agent loop needs from a frontend."""

    def notice(self, message: str) -> None: ...
    def error(self, message: str) -> None: ...
    def assistant_message(self, markdown: str) -> None: ...
    def stream_token(self, token: str) -> None: ...
    def stream_end(self) -> None: ...
    def tool_call(self, tool_call: ToolCall) -> None: ...
    def tool_result(self, result: ToolResult) -> None: ...
    def response_stats(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        total_time: float,
        ttft: float | None,
    ) -> None: ...
    def busy(self, message: str) -> AbstractBusy: ...
    async def confirm(self, request: ConfirmRequest) -> ConfirmDecision: ...


class AbstractBusy(Protocol):
    """A context manager shown while the agent is working (spinner/loading)."""

    def __enter__(self) -> object: ...
    def __exit__(self, *exc: object) -> None: ...


class RichConsoleSink:
    """Default UISink: renders to a Rich console using the existing display functions."""

    def __init__(self, console: Console) -> None:
        self._console = console

    def notice(self, message: str) -> None:
        from mita.ui.display import display_warning

        display_warning(self._console, message)

    def error(self, message: str) -> None:
        from mita.ui.display import display_error

        display_error(self._console, message)

    def assistant_message(self, markdown: str) -> None:
        from mita.ui.display import display_markdown

        display_markdown(self._console, markdown)

    def stream_token(self, token: str) -> None:
        from mita.ui.display import display_streaming_token

        display_streaming_token(self._console, token)

    def stream_end(self) -> None:
        from mita.ui.display import display_streaming_end

        display_streaming_end(self._console)

    def tool_call(self, tool_call: ToolCall) -> None:
        from mita.ui.display import display_tool_call

        display_tool_call(self._console, tool_call)

    def tool_result(self, result: ToolResult) -> None:
        from mita.ui.display import display_tool_result

        display_tool_result(self._console, result)

    def response_stats(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        total_time: float,
        ttft: float | None,
    ) -> None:
        from mita.ui.display import display_response_stats

        display_response_stats(self._console, prompt_tokens, completion_tokens, total_time, ttft)

    def busy(self, message: str) -> AbstractBusy:
        from mita.ui.spinner import thinking_spinner

        return thinking_spinner(self._console)  # type: ignore[return-value]

    async def confirm(self, request: ConfirmRequest) -> ConfirmDecision:
        from mita.ui.display import prompt_user_confirm_decision

        decision = await prompt_user_confirm_decision(self._console, request.summary)
        return {
            "once": ConfirmDecision.ALLOW_ONCE,
            "always": ConfirmDecision.ALLOW_SESSION,
            "deny": ConfirmDecision.DENY,
        }.get(decision, ConfirmDecision.DENY)


@contextmanager
def _null_busy() -> Iterator[None]:
    yield


class RecordingSink:
    """A UISink that records events, for tests and headless runs."""

    def __init__(self) -> None:
        self.events: list[tuple[str, object]] = []
        self.confirm_decision = ConfirmDecision.DENY

    def notice(self, message: str) -> None:
        self.events.append(("notice", message))

    def error(self, message: str) -> None:
        self.events.append(("error", message))

    def assistant_message(self, markdown: str) -> None:
        self.events.append(("assistant_message", markdown))

    def stream_token(self, token: str) -> None:
        self.events.append(("stream_token", token))

    def stream_end(self) -> None:
        self.events.append(("stream_end", None))

    def tool_call(self, tool_call: ToolCall) -> None:
        self.events.append(("tool_call", tool_call))

    def tool_result(self, result: ToolResult) -> None:
        self.events.append(("tool_result", result))

    def response_stats(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        total_time: float,
        ttft: float | None,
    ) -> None:
        self.events.append(("response_stats", (prompt_tokens, completion_tokens)))

    def busy(self, message: str) -> AbstractBusy:
        return _null_busy()  # type: ignore[return-value]

    async def confirm(self, request: ConfirmRequest) -> ConfirmDecision:
        self.events.append(("confirm", request))
        return self.confirm_decision
