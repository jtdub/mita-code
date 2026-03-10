"""Token-by-token streaming handler for Rich terminal display."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from mita.llm.client import LLMClient

_logger = logging.getLogger(__name__)


async def stream_to_terminal(
    client: LLMClient,
    messages: list[dict[str, Any]],
    on_token: Callable[[str], None] | None = None,
    on_complete: Callable[[str], None] | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> str:
    """Stream a chat completion, calling on_token for each chunk.

    Args:
        client: The LLM client to use.
        messages: Chat messages.
        on_token: Callback called with each text token as it arrives.
        on_complete: Callback called with the full text when streaming is done.
        tools: Optional tool schemas for the LLM.

    Returns:
        The complete response text.
    """
    full_text = ""

    try:
        async for chunk in client.stream_chat(messages, tools=tools):
            delta = _extract_delta_content(chunk)
            if delta:
                full_text += delta
                if on_token:
                    try:
                        on_token(delta)
                    except Exception:
                        _logger.warning("on_token callback failed", exc_info=True)
    finally:
        if on_complete:
            try:
                on_complete(full_text)
            except Exception:
                _logger.warning("on_complete callback failed", exc_info=True)

    return full_text


def _extract_delta_content(chunk: Any) -> str:
    """Extract text content from a streaming chunk."""
    try:
        choices = chunk.choices if hasattr(chunk, "choices") else chunk.get("choices", [])
        if not choices:
            return ""
        delta = choices[0].delta if hasattr(choices[0], "delta") else choices[0].get("delta", {})
        if hasattr(delta, "content"):
            return delta.content or ""
        if isinstance(delta, dict):
            return delta.get("content", "") or ""
    except (IndexError, AttributeError, KeyError):
        pass
    return ""
