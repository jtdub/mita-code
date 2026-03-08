"""LiteLLM client pointing at Ollama for chat completions."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import litellm

from mita.config.schema import MitaConfig


class LLMClient:
    """Thin wrapper around LiteLLM configured for Ollama."""

    def __init__(self, config: MitaConfig) -> None:
        self._model = f"ollama/{config.model.default}"
        self._api_base = config.ollama.host
        self._temperature = config.model.temperature
        self._max_tokens = config.model.max_tokens
        self._stream = config.ui.stream

        # Suppress LiteLLM's verbose logging
        litellm.suppress_debug_info = True

    @property
    def model(self) -> str:
        """The model identifier being used."""
        return self._model

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Send a chat completion request (non-streaming).

        Returns the full response dict from LiteLLM.
        """
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "api_base": self._api_base,
        }
        if tools:
            kwargs["tools"] = tools

        response = await litellm.acompletion(**kwargs)
        return response  # type: ignore[no-any-return]

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Send a streaming chat completion request.

        Yields delta dicts with content tokens as they arrive.
        """
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "api_base": self._api_base,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools

        response = await litellm.acompletion(**kwargs)
        async for chunk in response:
            yield chunk


def get_client(config: MitaConfig) -> LLMClient:
    """Create an LLM client from configuration."""
    return LLMClient(config)
