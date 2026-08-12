"""LiteLLM client — routes to the configured backend provider (finding C5)."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

import litellm

from mita.config.schema import MitaConfig
from mita.llm.providers import resolve_backend

_logger = logging.getLogger(__name__)


class LLMClient:
    """Thin wrapper around LiteLLM, configured for any supported backend."""

    def __init__(self, config: MitaConfig) -> None:
        backend = resolve_backend(config)
        self._model = backend.model
        self._api_base = backend.api_base
        self._api_key = backend.api_key
        self._is_ollama = backend.is_ollama
        self._temperature = config.model.temperature
        self._max_tokens = config.model.max_tokens
        self._stream = config.ui.stream
        # Ollama runtime options are provider-specific; only send them to Ollama.
        self._ollama_options = config.model.ollama_options.to_api_dict() if self._is_ollama else {}

        # Suppress LiteLLM's verbose logging, and drop OpenAI params a backend doesn't
        # support (e.g. stream_options) rather than erroring.
        litellm.suppress_debug_info = True
        litellm.drop_params = True

    @property
    def model(self) -> str:
        """The model identifier being used."""
        return self._model

    def _base_kwargs(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "api_base": self._api_base,
        }
        if self._api_key is not None:
            kwargs["api_key"] = self._api_key
        if tools:
            kwargs["tools"] = tools
        # The Ollama `options` blob goes via extra_body, which LiteLLM forwards
        # untouched — so it must only be attached for the Ollama provider.
        if self._ollama_options:
            kwargs["extra_body"] = {"options": self._ollama_options}
        return kwargs

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Send a chat completion request (non-streaming).

        Returns the full response dict from LiteLLM.
        """
        response = await litellm.acompletion(**self._base_kwargs(messages, tools))
        return response  # type: ignore[no-any-return]

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Send a streaming chat completion request.

        Yields delta dicts with content tokens as they arrive.
        """
        kwargs = self._base_kwargs(messages, tools)
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}

        response = await litellm.acompletion(**kwargs)
        async for chunk in response:
            yield chunk


def get_client(config: MitaConfig) -> LLMClient:
    """Create an LLM client from configuration."""
    return LLMClient(config)
