"""LangChain model factory — builds chat and embedding models per provider."""

from __future__ import annotations

import os
from typing import Any, cast

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import SecretStr

from mita.config.schema import MitaConfig
from mita.llm.providers import API_KEY_PLACEHOLDER, ResolvedBackend, resolve_backend

_SUPPORTED_OLLAMA_OPTIONS = ("num_ctx", "num_gpu", "num_thread")
"""The Ollama runtime options ChatOllama accepts as constructor keywords."""

_UNSUPPORTED_OLLAMA_OPTIONS = frozenset(
    {"num_batch", "use_mmap", "use_mlock", "num_keep", "main_gpu", "low_vram", "flash_attention"}
)
"""Options ChatOllama can only send through the ``options`` request blob."""


def _disable_langsmith_telemetry() -> None:
    """Turn LangSmith tracing off unless the user opted in explicitly.

    LangChain uploads prompts and completions when these variables are set in
    the ambient environment. Mita promises no telemetry egress.
    """
    os.environ.setdefault("LANGSMITH_TRACING", "false")
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")


_disable_langsmith_telemetry()


def build_chat_model(config: MitaConfig) -> BaseChatModel:
    """Build the chat model for the configured provider.

    ``stream_usage`` follows ``[ui] show_token_count``: it asks the backend for
    streamed token counts, and a server that rejects ``stream_options`` works
    when the flag is off.
    """
    backend = resolve_backend(config)
    if backend.is_ollama:
        return _build_ollama_model(config, backend)
    return ChatOpenAI(
        model=config.model.default,
        base_url=backend.api_base,
        api_key=SecretStr(backend.api_key or API_KEY_PLACEHOLDER),
        temperature=config.model.temperature,
        # ChatOpenAI rewrites `max_tokens` to `max_completion_tokens`, which a
        # local OpenAI-compatible server ignores. extra_body carries the cap
        # verbatim as `max_tokens` instead.
        extra_body={"max_tokens": config.model.max_tokens},
        stream_usage=config.ui.show_token_count,
    )


def _build_ollama_model(config: MitaConfig, backend: ResolvedBackend) -> BaseChatModel:
    """Build a ChatOllama, forwarding every configured runtime option.

    langchain-ollama sends the supported subset as constructor keywords and a
    bound ``options`` blob verbatim. The blob replaces the keyword-derived
    options, so when any unsupported option is set, the whole set goes into it.
    """
    runtime: dict[str, Any] = config.model.ollama_options.to_api_dict()
    if not (set(runtime) & _UNSUPPORTED_OLLAMA_OPTIONS):
        options = _map_ollama_options(config)
        return ChatOllama(
            model=config.model.default,
            base_url=backend.api_base,
            temperature=config.model.temperature,
            num_predict=config.model.max_tokens,
            num_ctx=options.get("num_ctx"),
            num_gpu=options.get("num_gpu"),
            num_thread=options.get("num_thread"),
        )
    full_options = {
        **runtime,
        "temperature": config.model.temperature,
        "num_predict": config.model.max_tokens,
    }
    model = ChatOllama(model=config.model.default, base_url=backend.api_base)
    return cast(BaseChatModel, model.bind(options=full_options))


def build_embedding_model(config: MitaConfig) -> Embeddings:
    """Build the embedding model for the configured provider.

    Off OpenAI itself, tiktoken length checks post token IDs that a local server
    cannot read, so ``check_embedding_ctx_length`` stays off.
    """
    backend = resolve_backend(config)
    if backend.is_ollama:
        return OllamaEmbeddings(model=config.model.embedding, base_url=backend.api_base)
    return OpenAIEmbeddings(
        model=config.model.embedding,
        base_url=backend.api_base,
        api_key=SecretStr(backend.api_key or API_KEY_PLACEHOLDER),
        check_embedding_ctx_length=False,
    )


def _map_ollama_options(config: MitaConfig) -> dict[str, int]:
    """Return the Ollama runtime options ChatOllama accepts as keywords."""
    raw: dict[str, Any] = config.model.ollama_options.to_api_dict()
    return {
        key: raw[key]
        for key in _SUPPORTED_OLLAMA_OPTIONS
        if isinstance(raw.get(key), int) and not isinstance(raw.get(key), bool)
    }
