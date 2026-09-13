"""LangChain model factory — builds chat and embedding models per provider."""

from __future__ import annotations

import logging
import os
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import SecretStr

from mita.config.schema import MitaConfig
from mita.llm.providers import API_KEY_PLACEHOLDER, resolve_backend

_logger = logging.getLogger(__name__)

_SUPPORTED_OLLAMA_OPTIONS = ("num_ctx", "num_gpu", "num_thread")
"""The Ollama runtime options ChatOllama accepts as constructor keywords."""


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

    A ``base_url`` turns off the automatic stream-usage request in ChatOpenAI, so
    the model asks for it directly and token counts still stream.
    """
    backend = resolve_backend(config)
    if backend.is_ollama:
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
    return ChatOpenAI(
        model=config.model.default,
        base_url=backend.api_base,
        api_key=SecretStr(backend.api_key or API_KEY_PLACEHOLDER),
        temperature=config.model.temperature,
        max_completion_tokens=config.model.max_tokens,
        stream_usage=True,
    )


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
    """Return the Ollama runtime options ChatOllama supports; warn on the rest.

    The unsupported set is derived from the config model, so a new option is
    either forwarded or reported, never dropped in silence.
    """
    raw: dict[str, Any] = config.model.ollama_options.to_api_dict()
    dropped = sorted(set(raw) - set(_SUPPORTED_OLLAMA_OPTIONS))
    if dropped:
        _logger.warning(
            "Ollama options not supported by ChatOllama, ignored: %s", ", ".join(dropped)
        )
    return {
        key: raw[key]
        for key in _SUPPORTED_OLLAMA_OPTIONS
        if isinstance(raw.get(key), int) and not isinstance(raw.get(key), bool)
    }
