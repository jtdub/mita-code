"""LangChain model factory — builds chat and embedding models per provider."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import SecretStr

from mita.config.schema import MitaConfig
from mita.llm.providers import API_KEY_PLACEHOLDER, resolve_backend

_logger = logging.getLogger(__name__)

# Ollama runtime options ChatOllama cannot express. Dropped with a warning.
_UNSUPPORTED_OLLAMA_OPTIONS = frozenset(
    {"num_batch", "use_mmap", "use_mlock", "num_keep", "main_gpu", "low_vram", "flash_attention"}
)


def build_chat_model(config: MitaConfig) -> BaseChatModel:
    """Build the chat model for the configured provider."""
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
    )


def build_embedding_model(config: MitaConfig) -> Embeddings:
    """Build the embedding model for the configured provider."""
    backend = resolve_backend(config)
    if backend.is_ollama:
        return OllamaEmbeddings(model=config.model.embedding, base_url=backend.api_base)
    return OpenAIEmbeddings(
        model=config.model.embedding,
        base_url=backend.api_base,
        api_key=SecretStr(backend.api_key or API_KEY_PLACEHOLDER),
    )


def _map_ollama_options(config: MitaConfig) -> dict[str, int]:
    """Return the Ollama runtime options ChatOllama supports; warn on the rest."""
    raw: dict[str, Any] = config.model.ollama_options.to_api_dict()
    dropped = sorted(set(raw) & _UNSUPPORTED_OLLAMA_OPTIONS)
    if dropped:
        _logger.warning(
            "Ollama options not supported by ChatOllama, ignored: %s", ", ".join(dropped)
        )
    supported: dict[str, int] = {}
    for key in ("num_ctx", "num_gpu", "num_thread"):
        value = raw.get(key)
        if isinstance(value, int):
            supported[key] = value
    return supported
