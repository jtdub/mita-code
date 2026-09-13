"""Backend provider registry — maps a provider to its LangChain routing.

One place that knows each backend's default base URL, whether it has a model registry
(pull/list), whether the OpenAI client requires a placeholder api_key, and its
context-probe strategy.
"""

from __future__ import annotations

from dataclasses import dataclass

from mita.config.schema import LLMProvider, MitaConfig

API_KEY_PLACEHOLDER = "sk-no-key-required"
"""Stand-in key for a local server that needs no auth.

OpenAI clients reject an empty api_key, so this is substituted when the user
left api_key empty.
"""


@dataclass(frozen=True)
class ProviderSpec:
    """Static routing facts for one backend."""

    is_ollama: bool
    """True when the backend uses the native Ollama API and daemon management."""

    default_base_url: str
    """Base URL used when the user set no `[llm] base_url`."""

    has_model_registry: bool
    """True when the backend can pull and list models. Ollama only."""

    needs_api_key_placeholder: bool
    """True when the OpenAI-routed client requires a non-empty key."""

    context_probe: str
    """One of "ollama_show", "openai_models", "llamacpp_props", or "none"."""


PROVIDERS: dict[LLMProvider, ProviderSpec] = {
    LLMProvider.OLLAMA: ProviderSpec(
        is_ollama=True,
        default_base_url="http://localhost:11434",
        has_model_registry=True,
        needs_api_key_placeholder=False,
        context_probe="ollama_show",
    ),
    LLMProvider.LLAMACPP: ProviderSpec(
        is_ollama=False,
        default_base_url="http://localhost:8080/v1",
        has_model_registry=False,
        needs_api_key_placeholder=True,
        context_probe="llamacpp_props",
    ),
    LLMProvider.VLLM: ProviderSpec(
        is_ollama=False,
        default_base_url="http://localhost:8000/v1",
        has_model_registry=False,
        needs_api_key_placeholder=False,
        context_probe="openai_models",
    ),
    LLMProvider.LMSTUDIO: ProviderSpec(
        is_ollama=False,
        default_base_url="http://localhost:1234/v1",
        has_model_registry=False,
        needs_api_key_placeholder=False,
        context_probe="none",
    ),
    LLMProvider.TGI: ProviderSpec(
        is_ollama=False,
        default_base_url="http://localhost:8080/v1",
        has_model_registry=False,
        needs_api_key_placeholder=True,
        context_probe="none",
    ),
    LLMProvider.OPENAI_COMPATIBLE: ProviderSpec(
        is_ollama=False,
        default_base_url="",
        has_model_registry=False,
        needs_api_key_placeholder=True,
        context_probe="none",
    ),
}


@dataclass(frozen=True)
class ResolvedBackend:
    """A concrete, ready-to-call backend derived from config."""

    provider: LLMProvider
    spec: ProviderSpec
    api_base: str
    api_key: str | None

    @property
    def is_ollama(self) -> bool:
        return self.spec.is_ollama


def resolve_backend(config: MitaConfig) -> ResolvedBackend:
    """Resolve the configured provider into concrete call parameters."""
    spec = PROVIDERS[config.llm.provider]

    base = config.llm.base_url.strip()
    if not base:
        # Ollama honors the existing [ollama] host for backward compatibility.
        base = config.ollama.host if spec.is_ollama else spec.default_base_url

    api_key: str | None = config.llm.api_key.strip() or None
    if api_key is None and spec.needs_api_key_placeholder:
        api_key = API_KEY_PLACEHOLDER

    return ResolvedBackend(
        provider=config.llm.provider,
        spec=spec,
        api_base=base,
        api_key=api_key,
    )
