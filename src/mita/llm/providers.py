"""Backend provider registry — maps a provider to its LangChain routing.

One place that knows each backend's default base URL, whether it has a model registry
(pull/list), whether the OpenAI client requires a placeholder api_key, and its
context-probe strategy.
"""

from __future__ import annotations

from dataclasses import dataclass

from mita.config.schema import LLMProvider, MitaConfig

# OpenAI clients require a non-empty api_key, even when the local server needs no
# auth. Substitute this when the user left api_key empty.
API_KEY_PLACEHOLDER = "sk-no-key-required"


@dataclass(frozen=True)
class ProviderSpec:
    """Static routing facts for one backend."""

    is_ollama: bool  # uses the native Ollama API and daemon management
    default_base_url: str
    has_model_registry: bool  # can pull/list models (Ollama only)
    needs_api_key_placeholder: bool  # openai-routed clients require a non-empty key
    context_probe: str  # "ollama_show" | "openai_models" | "llamacpp_props" | "none"


PROVIDERS: dict[LLMProvider, ProviderSpec] = {
    LLMProvider.OLLAMA: ProviderSpec(True, "http://localhost:11434", True, False, "ollama_show"),
    LLMProvider.LLAMACPP: ProviderSpec(
        False, "http://localhost:8080/v1", False, True, "llamacpp_props"
    ),
    LLMProvider.VLLM: ProviderSpec(
        False, "http://localhost:8000/v1", False, False, "openai_models"
    ),
    LLMProvider.LMSTUDIO: ProviderSpec(False, "http://localhost:1234/v1", False, False, "none"),
    LLMProvider.TGI: ProviderSpec(False, "http://localhost:8080/v1", False, True, "none"),
    LLMProvider.OPENAI_COMPATIBLE: ProviderSpec(False, "", False, True, "none"),
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
