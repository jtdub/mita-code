"""Backend provider registry — maps a provider to its LiteLLM routing (finding C5).

One place that knows each backend's LiteLLM prefix, default base URL, whether it has a
model registry (pull/list), and whether the OpenAI client requires a placeholder api_key.
"""

from __future__ import annotations

from dataclasses import dataclass

from mita.config.schema import LLMProvider, MitaConfig

# LiteLLM requires a non-empty api_key for its OpenAI-compatible client, even when the
# local server needs no auth. Substitute this when the user left api_key empty.
_API_KEY_PLACEHOLDER = "sk-no-key-required"


@dataclass(frozen=True)
class ProviderSpec:
    """Static routing facts for one backend."""

    prefix: str  # LiteLLM model prefix, e.g. "ollama_chat/", "hosted_vllm/", "openai/"
    default_base_url: str
    has_model_registry: bool  # can pull/list models (Ollama only)
    needs_api_key_placeholder: bool  # openai/-routed clients require a non-empty key
    context_probe: str  # "ollama_show" | "openai_models" | "llamacpp_props" | "none"

    @property
    def is_ollama(self) -> bool:
        return self.prefix == "ollama_chat/"


PROVIDERS: dict[LLMProvider, ProviderSpec] = {
    LLMProvider.OLLAMA: ProviderSpec(
        "ollama_chat/", "http://localhost:11434", True, False, "ollama_show"
    ),
    LLMProvider.LLAMACPP: ProviderSpec(
        "openai/", "http://localhost:8080/v1", False, True, "llamacpp_props"
    ),
    LLMProvider.VLLM: ProviderSpec(
        "hosted_vllm/", "http://localhost:8000/v1", False, False, "openai_models"
    ),
    LLMProvider.LMSTUDIO: ProviderSpec(
        "lm_studio/", "http://localhost:1234/v1", False, False, "none"
    ),
    LLMProvider.TGI: ProviderSpec("openai/", "http://localhost:8080/v1", False, True, "none"),
    LLMProvider.OPENAI_COMPATIBLE: ProviderSpec("openai/", "", False, True, "none"),
}


@dataclass(frozen=True)
class ResolvedBackend:
    """A concrete, ready-to-call backend derived from config."""

    provider: LLMProvider
    spec: ProviderSpec
    model: str  # full LiteLLM model string, e.g. "ollama_chat/qwen2.5-coder:7b"
    api_base: str
    api_key: str | None

    @property
    def is_ollama(self) -> bool:
        return self.spec.is_ollama


def resolve_backend(config: MitaConfig) -> ResolvedBackend:
    """Resolve the configured provider into concrete LiteLLM call parameters."""
    spec = PROVIDERS[config.llm.provider]

    base = config.llm.base_url.strip()
    if not base:
        # Ollama honors the existing [ollama] host for backward compatibility.
        base = config.ollama.host if spec.is_ollama else spec.default_base_url

    api_key: str | None = config.llm.api_key.strip() or None
    if api_key is None and spec.needs_api_key_placeholder:
        api_key = _API_KEY_PLACEHOLDER

    return ResolvedBackend(
        provider=config.llm.provider,
        spec=spec,
        model=f"{spec.prefix}{config.model.default}",
        api_base=base,
        api_key=api_key,
    )
