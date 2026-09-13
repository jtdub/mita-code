"""Tests for the backend provider registry (audit finding C5)."""

from __future__ import annotations

from mita.config.schema import LLMProvider, MitaConfig
from mita.llm.providers import resolve_backend


class TestResolveBackend:
    def test_ollama_default_uses_ollama_host(self) -> None:
        config = MitaConfig()
        config.ollama.host = "http://localhost:9999"
        backend = resolve_backend(config)
        assert backend.api_base == "http://localhost:9999"
        assert backend.api_key is None
        assert backend.is_ollama is True

    def test_vllm_default_base(self) -> None:
        config = MitaConfig()
        config.llm.provider = LLMProvider.VLLM
        backend = resolve_backend(config)
        assert backend.api_base == "http://localhost:8000/v1"
        assert backend.api_key is None  # vllm doesn't need a placeholder
        assert backend.is_ollama is False

    def test_openai_routed_gets_placeholder_key(self) -> None:
        config = MitaConfig()
        config.llm.provider = LLMProvider.LLAMACPP
        backend = resolve_backend(config)
        assert backend.api_key == "sk-no-key-required"

    def test_lmstudio_default_base(self) -> None:
        config = MitaConfig()
        config.llm.provider = LLMProvider.LMSTUDIO
        backend = resolve_backend(config)
        assert backend.api_base == "http://localhost:1234/v1"

    def test_explicit_base_url_wins(self) -> None:
        config = MitaConfig()
        config.llm.provider = LLMProvider.VLLM
        config.llm.base_url = "http://gpu-box:8000/v1"
        backend = resolve_backend(config)
        assert backend.api_base == "http://gpu-box:8000/v1"

    def test_explicit_api_key_used(self) -> None:
        config = MitaConfig()
        config.llm.provider = LLMProvider.VLLM
        config.llm.api_key = "secret-token"
        backend = resolve_backend(config)
        assert backend.api_key == "secret-token"
