"""Tests for the backend provider registry (audit finding C5)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from mita.config.schema import LLMProvider, MitaConfig
from mita.llm.client import LLMClient
from mita.llm.providers import resolve_backend


class TestResolveBackend:
    def test_ollama_default_uses_ollama_host(self) -> None:
        config = MitaConfig()
        config.ollama.host = "http://localhost:9999"
        backend = resolve_backend(config)
        assert backend.model == "ollama_chat/qwen2.5-coder:7b"
        assert backend.api_base == "http://localhost:9999"
        assert backend.api_key is None
        assert backend.is_ollama is True

    def test_vllm_prefix_and_default_base(self) -> None:
        config = MitaConfig()
        config.llm.provider = LLMProvider.VLLM
        backend = resolve_backend(config)
        assert backend.model.startswith("hosted_vllm/")
        assert backend.api_base == "http://localhost:8000/v1"
        assert backend.api_key is None  # vllm doesn't need a placeholder

    def test_openai_routed_gets_placeholder_key(self) -> None:
        config = MitaConfig()
        config.llm.provider = LLMProvider.LLAMACPP
        backend = resolve_backend(config)
        assert backend.model.startswith("openai/")
        assert backend.api_key == "sk-no-key-required"

    def test_lmstudio_prefix(self) -> None:
        config = MitaConfig()
        config.llm.provider = LLMProvider.LMSTUDIO
        backend = resolve_backend(config)
        assert backend.model.startswith("lm_studio/")

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


class TestClientProviderBehavior:
    @pytest.mark.asyncio()
    async def test_non_ollama_omits_options_blob(self) -> None:
        config = MitaConfig()
        config.llm.provider = LLMProvider.LLAMACPP  # openai/-routed
        config.model.ollama_options.num_gpu = 1  # would be sent on ollama, not here
        client = LLMClient(config)

        with patch("mita.llm.client.litellm.acompletion", new_callable=AsyncMock) as mock_comp:
            mock_comp.return_value = {}
            await client.chat([{"role": "user", "content": "hi"}])
            kwargs = mock_comp.call_args.kwargs
            assert "extra_body" not in kwargs
            assert kwargs["api_key"] == "sk-no-key-required"

    @pytest.mark.asyncio()
    async def test_ollama_still_sends_options(self) -> None:
        config = MitaConfig()
        config.model.ollama_options.num_gpu = 1
        client = LLMClient(config)

        with patch("mita.llm.client.litellm.acompletion", new_callable=AsyncMock) as mock_comp:
            mock_comp.return_value = {}
            await client.chat([{"role": "user", "content": "hi"}])
            kwargs = mock_comp.call_args.kwargs
            assert kwargs["extra_body"] == {"options": {"num_gpu": 1}}
            assert "api_key" not in kwargs
