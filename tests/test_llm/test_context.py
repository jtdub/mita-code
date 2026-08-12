"""Tests for backend context-window probing (audit finding C5)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from mita.config.schema import LLMProvider, MitaConfig
from mita.llm import context


@pytest.fixture(autouse=True)
def _clear_probe_cache() -> None:
    context.clear_cache()


class TestResolveContextWindow:
    @pytest.mark.asyncio()
    async def test_off_uses_config_value(self) -> None:
        config = MitaConfig()
        config.llm.context_probe = "off"
        config.model.context_window = 12345
        assert await context.resolve_context_window(config) == 12345

    @pytest.mark.asyncio()
    async def test_explicit_int(self) -> None:
        config = MitaConfig()
        config.llm.context_probe = "8192"
        assert await context.resolve_context_window(config) == 8192

    @pytest.mark.asyncio()
    async def test_auto_uses_probe(self) -> None:
        config = MitaConfig()
        config.llm.provider = LLMProvider.VLLM
        config.llm.context_probe = "auto"
        with patch("mita.llm.context._probe", new_callable=AsyncMock, return_value=16384):
            assert await context.resolve_context_window(config) == 16384

    @pytest.mark.asyncio()
    async def test_auto_falls_back_when_probe_fails(self) -> None:
        config = MitaConfig()
        config.llm.provider = LLMProvider.VLLM
        config.llm.context_probe = "auto"
        config.model.context_window = 4096
        with patch("mita.llm.context._probe", new_callable=AsyncMock, return_value=None):
            assert await context.resolve_context_window(config) == 4096

    @pytest.mark.asyncio()
    async def test_auto_result_is_cached(self) -> None:
        config = MitaConfig()
        config.llm.provider = LLMProvider.VLLM
        config.llm.context_probe = "auto"
        with patch(
            "mita.llm.context._probe", new_callable=AsyncMock, return_value=16384
        ) as mock_probe:
            await context.resolve_context_window(config)
            await context.resolve_context_window(config)
            mock_probe.assert_awaited_once()


class TestEnvExpansion:
    def test_api_key_expands_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MITA_TEST_KEY", "secret-123")
        config = MitaConfig.model_validate({"llm": {"api_key": "${MITA_TEST_KEY}"}})
        assert config.llm.api_key == "secret-123"

    def test_no_llm_section_defaults_to_ollama(self) -> None:
        config = MitaConfig()
        assert config.llm.provider == LLMProvider.OLLAMA
        assert config.llm.base_url == ""
