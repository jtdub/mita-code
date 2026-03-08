"""Tests for the LLM client."""

from __future__ import annotations

from mita.config.schema import MitaConfig
from mita.llm.client import LLMClient, get_client


class TestLLMClient:
    def test_model_name(self) -> None:
        config = MitaConfig()
        client = LLMClient(config)
        assert client.model == "ollama/qwen2.5-coder:7b"

    def test_custom_model(self) -> None:
        config = MitaConfig()
        config.model.default = "codellama:7b"
        client = LLMClient(config)
        assert client.model == "ollama/codellama:7b"

    def test_get_client(self) -> None:
        config = MitaConfig()
        client = get_client(config)
        assert isinstance(client, LLMClient)
