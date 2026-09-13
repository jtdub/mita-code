"""Tests for the LangChain model factory."""

from __future__ import annotations

import logging

import pytest
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from mita.config.schema import LLMProvider, MitaConfig
from mita.llm.factory import _map_ollama_options, build_chat_model, build_embedding_model


class TestBuildChatModel:
    def test_ollama_default(self) -> None:
        config = MitaConfig()
        model = build_chat_model(config)
        assert isinstance(model, ChatOllama)
        assert model.model == "qwen2.5-coder:7b"
        assert model.base_url == "http://localhost:11434"
        assert model.temperature == 0.1
        assert model.num_predict == 4096

    def test_ollama_uses_ollama_host(self) -> None:
        config = MitaConfig()
        config.ollama.host = "http://localhost:9999"
        model = build_chat_model(config)
        assert model.base_url == "http://localhost:9999"

    def test_openai_compatible(self) -> None:
        config = MitaConfig()
        config.llm.provider = LLMProvider.LLAMACPP
        model = build_chat_model(config)
        assert isinstance(model, ChatOpenAI)
        assert model.model_name == "qwen2.5-coder:7b"
        assert model.openai_api_base == "http://localhost:8080/v1"
        assert model.openai_api_key.get_secret_value() == "sk-no-key-required"
        assert model.stream_usage is True

    def test_explicit_api_key(self) -> None:
        config = MitaConfig()
        config.llm.provider = LLMProvider.LLAMACPP
        config.llm.api_key = "secret"
        model = build_chat_model(config)
        assert model.openai_api_key.get_secret_value() == "secret"

    def test_supported_ollama_options_mapped(self) -> None:
        config = MitaConfig()
        config.model.ollama_options.num_gpu = 2
        config.model.ollama_options.num_thread = 8
        config.model.ollama_options.num_ctx = 8192
        model = build_chat_model(config)
        assert model.num_gpu == 2
        assert model.num_thread == 8
        assert model.num_ctx == 8192

    def test_unsupported_ollama_options_dropped(self, caplog: pytest.LogCaptureFixture) -> None:
        config = MitaConfig()
        config.model.ollama_options.use_mmap = True
        config.model.ollama_options.low_vram = True
        with caplog.at_level(logging.WARNING):
            model = build_chat_model(config)
        assert isinstance(model, ChatOllama)
        assert "use_mmap" in caplog.text
        assert "low_vram" in caplog.text


class TestBuildEmbeddingModel:
    def test_ollama_default(self) -> None:
        config = MitaConfig()
        model = build_embedding_model(config)
        assert isinstance(model, OllamaEmbeddings)
        assert model.model == "nomic-embed-text"

    def test_openai_compatible(self) -> None:
        config = MitaConfig()
        config.llm.provider = LLMProvider.LLAMACPP
        model = build_embedding_model(config)
        assert isinstance(model, OpenAIEmbeddings)
        assert model.model == "nomic-embed-text"
        assert model.openai_api_base == "http://localhost:8080/v1"
        assert model.check_embedding_ctx_length is False


class TestMapOllamaOptions:
    def test_supported_only(self) -> None:
        config = MitaConfig()
        config.model.ollama_options.num_gpu = 1
        assert _map_ollama_options(config) == {"num_gpu": 1}

    def test_unsupported_excluded(self) -> None:
        config = MitaConfig()
        config.model.ollama_options.num_gpu = 1
        config.model.ollama_options.use_mmap = True
        result = _map_ollama_options(config)
        assert result == {"num_gpu": 1}
