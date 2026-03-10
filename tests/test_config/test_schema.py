"""Tests for config schema models."""

import pytest
from pydantic import ValidationError

from mita.config.schema import (
    IndexSettings,
    MitaConfig,
    ModelSettings,
    OllamaSettings,
    PluginDefinition,
    ToolSettings,
)


class TestMitaConfig:
    def test_default_config(self) -> None:
        cfg = MitaConfig()
        assert cfg.ollama.host == "http://localhost:11434"
        assert cfg.model.default == "qwen2.5-coder:7b"
        assert cfg.tools.confirm_destructive is True
        assert cfg.memory.max_lines_per_file == 200
        assert cfg.index.enabled is True
        assert cfg.ui.stream is True
        assert cfg.hooks == []
        assert cfg.plugins == []

    def test_custom_values(self) -> None:
        cfg = MitaConfig(
            ollama=OllamaSettings(host="http://localhost:9999"),
            model=ModelSettings(default="codellama:13b", temperature=0.5),
        )
        assert cfg.ollama.host == "http://localhost:9999"
        assert cfg.model.default == "codellama:13b"
        assert cfg.model.temperature == 0.5
        # Other fields keep defaults
        assert cfg.tools.shell_timeout == 120

    def test_model_validate_from_dict(self) -> None:
        data = {
            "ollama": {"host": "http://custom:11434"},
            "model": {"default": "deepseek-coder:6.7b"},
        }
        cfg = MitaConfig.model_validate(data)
        assert cfg.ollama.host == "http://custom:11434"
        assert cfg.model.default == "deepseek-coder:6.7b"
        # Unspecified fields use defaults
        assert cfg.ollama.timeout == 120


class TestValidators:
    def test_ollama_timeout_positive(self) -> None:
        with pytest.raises(ValidationError, match="timeout must be positive"):
            OllamaSettings(timeout=0)

    def test_max_tokens_positive(self) -> None:
        with pytest.raises(ValidationError, match="max_tokens must be positive"):
            ModelSettings(max_tokens=0)

    def test_context_window_positive(self) -> None:
        with pytest.raises(ValidationError, match="context_window must be positive"):
            ModelSettings(context_window=0)

    def test_context_window_gte_max_tokens(self) -> None:
        with pytest.raises(ValidationError, match="context_window.*must be >= max_tokens"):
            ModelSettings(max_tokens=8192, context_window=4096)

    def test_context_window_equals_max_tokens_ok(self) -> None:
        m = ModelSettings(max_tokens=4096, context_window=4096)
        assert m.context_window == m.max_tokens

    def test_shell_timeout_positive(self) -> None:
        with pytest.raises(ValidationError, match="shell_timeout must be positive"):
            ToolSettings(shell_timeout=-1)

    def test_chunk_size_gt_overlap(self) -> None:
        with pytest.raises(ValidationError, match="chunk_size.*must be > chunk_overlap"):
            IndexSettings(chunk_size=64, chunk_overlap=64)

    def test_top_k_positive(self) -> None:
        with pytest.raises(ValidationError, match="top_k must be positive"):
            IndexSettings(top_k=0)

    def test_plugin_url_validation(self) -> None:
        with pytest.raises(ValidationError, match="must start with http"):
            PluginDefinition(name="bad", transport="sse", url="not a url")

    def test_plugin_url_valid(self) -> None:
        p = PluginDefinition(name="good", transport="sse", url="http://localhost:3001/sse")
        assert p.url == "http://localhost:3001/sse"

    def test_plugin_url_none_ok(self) -> None:
        p = PluginDefinition(name="stdio", transport="stdio", command="echo")
        assert p.url is None

    def test_max_iterations_positive(self) -> None:
        with pytest.raises(ValidationError, match="max_iterations must be positive"):
            MitaConfig(max_iterations=0)
