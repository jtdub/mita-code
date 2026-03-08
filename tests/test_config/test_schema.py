"""Tests for config schema models."""

from mita.config.schema import MitaConfig, ModelSettings, OllamaSettings


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
