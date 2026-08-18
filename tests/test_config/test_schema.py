"""Tests for config schema models."""

import pytest
from pydantic import ValidationError

from mita.config.schema import (
    PERMISSION_MODE_TOOLS,
    IndexSettings,
    MitaConfig,
    ModelSettings,
    OllamaSettings,
    PermissionMode,
    PluginDefinition,
    SessionSettings,
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


class TestPermissionMode:
    def test_default_mode_is_ask(self) -> None:
        settings = ToolSettings()
        assert settings.permission_mode == PermissionMode.ASK

    def test_effective_auto_approve_ask(self) -> None:
        settings = ToolSettings(permission_mode=PermissionMode.ASK)
        effective = settings.effective_auto_approve
        assert effective == {"file_read", "glob", "grep"}

    def test_effective_auto_approve_auto_edit(self) -> None:
        settings = ToolSettings(permission_mode=PermissionMode.AUTO_EDIT)
        effective = settings.effective_auto_approve
        assert "file_write" in effective
        assert "file_edit" in effective
        assert "git" in effective
        assert "shell" not in effective

    def test_effective_auto_approve_trust(self) -> None:
        settings = ToolSettings(permission_mode=PermissionMode.TRUST)
        effective = settings.effective_auto_approve
        assert "shell" in effective
        assert "file_write" in effective
        assert "git" in effective

    def test_effective_merges_config_overrides(self) -> None:
        """Explicit auto_approve entries are merged with mode defaults."""
        settings = ToolSettings(
            permission_mode=PermissionMode.ASK,
            auto_approve=["file_read", "glob", "grep", "custom_tool"],
        )
        effective = settings.effective_auto_approve
        assert "custom_tool" in effective
        assert "file_read" in effective

    def test_auto_approve_default_matches_ask_mode(self) -> None:
        """Default auto_approve list derives from ASK mode — no duplication drift."""
        settings = ToolSettings()
        assert set(settings.auto_approve) == set(PERMISSION_MODE_TOOLS[PermissionMode.ASK])

    def test_permission_mode_from_string(self) -> None:
        settings = ToolSettings(permission_mode="auto_edit")  # type: ignore[arg-type]
        assert settings.permission_mode == PermissionMode.AUTO_EDIT

    def test_each_mode_is_superset_of_previous(self) -> None:
        ask = set(PERMISSION_MODE_TOOLS[PermissionMode.ASK])
        auto_edit = set(PERMISSION_MODE_TOOLS[PermissionMode.AUTO_EDIT])
        trust = set(PERMISSION_MODE_TOOLS[PermissionMode.TRUST])
        assert ask.issubset(auto_edit)
        assert auto_edit.issubset(trust)


class TestSessionSettings:
    def test_defaults(self) -> None:
        cfg = MitaConfig()
        assert cfg.sessions.autosave is True
        assert cfg.sessions.max_age_days == 7

    def test_negative_max_age_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SessionSettings(max_age_days=-1)

    def test_zero_max_age_allowed(self) -> None:
        assert SessionSettings(max_age_days=0).max_age_days == 0
