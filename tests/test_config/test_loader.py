"""Tests for config loading and merging."""

from pathlib import Path

from mita.config.loader import _deep_merge, load_config


class TestDeepMerge:
    def test_simple_override(self) -> None:
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_dict_merge(self) -> None:
        base = {"section": {"a": 1, "b": 2}}
        override = {"section": {"b": 3, "c": 4}}
        result = _deep_merge(base, override)
        assert result == {"section": {"a": 1, "b": 3, "c": 4}}

    def test_list_append(self) -> None:
        base = {"items": [1, 2]}
        override = {"items": [3, 4]}
        result = _deep_merge(base, override)
        assert result == {"items": [1, 2, 3, 4]}

    def test_empty_base(self) -> None:
        result = _deep_merge({}, {"a": 1})
        assert result == {"a": 1}

    def test_empty_override(self) -> None:
        result = _deep_merge({"a": 1}, {})
        assert result == {"a": 1}

    def test_deeply_nested(self) -> None:
        base = {"a": {"b": {"c": 1, "d": 2}}}
        override = {"a": {"b": {"d": 3, "e": 4}}}
        result = _deep_merge(base, override)
        assert result == {"a": {"b": {"c": 1, "d": 3, "e": 4}}}


class TestLoadConfig:
    def test_load_defaults_no_files(self) -> None:
        """With no config files, should return all defaults."""
        cfg = load_config()
        assert cfg.ollama.host == "http://localhost:11434"
        assert cfg.model.default == "qwen2.5-coder:7b"

    def test_load_global_config(self, tmp_path: Path) -> None:
        """Global config file should be loaded."""
        global_dir = Path.home() / ".config" / "mita"
        global_dir.mkdir(parents=True)
        global_config = global_dir / "config.toml"
        global_config.write_text('[model]\ndefault = "codellama:13b"\ntemperature = 0.7\n')

        cfg = load_config()
        assert cfg.model.default == "codellama:13b"
        assert cfg.model.temperature == 0.7
        # Non-overridden values keep defaults
        assert cfg.ollama.host == "http://localhost:11434"

    def test_load_project_overrides_global(self, tmp_project: Path) -> None:
        """Project config should override global config."""
        # Set up global config
        global_dir = Path.home() / ".config" / "mita"
        global_dir.mkdir(parents=True)
        (global_dir / "config.toml").write_text(
            '[model]\ndefault = "codellama:13b"\ntemperature = 0.3\n'
        )

        # Set up project config
        mita_dir = tmp_project / ".mita"
        mita_dir.mkdir(exist_ok=True)
        (mita_dir / "settings.toml").write_text('[model]\ndefault = "qwen2.5-coder:14b"\n')

        cfg = load_config(project_root=tmp_project)
        # Project overrides global for 'default'
        assert cfg.model.default == "qwen2.5-coder:14b"
        # Global value preserved for 'temperature'
        assert cfg.model.temperature == 0.3

    def test_list_append_on_merge(self, tmp_project: Path) -> None:
        """Lists (like skills_paths) should be appended, not replaced."""
        global_dir = Path.home() / ".config" / "mita"
        global_dir.mkdir(parents=True)
        (global_dir / "config.toml").write_text('skills_paths = ["~/.config/mita/skills"]\n')

        mita_dir = tmp_project / ".mita"
        mita_dir.mkdir(exist_ok=True)
        (mita_dir / "settings.toml").write_text('skills_paths = [".mita/skills"]\n')

        cfg = load_config(project_root=tmp_project)
        assert "~/.config/mita/skills" in cfg.skills_paths
        assert ".mita/skills" in cfg.skills_paths


class TestConfigErrors:
    """Audit finding S1: a malformed config raises a clean ConfigError, not a traceback."""

    def test_malformed_toml_raises_config_error(self) -> None:
        from pathlib import Path

        import pytest

        from mita.config.loader import ConfigError, load_config

        global_dir = Path.home() / ".config" / "mita"
        global_dir.mkdir(parents=True)
        (global_dir / "config.toml").write_text("this is = not valid toml [[[\n")

        with pytest.raises(ConfigError, match="Malformed TOML"):
            load_config()

    def test_invalid_value_raises_config_error(self) -> None:
        from pathlib import Path

        import pytest

        from mita.config.loader import ConfigError, load_config

        global_dir = Path.home() / ".config" / "mita"
        global_dir.mkdir(parents=True)
        (global_dir / "config.toml").write_text('[model]\ntemperature = "hot"\n')

        with pytest.raises(ConfigError, match="Invalid configuration"):
            load_config()


class TestListDedup:
    """Audit finding S10: appended lists are de-duplicated so hooks don't run twice."""

    def test_duplicate_scalars_deduped(self) -> None:
        from mita.config.loader import _deep_merge

        result = _deep_merge({"items": [1, 2]}, {"items": [2, 3]})
        assert result["items"] == [1, 2, 3]

    def test_duplicate_dicts_deduped(self) -> None:
        from mita.config.loader import _deep_merge

        hook = {"event": "session_start", "command": "echo hi"}
        result = _deep_merge({"hooks": [hook]}, {"hooks": [dict(hook)]})
        assert result["hooks"] == [hook]


class TestUnknownKeys:
    """Audit finding S2: unknown config keys are detected (so the CLI can warn)."""

    def test_detects_unknown_top_level(self, tmp_project: Path) -> None:
        from mita.config.loader import find_unknown_config_keys

        mita_dir = tmp_project / ".mita"
        mita_dir.mkdir(exist_ok=True)
        (mita_dir / "settings.toml").write_text("max_iteration = 5\ntotally_bogus = 1\n")
        unknown = find_unknown_config_keys(project_root=tmp_project)
        assert "max_iteration" in unknown
        assert "totally_bogus" in unknown

    def test_detects_unknown_nested(self, tmp_project: Path) -> None:
        from mita.config.loader import find_unknown_config_keys

        mita_dir = tmp_project / ".mita"
        mita_dir.mkdir(exist_ok=True)
        (mita_dir / "settings.toml").write_text("[model]\nnope = true\n")
        unknown = find_unknown_config_keys(project_root=tmp_project)
        assert "model.nope" in unknown

    def test_known_keys_are_clean(self, tmp_project: Path) -> None:
        from mita.config.loader import find_unknown_config_keys

        mita_dir = tmp_project / ".mita"
        mita_dir.mkdir(exist_ok=True)
        (mita_dir / "settings.toml").write_text('[model]\ndefault = "x"\nmax_tokens = 2048\n')
        assert find_unknown_config_keys(project_root=tmp_project) == []
