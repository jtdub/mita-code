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
