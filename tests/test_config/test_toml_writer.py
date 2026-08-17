"""Tests for TOML writing utilities (audit finding C7)."""

from __future__ import annotations

import tomllib
from pathlib import Path

from mita.config.schema import MitaConfig
from mita.config.toml_writer import atomic_write_text, config_to_toml, write_toml


class TestConfigToToml:
    def test_output_is_valid_toml(self) -> None:
        """The default config must serialize to TOML that reparses cleanly."""
        text = config_to_toml(MitaConfig())
        parsed = tomllib.loads(text)  # must not raise
        assert isinstance(parsed, dict)

    def test_no_bare_none_token(self) -> None:
        text = config_to_toml(MitaConfig())
        assert "= None" not in text
        assert "None" not in text.replace("none", "")  # no capital-N None token


class TestWriteToml:
    def test_atomic_and_valid(self, tmp_path: Path) -> None:
        data = {
            "top": 1,
            "names": ["a", "b"],
            "model": {"default": "x", "temperature": 0.2},
            "hooks": [{"event": "session_start", "command": "echo hi"}],
        }
        path = tmp_path / "out.toml"
        write_toml(path, data)
        parsed = tomllib.loads(path.read_text())
        assert parsed["top"] == 1
        assert parsed["names"] == ["a", "b"]
        assert parsed["model"]["default"] == "x"
        assert parsed["hooks"][0]["event"] == "session_start"

    def test_none_values_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "out.toml"
        write_toml(path, {"kept": 1, "dropped": None, "table": {"a": None, "b": 2}})
        parsed = tomllib.loads(path.read_text())
        assert parsed == {"kept": 1, "table": {"b": 2}}

    def test_empty_table_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "out.toml"
        write_toml(path, {"kept": 1, "empty": {}})
        parsed = tomllib.loads(path.read_text())
        assert parsed == {"kept": 1}


class TestAtomicWriteText:
    def test_creates_parent_and_replaces(self, tmp_path: Path) -> None:
        path = tmp_path / "sub" / "file.txt"
        atomic_write_text(path, "first")
        assert path.read_text() == "first"
        atomic_write_text(path, "second")
        assert path.read_text() == "second"
        # No leftover temp files in the directory.
        assert [p.name for p in path.parent.iterdir()] == ["file.txt"]
