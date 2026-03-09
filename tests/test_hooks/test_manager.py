"""Tests for hooks manager."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from mita.config.schema import HookDefinition, MitaConfig
from mita.hooks.manager import add_hook, list_hooks, remove_hooks


class TestListHooks:
    def test_no_hooks(self, capsys: pytest.CaptureFixture[str]) -> None:
        with patch("mita.hooks.manager.load_config") as mock_config:
            mock_config.return_value = MitaConfig()
            list_hooks()

    def test_with_hooks(self, capsys: pytest.CaptureFixture[str]) -> None:
        cfg = MitaConfig(
            hooks=[
                HookDefinition(event="session_start", command="echo start"),
                HookDefinition(event="on_file_write", command="ruff {file_path}", match="*.py"),
            ]
        )
        with patch("mita.hooks.manager.load_config", return_value=cfg):
            list_hooks()


class TestAddHook:
    def test_invalid_event(self) -> None:
        with patch("mita.hooks.manager.console"):
            add_hook("invalid_event", "echo hi")

    def test_add_hook_creates_file(self, tmp_path: Path) -> None:
        config_dir = tmp_path / ".mita"
        config_dir.mkdir()
        settings = config_dir / "settings.toml"
        settings.write_text("")

        with (
            patch("mita.config.defaults.get_project_config_path", return_value=settings),
            patch("mita.hooks.manager.console"),
        ):
            add_hook("on_file_write", "ruff check {file_path}", match="*.py")

        content = settings.read_text()
        assert "on_file_write" in content
        assert "ruff check {file_path}" in content
        assert "*.py" in content

    def test_add_hook_without_match(self, tmp_path: Path) -> None:
        config_dir = tmp_path / ".mita"
        config_dir.mkdir()
        settings = config_dir / "settings.toml"
        settings.write_text("")

        with (
            patch("mita.config.defaults.get_project_config_path", return_value=settings),
            patch("mita.hooks.manager.console"),
        ):
            add_hook("session_start", "echo hello")

        content = settings.read_text()
        assert "session_start" in content
        assert "echo hello" in content
        assert "match" not in content

    def test_add_hook_no_project(self) -> None:
        with (
            patch("mita.config.defaults.get_project_config_path", return_value=None),
            patch("mita.config.defaults._find_project_root", return_value=None),
            patch("mita.hooks.manager.console"),
        ):
            add_hook("session_start", "echo hi")


class TestRemoveHooks:
    def test_no_project_config(self) -> None:
        with (
            patch("mita.config.defaults.get_project_config_path", return_value=None),
            patch("mita.hooks.manager.console"),
        ):
            remove_hooks("session_start")

    def test_event_not_found(self, tmp_path: Path) -> None:
        settings = tmp_path / "settings.toml"
        settings.write_text("")

        with (
            patch("mita.config.defaults.get_project_config_path", return_value=settings),
            patch("mita.hooks.manager.console"),
        ):
            remove_hooks("session_start")

    def test_remove_existing_hook(self, tmp_path: Path) -> None:
        settings = tmp_path / "settings.toml"
        settings.write_text(
            '[[hooks]]\nevent = "session_start"\ncommand = "echo hi"\n\n'
            '[[hooks]]\nevent = "on_file_write"\ncommand = "ruff {file_path}"\n'
        )

        with (
            patch("mita.config.defaults.get_project_config_path", return_value=settings),
            patch("mita.hooks.manager.console"),
        ):
            remove_hooks("session_start")

        content = settings.read_text()
        assert "session_start" not in content
        assert "on_file_write" in content
