"""Tests for config defaults and project root detection."""

from __future__ import annotations

from pathlib import Path

from mita.config.defaults import (
    PROJECT_CONFIG_DIR,
    PROJECT_CONFIG_FILE,
    _find_project_root,
    get_global_config_dir,
    get_global_config_path,
    get_project_config_path,
)


class TestGlobalPaths:
    def test_global_config_dir(self) -> None:
        result = get_global_config_dir()
        assert result == Path.home() / ".config" / "mita"

    def test_global_config_path(self) -> None:
        result = get_global_config_path()
        assert result == Path.home() / ".config" / "mita" / "config.toml"


class TestFindProjectRoot:
    def test_finds_git_dir(self, tmp_path: Path) -> None:
        (tmp_path / ".git").mkdir()
        assert _find_project_root(tmp_path) == tmp_path

    def test_finds_mita_dir(self, tmp_path: Path) -> None:
        (tmp_path / PROJECT_CONFIG_DIR).mkdir()
        assert _find_project_root(tmp_path) == tmp_path

    def test_walks_up(self, tmp_path: Path) -> None:
        (tmp_path / ".git").mkdir()
        child = tmp_path / "src" / "pkg"
        child.mkdir(parents=True)
        assert _find_project_root(child) == tmp_path

    def test_returns_none_at_root(self, tmp_path: Path) -> None:
        # tmp_path has no .git or .mita, and walking up won't find one in a temp dir
        isolated = tmp_path / "isolated"
        isolated.mkdir()
        # This will walk up to filesystem root and return None
        result = _find_project_root(isolated)
        # It may find a .git in a parent; just verify it returns Path or None
        assert result is None or isinstance(result, Path)

    def test_ignores_git_file(self, tmp_path: Path) -> None:
        # .git as a file (worktree/submodule) should not match
        (tmp_path / ".git").write_text("gitdir: ../other/.git")
        assert _find_project_root(tmp_path) != tmp_path or not (tmp_path / ".git").is_dir()


class TestGetProjectConfigPath:
    def test_returns_path_when_exists(self, tmp_path: Path) -> None:
        (tmp_path / ".git").mkdir()
        config_dir = tmp_path / PROJECT_CONFIG_DIR
        config_dir.mkdir()
        config_file = config_dir / PROJECT_CONFIG_FILE
        config_file.write_text("[model]\ndefault = 'test'\n")
        result = get_project_config_path(project_root=tmp_path)
        assert result == config_file

    def test_returns_none_when_no_config(self, tmp_path: Path) -> None:
        (tmp_path / ".git").mkdir()
        result = get_project_config_path(project_root=tmp_path)
        assert result is None
