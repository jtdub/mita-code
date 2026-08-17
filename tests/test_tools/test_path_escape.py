"""Regression tests: glob/grep must not escape the workspace (audit finding C3)."""

from __future__ import annotations

from pathlib import Path

import pytest

import mita.tools.safety as safety_module
from mita.tools.builtins.glob_tool import execute as glob_execute
from mita.tools.builtins.grep_tool import execute as grep_execute


@pytest.fixture
def repo_with_outside_secret(tmp_path: Path) -> tuple[Path, Path]:
    """A workspace 'repo' with a secret file located OUTSIDE it."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("inside the repo\n")
    secret = tmp_path / "secret.txt"
    secret.write_text("TOPSECRET_VALUE\n")
    safety_module._workspace_root_override = repo
    return repo, secret


class TestGrepEscape:
    async def test_include_traversal_blocked(
        self, repo_with_outside_secret: tuple[Path, Path]
    ) -> None:
        repo, _ = repo_with_outside_secret
        result = await grep_execute(
            {"pattern": "TOPSECRET", "path": str(repo), "include": "../secret.txt"}
        )
        assert result.success is False
        assert "must not contain" in (result.error or "")

    async def test_absolute_include_blocked(
        self, repo_with_outside_secret: tuple[Path, Path]
    ) -> None:
        repo, _ = repo_with_outside_secret
        result = await grep_execute({"pattern": "root", "path": str(repo), "include": "/etc/*"})
        assert result.success is False
        assert "Absolute" in (result.error or "")

    async def test_symlink_escape_not_read(
        self, repo_with_outside_secret: tuple[Path, Path]
    ) -> None:
        """A symlink inside the repo pointing outside must not have its target read."""
        repo, secret = repo_with_outside_secret
        (repo / "link.txt").symlink_to(secret)
        result = await grep_execute({"pattern": "TOPSECRET", "path": str(repo)})
        assert "TOPSECRET_VALUE" not in (result.output or "")


class TestGlobEscape:
    async def test_parent_traversal_blocked(
        self, repo_with_outside_secret: tuple[Path, Path]
    ) -> None:
        repo, _ = repo_with_outside_secret
        result = await glob_execute({"pattern": "../*", "path": str(repo)})
        assert result.success is False
        assert ".." in (result.error or "")

    async def test_absolute_pattern_blocked(
        self, repo_with_outside_secret: tuple[Path, Path]
    ) -> None:
        repo, _ = repo_with_outside_secret
        result = await glob_execute({"pattern": "/etc/*", "path": str(repo)})
        assert result.success is False
        assert "Absolute" in (result.error or "")

    async def test_symlink_escape_filtered(
        self, repo_with_outside_secret: tuple[Path, Path]
    ) -> None:
        repo, secret = repo_with_outside_secret
        (repo / "link.txt").symlink_to(secret)
        result = await glob_execute({"pattern": "*", "path": str(repo)})
        assert str(secret) not in (result.output or "")
