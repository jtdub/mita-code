"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def tmp_project(tmp_path: Path) -> Path:
    """Create a temporary project directory with a .git marker."""
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    return tmp_path


@pytest.fixture
def tmp_project_with_mita(tmp_project: Path) -> Path:
    """Create a temporary project with a .mita directory."""
    mita_dir = tmp_project / ".mita"
    mita_dir.mkdir()
    return tmp_project


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate HOME so tests don't read/write the real global config."""
    fake_home = tmp_path / "fakehome"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    # Also patch Path.home() for consistency
    monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate the working directory.

    Without this, project-root discovery walks up from the real cwd and loads the
    repository's own .mita/settings.toml (which declares a ruff --fix file-write
    hook), so the suite could execute that hook against source files. Chdir into a
    neutral temp dir with no project markers above it. See audit finding C8.
    """
    workdir = tmp_path / "cwd"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
