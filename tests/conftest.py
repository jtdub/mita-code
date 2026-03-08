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
