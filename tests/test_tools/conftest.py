"""Fixtures for tool tests."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

import mita.tools.safety as safety_module


@pytest.fixture(autouse=True)
def _workspace_root_to_tmp(tmp_path: Path) -> Iterator[None]:
    """Set the workspace root to tmp_path so file tools work with temp dirs."""
    safety_module._workspace_root_override = tmp_path
    yield
    safety_module._workspace_root_override = None
