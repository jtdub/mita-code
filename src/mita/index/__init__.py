"""Codebase indexing: Tree-sitter parsing, embeddings, LanceDB vector store."""

from __future__ import annotations

from pathlib import Path


def get_index_dir() -> Path:
    """Return the index directory for the current project."""
    return Path.cwd() / ".mita" / "index"
