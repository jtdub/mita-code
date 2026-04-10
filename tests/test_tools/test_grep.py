"""Extended tests for the grep built-in tool."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from mita.tools.builtins.grep_tool import MAX_MATCHES, execute


class TestGrepTool:
    @pytest.mark.asyncio()
    async def test_missing_pattern(self) -> None:
        result = await execute({})
        assert result.success is False
        assert "Missing" in (result.error or "")

    @pytest.mark.asyncio()
    async def test_empty_pattern(self) -> None:
        result = await execute({"pattern": ""})
        assert result.success is False

    @pytest.mark.asyncio()
    async def test_invalid_regex(self) -> None:
        result = await execute({"pattern": "[invalid"})
        assert result.success is False
        assert "regex" in (result.error or "").lower()

    @pytest.mark.asyncio()
    async def test_file_not_found(self) -> None:
        result = await execute({"pattern": "test", "path": "/nonexistent/path"})
        assert result.success is False
        assert "not found" in (result.error or "").lower()

    @pytest.mark.asyncio()
    async def test_search_single_file(self, tmp_path: Path) -> None:
        f = tmp_path / "test.py"
        f.write_text("def hello():\n    pass\ndef world():\n    pass\n")

        with patch("mita.tools.safety.validate_path_for_read", return_value=None):
            result = await execute({"pattern": "def \\w+", "path": str(f)})
        assert result.success is True
        assert "hello" in result.output
        assert "world" in result.output
        assert "2 match" in result.output

    @pytest.mark.asyncio()
    async def test_search_directory(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("foo = 1\nbar = 2\n")
        (tmp_path / "b.py").write_text("baz = 3\nfoo = 4\n")

        with patch("mita.tools.safety.validate_search_base", return_value=None):
            result = await execute({"pattern": "foo", "path": str(tmp_path)})
        assert result.success is True
        assert "foo" in result.output

    @pytest.mark.asyncio()
    async def test_search_directory_with_include(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("target = 1\n")
        (tmp_path / "b.txt").write_text("target = 2\n")

        with patch("mita.tools.safety.validate_search_base", return_value=None):
            result = await execute({"pattern": "target", "path": str(tmp_path), "include": "*.py"})
        assert result.success is True
        assert "a.py" in result.output
        assert "b.txt" not in result.output

    @pytest.mark.asyncio()
    async def test_no_matches(self, tmp_path: Path) -> None:
        (tmp_path / "test.py").write_text("nothing here\n")

        with patch("mita.tools.safety.validate_search_base", return_value=None):
            result = await execute({"pattern": "xyz123", "path": str(tmp_path)})
        assert result.success is True
        assert "No matches" in result.output

    @pytest.mark.asyncio()
    async def test_max_matches_limit(self, tmp_path: Path) -> None:
        # Create a file with many matching lines
        lines = [f"match line {i}" for i in range(MAX_MATCHES + 50)]
        (tmp_path / "big.txt").write_text("\n".join(lines))

        with patch("mita.tools.safety.validate_search_base", return_value=None):
            result = await execute({"pattern": "match", "path": str(tmp_path)})
        assert result.success is True
        assert f"first {MAX_MATCHES}" in result.output

    @pytest.mark.asyncio()
    async def test_default_path(self, tmp_path: Path) -> None:
        """When path is None, defaults to '.'."""
        result = await execute({"pattern": "test", "path": None})
        # Just verify it doesn't crash - the path resolution will use cwd
        assert result is not None

    @pytest.mark.asyncio()
    async def test_sensitive_file_blocked(self, tmp_path: Path) -> None:
        f = tmp_path / ".ssh" / "id_rsa"
        f.parent.mkdir()
        f.write_text("secret key")

        with patch(
            "mita.tools.safety.validate_path_for_read",
            return_value="Access denied",
        ):
            result = await execute({"pattern": "secret", "path": str(f)})
        assert result.success is False
        assert "Access denied" in (result.error or "")

    @pytest.mark.asyncio()
    async def test_search_base_blocked(self, tmp_path: Path) -> None:
        with patch(
            "mita.tools.safety.validate_search_base",
            return_value="Access denied: outside workspace",
        ):
            result = await execute({"pattern": "test", "path": str(tmp_path)})
        assert result.success is False

    @pytest.mark.asyncio()
    async def test_unreadable_file_skipped(self, tmp_path: Path) -> None:
        """Files that can't be read are silently skipped."""
        f = tmp_path / "test.py"
        f.write_text("hello")
        # Make it unreadable by replacing read_text
        with (
            patch("mita.tools.safety.validate_search_base", return_value=None),
            patch.object(Path, "read_text", side_effect=OSError("permission denied")),
        ):
            result = await execute({"pattern": "hello", "path": str(tmp_path)})
        # Should succeed but find no matches (files skipped)
        assert result.success is True
