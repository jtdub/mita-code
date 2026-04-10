"""Tests for built-in tool handlers."""

from __future__ import annotations

from pathlib import Path

import pytest

from mita.tools.builtins.file_edit import execute as file_edit
from mita.tools.builtins.file_read import execute as file_read
from mita.tools.builtins.file_write import execute as file_write
from mita.tools.builtins.glob_tool import execute as glob_exec
from mita.tools.builtins.grep_tool import execute as grep_exec
from mita.tools.builtins.shell import execute as shell_exec


class TestFileRead:
    @pytest.mark.asyncio()
    async def test_read_existing_file(self, tmp_path: object) -> None:
        from pathlib import Path

        p = Path(str(tmp_path)) / "test.txt"
        p.write_text("line1\nline2\nline3\n")
        result = await file_read({"path": str(p)})
        assert result.success is True
        assert "line1" in result.output
        assert "line2" in result.output

    @pytest.mark.asyncio()
    async def test_read_with_offset_and_limit(self, tmp_path: object) -> None:
        from pathlib import Path

        p = Path(str(tmp_path)) / "test.txt"
        p.write_text("a\nb\nc\nd\ne\n")
        result = await file_read({"path": str(p), "offset": 2, "limit": 2})
        assert result.success is True
        assert "b" in result.output
        assert "c" in result.output

    @pytest.mark.asyncio()
    async def test_read_outside_workspace_blocked(self) -> None:
        result = await file_read({"path": "/nonexistent/file.txt"})
        assert result.success is False
        assert "access denied" in (result.error or "").lower()

    @pytest.mark.asyncio()
    async def test_read_nonexistent_in_workspace(self, tmp_path: Path) -> None:
        from unittest.mock import patch

        missing = tmp_path / "does_not_exist.txt"
        with patch("mita.tools.safety._workspace_root_override", tmp_path):
            result = await file_read({"path": str(missing)})
        assert result.success is False
        assert "not found" in (result.error or "").lower()

    @pytest.mark.asyncio()
    async def test_read_missing_path(self) -> None:
        result = await file_read({})
        assert result.success is False


class TestFileWrite:
    @pytest.mark.asyncio()
    async def test_write_new_file(self, tmp_path: object) -> None:
        from pathlib import Path

        p = Path(str(tmp_path)) / "new.txt"
        result = await file_write({"path": str(p), "content": "hello world"})
        assert result.success is True
        assert "Created" in result.output
        assert p.read_text() == "hello world"

    @pytest.mark.asyncio()
    async def test_write_overwrites_existing(self, tmp_path: object) -> None:
        from pathlib import Path

        p = Path(str(tmp_path)) / "existing.txt"
        p.write_text("old")
        result = await file_write({"path": str(p), "content": "new"})
        assert result.success is True
        assert "Updated" in result.output
        assert p.read_text() == "new"

    @pytest.mark.asyncio()
    async def test_write_creates_parents(self, tmp_path: object) -> None:
        from pathlib import Path

        p = Path(str(tmp_path)) / "deep" / "nested" / "file.txt"
        result = await file_write({"path": str(p), "content": "nested"})
        assert result.success is True
        assert p.exists()


class TestFileEdit:
    @pytest.mark.asyncio()
    async def test_edit_replaces_string(self, tmp_path: object) -> None:
        from pathlib import Path

        p = Path(str(tmp_path)) / "edit.txt"
        p.write_text("hello world")
        result = await file_edit({"path": str(p), "old_string": "world", "new_string": "python"})
        assert result.success is True
        assert p.read_text() == "hello python"

    @pytest.mark.asyncio()
    async def test_edit_string_not_found(self, tmp_path: object) -> None:
        from pathlib import Path

        p = Path(str(tmp_path)) / "edit.txt"
        p.write_text("hello world")
        result = await file_edit({"path": str(p), "old_string": "xyz", "new_string": "abc"})
        assert result.success is False
        assert "not found" in (result.error or "").lower()

    @pytest.mark.asyncio()
    async def test_edit_ambiguous_match(self, tmp_path: object) -> None:
        from pathlib import Path

        p = Path(str(tmp_path)) / "edit.txt"
        p.write_text("aaa bbb aaa")
        result = await file_edit({"path": str(p), "old_string": "aaa", "new_string": "ccc"})
        assert result.success is False
        assert "2 times" in (result.error or "")


class TestGlob:
    @pytest.mark.asyncio()
    async def test_glob_finds_files(self, tmp_path: object) -> None:
        from pathlib import Path

        d = Path(str(tmp_path))
        (d / "a.py").write_text("")
        (d / "b.py").write_text("")
        (d / "c.txt").write_text("")
        result = await glob_exec({"pattern": "*.py", "path": str(d)})
        assert result.success is True
        assert "a.py" in result.output
        assert "b.py" in result.output
        assert "c.txt" not in result.output

    @pytest.mark.asyncio()
    async def test_glob_no_matches(self, tmp_path: object) -> None:
        result = await glob_exec({"pattern": "*.xyz", "path": str(tmp_path)})
        assert result.success is True
        assert "No files matched" in result.output


class TestGrep:
    @pytest.mark.asyncio()
    async def test_grep_finds_pattern(self, tmp_path: object) -> None:
        from pathlib import Path

        p = Path(str(tmp_path)) / "test.py"
        p.write_text("def hello():\n    pass\ndef world():\n    pass\n")
        result = await grep_exec({"pattern": "def \\w+", "path": str(p)})
        assert result.success is True
        assert "hello" in result.output
        assert "world" in result.output

    @pytest.mark.asyncio()
    async def test_grep_no_matches(self, tmp_path: object) -> None:
        from pathlib import Path

        p = Path(str(tmp_path)) / "test.txt"
        p.write_text("nothing here")
        result = await grep_exec({"pattern": "xyz123", "path": str(p)})
        assert result.success is True
        assert "No matches" in result.output

    @pytest.mark.asyncio()
    async def test_grep_invalid_regex(self) -> None:
        result = await grep_exec({"pattern": "[invalid", "path": "."})
        assert result.success is False
        assert "regex" in (result.error or "").lower()


class TestShell:
    @pytest.mark.asyncio()
    async def test_echo(self) -> None:
        result = await shell_exec({"command": "echo hello"})
        assert result.success is True
        assert "hello" in result.output

    @pytest.mark.asyncio()
    async def test_failing_command(self) -> None:
        result = await shell_exec({"command": "false"})
        assert result.success is False

    @pytest.mark.asyncio()
    async def test_timeout(self) -> None:
        result = await shell_exec({"command": "sleep 10", "timeout": 1})
        assert result.success is False
        assert "timed out" in (result.error or "").lower()

    @pytest.mark.asyncio()
    async def test_missing_command(self) -> None:
        result = await shell_exec({})
        assert result.success is False
