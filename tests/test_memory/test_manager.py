"""Tests for memory manager CLI commands."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from mita.memory.manager import (
    _ensure_memory_file,
    add_memory,
    edit_memory,
    show_memory,
    show_memory_paths,
)


class TestEnsureMemoryFile:
    def test_creates_file_if_not_exists(self, tmp_path: Path) -> None:
        path = tmp_path / "subdir" / "MITA.md"
        _ensure_memory_file(path, "Test Title")
        assert path.exists()
        content = path.read_text()
        assert "# Test Title" in content

    def test_noop_if_exists(self, tmp_path: Path) -> None:
        path = tmp_path / "MITA.md"
        path.write_text("existing content")
        _ensure_memory_file(path, "Should Not Change")
        assert path.read_text() == "existing content"


class TestShowMemory:
    def test_no_files(self) -> None:
        with patch("mita.memory.manager.load_memory_raw", return_value=[]):
            with patch("mita.memory.manager.console") as mock_console:
                show_memory()
                mock_console.print.assert_called()

    def test_with_files(self, tmp_path: Path) -> None:
        path = tmp_path / "MITA.md"
        path.write_text("# Test\n- item")
        with patch(
            "mita.memory.manager.load_memory_raw",
            return_value=[(path, "project", "# Test\n- item")],
        ):
            with patch("mita.memory.manager.console") as mock_console:
                show_memory()
                mock_console.print.assert_called()

    def test_empty_content(self, tmp_path: Path) -> None:
        path = tmp_path / "MITA.md"
        path.write_text("")
        with patch(
            "mita.memory.manager.load_memory_raw",
            return_value=[(path, "project", "")],
        ):
            with patch("mita.memory.manager.console") as mock_console:
                show_memory()
                mock_console.print.assert_called()


class TestShowMemoryPaths:
    def test_no_files(self) -> None:
        with patch("mita.memory.manager.discover_memory_files", return_value=[]):
            with patch("mita.memory.manager.console") as mock_console:
                show_memory_paths()
                mock_console.print.assert_called()

    def test_with_files(self, tmp_path: Path) -> None:
        paths = [tmp_path / "MITA.md"]
        with patch("mita.memory.manager.discover_memory_files", return_value=paths):
            with patch("mita.memory.manager.console") as mock_console:
                show_memory_paths()
                mock_console.print.assert_called()


class TestEditMemory:
    def test_global(self) -> None:
        with (
            patch("mita.memory.manager.subprocess.run") as mock_run,
            patch("mita.memory.manager.console"),
        ):
            mock_run.return_value = MagicMock(returncode=0)
            edit_memory("global")
            mock_run.assert_called_once()

    def test_project(self, tmp_project: Path) -> None:
        with (
            patch("mita.memory.manager.Path.cwd", return_value=tmp_project),
            patch("mita.memory.manager.subprocess.run") as mock_run,
            patch("mita.memory.manager.console"),
        ):
            mock_run.return_value = MagicMock(returncode=0)
            edit_memory("project")
            mock_run.assert_called_once()

    def test_project_no_root(self, tmp_path: Path) -> None:
        # tmp_path has no .git or .mita
        fake_dir = tmp_path / "no_project"
        fake_dir.mkdir()
        with (
            patch("mita.memory.manager._find_project_root", return_value=None),
            patch("mita.memory.manager.console") as mock_console,
        ):
            edit_memory("project")
            # Should print error about not being in a project
            args_str = str(mock_console.print.call_args)
            assert "project" in args_str.lower() or "not" in args_str.lower()

    def test_default_with_local_files(self, tmp_project: Path) -> None:
        mem_file = tmp_project / "MITA.md"
        mem_file.write_text("# Memory")
        with (
            patch(
                "mita.memory.manager.discover_memory_files",
                return_value=[mem_file],
            ),
            patch("mita.memory.manager.subprocess.run") as mock_run,
            patch("mita.memory.manager.console"),
        ):
            mock_run.return_value = MagicMock(returncode=0)
            edit_memory()
            mock_run.assert_called_once()

    def test_default_no_local_creates_at_project_root(self, tmp_project: Path) -> None:
        with (
            patch("mita.memory.manager.discover_memory_files", return_value=[]),
            patch("mita.memory.manager._find_project_root", return_value=tmp_project),
            patch("mita.memory.manager.subprocess.run") as mock_run,
            patch("mita.memory.manager.console"),
        ):
            mock_run.return_value = MagicMock(returncode=0)
            edit_memory()
            mock_run.assert_called_once()

    def test_default_no_project_root(self) -> None:
        with (
            patch("mita.memory.manager.discover_memory_files", return_value=[]),
            patch("mita.memory.manager._find_project_root", return_value=None),
            patch("mita.memory.manager.console") as mock_console,
        ):
            edit_memory()
            args_str = str(mock_console.print.call_args)
            assert "project" in args_str.lower() or "not" in args_str.lower()


class TestAddMemory:
    def test_add_global(self, tmp_path: Path) -> None:
        global_path = tmp_path / "fakehome" / ".config" / "mita" / "MITA.md"
        with (
            patch("mita.memory.manager.get_global_memory_path", return_value=global_path),
            patch("mita.memory.manager.console"),
        ):
            add_memory("test note", "global")
            assert global_path.exists()
            assert "- test note" in global_path.read_text()

    def test_add_project(self, tmp_project: Path) -> None:
        with (
            patch("mita.memory.manager._find_project_root", return_value=tmp_project),
            patch("mita.memory.manager.Path.cwd", return_value=tmp_project),
            patch("mita.memory.manager.console"),
        ):
            add_memory("project note", "project")
            mem_file = tmp_project / "MITA.md"
            assert mem_file.exists()
            assert "- project note" in mem_file.read_text()

    def test_add_no_project_root(self) -> None:
        with (
            patch("mita.memory.manager._find_project_root", return_value=None),
            patch("mita.memory.manager.console") as mock_console,
        ):
            add_memory("orphan note", "project")
            args_str = str(mock_console.print.call_args)
            assert "project" in args_str.lower() or "not" in args_str.lower()
