"""Tests for MITA.md memory file discovery."""

from pathlib import Path

from mita.memory.discovery import MEMORY_FILENAME, discover_memory_files


class TestDiscoverMemoryFiles:
    def test_no_memory_files(self, tmp_project: Path) -> None:
        """No MITA.md files exist anywhere."""
        result = discover_memory_files(tmp_project)
        assert result == []

    def test_project_root_memory(self, tmp_project: Path) -> None:
        """MITA.md at project root is discovered."""
        mem = tmp_project / MEMORY_FILENAME
        mem.write_text("# Project memory")

        result = discover_memory_files(tmp_project)
        assert len(result) == 1
        assert result[0] == mem

    def test_dotmita_memory(self, tmp_project: Path) -> None:
        """MITA.md inside .mita/ is discovered."""
        mita_dir = tmp_project / ".mita"
        mita_dir.mkdir()
        mem = mita_dir / MEMORY_FILENAME
        mem.write_text("# Project memory in .mita/")

        result = discover_memory_files(tmp_project)
        assert len(result) == 1
        assert result[0] == mem

    def test_global_memory(self, tmp_project: Path) -> None:
        """Global MITA.md is discovered."""
        global_mem = Path.home() / ".config" / "mita" / MEMORY_FILENAME
        global_mem.parent.mkdir(parents=True)
        global_mem.write_text("# Global memory")

        result = discover_memory_files(tmp_project)
        assert len(result) == 1
        assert result[0] == global_mem

    def test_ordering_global_before_project(self, tmp_project: Path) -> None:
        """Global memory comes before project memory (lower priority first)."""
        global_mem = Path.home() / ".config" / "mita" / MEMORY_FILENAME
        global_mem.parent.mkdir(parents=True)
        global_mem.write_text("# Global")

        project_mem = tmp_project / MEMORY_FILENAME
        project_mem.write_text("# Project")

        result = discover_memory_files(tmp_project)
        assert len(result) == 2
        assert result[0] == global_mem  # global first (lower priority)
        assert result[1] == project_mem  # project second (higher priority)

    def test_subdirectory_memory(self, tmp_project: Path) -> None:
        """MITA.md in a subdirectory is discovered when cwd is that subdirectory."""
        subdir = tmp_project / "src"
        subdir.mkdir()

        project_mem = tmp_project / MEMORY_FILENAME
        project_mem.write_text("# Project")

        sub_mem = subdir / MEMORY_FILENAME
        sub_mem.write_text("# Subdir")

        result = discover_memory_files(subdir)
        assert len(result) == 2
        assert result[0] == project_mem  # project root first (lower priority)
        assert result[1] == sub_mem  # subdir second (higher priority)

    def test_no_duplicates_root_and_dotmita(self, tmp_project: Path) -> None:
        """Both MITA.md and .mita/MITA.md at root — no duplicates, both found."""
        root_mem = tmp_project / MEMORY_FILENAME
        root_mem.write_text("# Root")

        mita_dir = tmp_project / ".mita"
        mita_dir.mkdir()
        dotmita_mem = mita_dir / MEMORY_FILENAME
        dotmita_mem.write_text("# DotMita")

        result = discover_memory_files(tmp_project)
        assert len(result) == 2
        # Both should appear, no duplicates
        assert root_mem in result
        assert dotmita_mem in result
