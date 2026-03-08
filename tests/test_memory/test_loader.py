"""Tests for MITA.md memory loading and merging."""

from pathlib import Path

from mita.config.schema import MemorySettings
from mita.memory.discovery import MEMORY_FILENAME
from mita.memory.loader import _classify_scope, _read_and_truncate, load_memory, load_memory_raw


class TestClassifyScope:
    def test_global_scope(self) -> None:
        path = Path.home() / ".config" / "mita" / MEMORY_FILENAME
        assert _classify_scope(path) == "global"

    def test_project_scope(self, tmp_project: Path) -> None:
        path = tmp_project / MEMORY_FILENAME
        assert _classify_scope(path) == "project"

    def test_project_scope_dotmita(self, tmp_project_with_mita: Path) -> None:
        path = tmp_project_with_mita / ".mita" / MEMORY_FILENAME
        assert _classify_scope(path) == "project"

    def test_directory_scope(self, tmp_project: Path) -> None:
        subdir = tmp_project / "src" / "components"
        subdir.mkdir(parents=True)
        path = subdir / MEMORY_FILENAME
        assert _classify_scope(path) == "directory"


class TestReadAndTruncate:
    def test_short_file(self, tmp_path: Path) -> None:
        f = tmp_path / "test.md"
        f.write_text("line1\nline2\nline3")
        content, truncated = _read_and_truncate(f, max_lines=10)
        assert content == "line1\nline2\nline3"
        assert truncated is False

    def test_exact_limit(self, tmp_path: Path) -> None:
        f = tmp_path / "test.md"
        f.write_text("line1\nline2\nline3")
        content, truncated = _read_and_truncate(f, max_lines=3)
        assert truncated is False

    def test_over_limit(self, tmp_path: Path) -> None:
        f = tmp_path / "test.md"
        lines = [f"line{i}" for i in range(10)]
        f.write_text("\n".join(lines))
        content, truncated = _read_and_truncate(f, max_lines=5)
        assert truncated is True
        assert "line0" in content
        assert "line4" in content
        assert "line5" not in content
        assert "truncated at 5 lines" in content


class TestLoadMemory:
    def test_no_files(self, tmp_project: Path) -> None:
        result = load_memory(tmp_project)
        assert result == ""

    def test_single_file(self, tmp_project: Path) -> None:
        mem = tmp_project / MEMORY_FILENAME
        mem.write_text("# Project conventions\n- Use pytest")
        result = load_memory(tmp_project)
        assert "<memory>" in result
        assert "</memory>" in result
        assert "# Project conventions" in result
        assert "project" in result  # scope annotation

    def test_multiple_files_merged(self, tmp_project: Path) -> None:
        global_mem = Path.home() / ".config" / "mita" / MEMORY_FILENAME
        global_mem.parent.mkdir(parents=True)
        global_mem.write_text("# Global prefs")

        project_mem = tmp_project / MEMORY_FILENAME
        project_mem.write_text("# Project rules")

        result = load_memory(tmp_project)
        assert "Global prefs" in result
        assert "Project rules" in result
        # Global should appear before project in the output
        global_pos = result.index("Global prefs")
        project_pos = result.index("Project rules")
        assert global_pos < project_pos

    def test_empty_file_skipped(self, tmp_project: Path) -> None:
        mem = tmp_project / MEMORY_FILENAME
        mem.write_text("   \n  \n")
        result = load_memory(tmp_project)
        assert result == ""

    def test_truncation(self, tmp_project: Path) -> None:
        mem = tmp_project / MEMORY_FILENAME
        lines = [f"line {i}" for i in range(300)]
        mem.write_text("\n".join(lines))

        settings = MemorySettings(max_lines_per_file=10)
        result = load_memory(tmp_project, settings=settings)
        assert "line 0" in result
        assert "line 9" in result
        assert "line 10" not in result
        assert "truncated" in result


class TestLoadMemoryRaw:
    def test_returns_tuples(self, tmp_project: Path) -> None:
        mem = tmp_project / MEMORY_FILENAME
        mem.write_text("# Test")

        result = load_memory_raw(tmp_project)
        assert len(result) == 1
        path, scope, content = result[0]
        assert path == mem
        assert scope == "project"
        assert "# Test" in content
