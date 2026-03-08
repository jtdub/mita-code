"""Tests for Tree-sitter code parsing and chunking."""

from __future__ import annotations

from pathlib import Path

from mita.config.schema import IndexSettings
from mita.index.parser import (
    _chunk_by_lines,
    _is_binary,
    _matches_exclude,
    get_language_for_file,
    parse_codebase,
    parse_file,
)


class TestLanguageDetection:
    def test_python(self) -> None:
        assert get_language_for_file(Path("foo.py")) == "python"

    def test_javascript(self) -> None:
        assert get_language_for_file(Path("app.js")) == "javascript"

    def test_typescript(self) -> None:
        assert get_language_for_file(Path("app.ts")) == "typescript"

    def test_go(self) -> None:
        assert get_language_for_file(Path("main.go")) == "go"

    def test_rust(self) -> None:
        assert get_language_for_file(Path("lib.rs")) == "rust"

    def test_unknown(self) -> None:
        assert get_language_for_file(Path("data.xyz")) is None

    def test_case_insensitive_suffix(self) -> None:
        assert get_language_for_file(Path("file.PY")) == "python"


class TestExcludePatterns:
    def test_matches_lock_file(self) -> None:
        assert _matches_exclude("poetry.lock", ["*.lock"])

    def test_matches_node_modules(self) -> None:
        assert _matches_exclude("node_modules/foo/bar.js", ["node_modules/**"])

    def test_matches_git(self) -> None:
        assert _matches_exclude(".git/config", [".git/**"])

    def test_no_match(self) -> None:
        assert not _matches_exclude("src/main.py", ["*.lock", ".git/**"])

    def test_pycache(self) -> None:
        assert _matches_exclude("__pycache__/foo.pyc", ["__pycache__/**"])


class TestBinaryDetection:
    def test_text_file(self, tmp_path: Path) -> None:
        f = tmp_path / "test.py"
        f.write_text("print('hello')")
        assert not _is_binary(f)

    def test_binary_file(self, tmp_path: Path) -> None:
        f = tmp_path / "test.bin"
        f.write_bytes(b"\x00\x01\x02\x03")
        assert _is_binary(f)


class TestParseFilePython:
    def test_parses_functions(self, tmp_path: Path) -> None:
        code = '''def hello():
    """Say hello."""
    print("hello")


def goodbye():
    """Say goodbye."""
    print("goodbye")
'''
        f = tmp_path / "test.py"
        f.write_text(code)
        config = IndexSettings()
        chunks = parse_file(f, tmp_path, config)
        assert len(chunks) >= 2
        # Check that chunks have correct language
        for c in chunks:
            assert c.language == "python"
        # Should find function symbols
        symbols = [c.symbol for c in chunks if c.symbol]
        assert "hello" in symbols
        assert "goodbye" in symbols

    def test_parses_class(self, tmp_path: Path) -> None:
        code = """class Greeter:
    def greet(self):
        pass
"""
        f = tmp_path / "test.py"
        f.write_text(code)
        config = IndexSettings()
        chunks = parse_file(f, tmp_path, config)
        assert len(chunks) >= 1
        symbols = [c.symbol for c in chunks if c.symbol]
        assert "Greeter" in symbols


class TestChunkByLines:
    def test_basic_chunking(self) -> None:
        content = "\n".join(f"line {i}" for i in range(100))
        config = IndexSettings(chunk_size=20, chunk_overlap=2)
        chunks = _chunk_by_lines(content, "test.txt", "text", config)
        assert len(chunks) > 1
        # All chunks should have valid line numbers
        for c in chunks:
            assert c.start_line >= 1
            assert c.end_line >= c.start_line

    def test_small_file_single_chunk(self) -> None:
        content = "line 1\nline 2\nline 3"
        config = IndexSettings(chunk_size=512)
        chunks = _chunk_by_lines(content, "test.txt", "text", config)
        assert len(chunks) == 1

    def test_empty_content_no_chunks(self) -> None:
        config = IndexSettings()
        chunks = _chunk_by_lines("", "test.txt", "text", config)
        assert len(chunks) == 0


class TestParseCodebase:
    def test_discovers_and_parses(self, tmp_path: Path) -> None:
        # Create a simple project
        (tmp_path / "main.py").write_text("def main():\n    pass\n")
        (tmp_path / "readme.md").write_text("# Hello\nThis is a project.\n")
        (tmp_path / "data.lock").write_text("lock data")

        config = IndexSettings(exclude_patterns=["*.lock"])
        chunks = parse_codebase(tmp_path, config)
        # Should have chunks from main.py and readme.md but not data.lock
        file_paths = {c.file_path for c in chunks}
        assert "main.py" in file_paths
        assert "data.lock" not in file_paths

    def test_skips_binary(self, tmp_path: Path) -> None:
        (tmp_path / "binary.dat").write_bytes(b"\x00" * 100)
        (tmp_path / "text.py").write_text("x = 1\n")

        config = IndexSettings(exclude_patterns=[])
        chunks = parse_codebase(tmp_path, config)
        file_paths = {c.file_path for c in chunks}
        assert "binary.dat" not in file_paths

    def test_empty_dir(self, tmp_path: Path) -> None:
        config = IndexSettings()
        chunks = parse_codebase(tmp_path, config)
        assert chunks == []
