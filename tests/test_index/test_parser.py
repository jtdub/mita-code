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


class TestSymbolExtractionAcrossLanguages:
    """Audit finding C6: _extract_symbol must work beyond plain Python."""

    def _parse(self, tmp_path: Path, name: str, code: str) -> list:
        (tmp_path / name).write_text(code)
        cfg = IndexSettings()
        return parse_file(tmp_path / name, tmp_path, cfg)

    def _symbols(self, chunks: list) -> set[str]:
        return {c.symbol for c in chunks if c.symbol}

    def test_python_decorated_function(self, tmp_path: Path) -> None:
        chunks = self._parse(
            tmp_path, "a.py", "import os\n\n@deco\ndef handler(x):\n    return x\n"
        )
        assert "handler" in self._symbols(chunks)

    def test_python_module_constants_are_indexed(self, tmp_path: Path) -> None:
        # The import + top-level constant are NOT inside any def; must still be indexed.
        code = "import os\n\nCONFIG = {'a': 1}\n\ndef f():\n    return CONFIG\n"
        chunks = self._parse(tmp_path, "b.py", code)
        joined = "\n".join(c.content for c in chunks)
        assert "CONFIG = {'a': 1}" in joined
        assert "import os" in joined

    def test_cpp_class(self, tmp_path: Path) -> None:
        chunks = self._parse(tmp_path, "w.cpp", "class Widget {\npublic:\n  int x;\n};\n")
        assert "Widget" in self._symbols(chunks)

    def test_cpp_function(self, tmp_path: Path) -> None:
        chunks = self._parse(tmp_path, "f.cpp", "int add(int a, int b) {\n  return a + b;\n}\n")
        assert "add" in self._symbols(chunks)

    def test_typescript_export_function(self, tmp_path: Path) -> None:
        chunks = self._parse(
            tmp_path, "m.ts", "export function hello(name: string) {\n  return name;\n}\n"
        )
        assert "hello" in self._symbols(chunks)

    def test_ruby_class(self, tmp_path: Path) -> None:
        chunks = self._parse(tmp_path, "a.rb", "class Animal\n  def speak\n    'hi'\n  end\nend\n")
        assert "Animal" in self._symbols(chunks)

    def test_go_method(self, tmp_path: Path) -> None:
        code = "package main\n\nfunc (r Recv) Handle() int {\n\treturn 1\n}\n"
        chunks = self._parse(tmp_path, "h.go", code)
        assert "Handle" in self._symbols(chunks)

    def test_chunk_type_and_hash_populated(self, tmp_path: Path) -> None:
        chunks = self._parse(tmp_path, "c.py", "def foo():\n    return 1\n")
        fn = next(c for c in chunks if c.symbol == "foo")
        assert fn.chunk_type == "function"
        assert fn.content_hash  # non-empty
        assert fn.symbol_path == "foo"


class TestIgnoreRules:
    """Audit finding C6: don't index virtualenvs, secrets, or huge files."""

    def test_venv_excluded_by_default(self, tmp_path: Path) -> None:
        (tmp_path / ".venv").mkdir()
        (tmp_path / ".venv" / "mod.py").write_text("def x():\n    return 1\n")
        (tmp_path / "app.py").write_text("def y():\n    return 2\n")
        chunks = parse_codebase(tmp_path, IndexSettings())
        paths = {c.file_path for c in chunks}
        assert "app.py" in paths
        assert not any(".venv" in p for p in paths)

    def test_secret_files_excluded(self, tmp_path: Path) -> None:
        (tmp_path / ".env").write_text("SECRET=abc\n")
        (tmp_path / "id_rsa").write_text("PRIVATE KEY\n")
        (tmp_path / "app.py").write_text("x = 1\n")
        chunks = parse_codebase(tmp_path, IndexSettings())
        paths = {c.file_path for c in chunks}
        assert ".env" not in paths
        assert "id_rsa" not in paths

    def test_max_file_size_skipped(self, tmp_path: Path) -> None:
        big = tmp_path / "big.py"
        big.write_text("# " + "x" * 5000 + "\ndef f():\n    return 1\n")
        cfg = IndexSettings(max_file_size=100)
        assert parse_file(big, tmp_path, cfg) == []
