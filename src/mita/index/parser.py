"""Tree-sitter code parsing and chunking for codebase indexing."""

from __future__ import annotations

import hashlib
import logging
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from mita.config.schema import IndexSettings
from mita.index.store import CodeChunk

# Rough estimate: ~4 characters per token for budget calculations
CHARS_PER_TOKEN = 4


def content_hash(text: str) -> str:
    """Stable hash of chunk content, for incremental re-embedding (finding C6)."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


# Extension → tree-sitter language name
EXTENSION_MAP: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cc": "cpp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".lua": "lua",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".css": "css",
    ".html": "html",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
}

# Node types that represent semantic code units per language
CHUNK_NODE_TYPES: dict[str, set[str]] = {
    "python": {"function_definition", "class_definition", "decorated_definition"},
    "javascript": {
        "function_declaration",
        "class_declaration",
        "method_definition",
        "export_statement",
    },
    "typescript": {
        "function_declaration",
        "class_declaration",
        "method_definition",
        "export_statement",
        "interface_declaration",
        "type_alias_declaration",
    },
    "tsx": {
        "function_declaration",
        "class_declaration",
        "method_definition",
        "export_statement",
        "interface_declaration",
        "type_alias_declaration",
    },
    "go": {"function_declaration", "method_declaration", "type_declaration"},
    "rust": {"function_item", "impl_item", "struct_item", "enum_item", "trait_item"},
    "java": {"class_declaration", "method_declaration", "interface_declaration"},
    "c": {"function_definition", "struct_specifier"},
    "cpp": {"function_definition", "class_specifier", "struct_specifier"},
    "ruby": {"method", "class", "module", "singleton_method"},
    "php": {"function_definition", "class_declaration", "method_declaration"},
    "swift": {"function_declaration", "class_declaration", "protocol_declaration"},
    "kotlin": {"function_declaration", "class_declaration", "object_declaration"},
    "scala": {"function_definition", "class_definition", "object_definition", "trait_definition"},
    "lua": {"function_declaration", "function_definition"},
    "bash": {"function_definition"},
}

# Coarse classification of a definition node type for the chunk_type column.
_CLASS_LIKE = (
    "class",
    "struct",
    "impl",
    "interface",
    "trait",
    "enum",
    "module",
    "object",
    "protocol",
    "type",
)
_FUNC_LIKE = ("function", "method", "func")


def _classify(node_type: str) -> str:
    """Map a tree-sitter node type to a coarse chunk_type."""
    lowered = node_type.lower()
    if any(tok in lowered for tok in _CLASS_LIKE):
        return "class"
    if any(tok in lowered for tok in _FUNC_LIKE):
        return "function"
    return "definition"


def parse_codebase(root: Path, config: IndexSettings) -> list[CodeChunk]:
    """Parse all supported files under root into code chunks."""
    chunks: list[CodeChunk] = []
    for file_path in _discover_files(root, config.exclude_patterns):
        file_chunks = parse_file(file_path, root, config)
        chunks.extend(file_chunks)
    return chunks


def parse_file(file_path: Path, root: Path, config: IndexSettings) -> list[CodeChunk]:
    """Parse a single file into code chunks."""
    if _is_binary(file_path):
        return []

    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    if not content.strip():
        return []

    rel_path = str(file_path.relative_to(root))
    language = get_language_for_file(file_path)

    if language and language in CHUNK_NODE_TYPES:
        chunks = _parse_with_treesitter(content, rel_path, language, config)
        if chunks:
            return chunks

    # Fallback: line-based chunking
    return _chunk_by_lines(content, rel_path, language or "text", config)


def get_language_for_file(path: Path) -> str | None:
    """Map a file extension to a tree-sitter language name."""
    return EXTENSION_MAP.get(path.suffix.lower())


def _discover_files(root: Path, exclude_patterns: list[str]) -> list[Path]:
    """Walk the project tree, skipping excluded directories early.

    Uses os.walk instead of rglob so we can prune entire directory
    subtrees (e.g. node_modules, .git) before descending into them.
    """
    import os

    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root)
        # Prune excluded directories in-place so os.walk skips them
        dirnames[:] = sorted(
            d
            for d in dirnames
            if not _matches_exclude(f"{rel_dir}/{d}" if rel_dir != "." else d, exclude_patterns)
        )
        for filename in sorted(filenames):
            rel_file = f"{rel_dir}/{filename}" if rel_dir != "." else filename
            if not _matches_exclude(rel_file, exclude_patterns):
                files.append(Path(dirpath) / filename)
    return files


def _matches_exclude(rel_path: str, patterns: list[str]) -> bool:
    """Check if a relative path matches any exclude pattern."""
    for pattern in patterns:
        if fnmatch(rel_path, pattern):
            return True
        # Also check each path component for directory patterns
        parts = rel_path.split("/")
        for i in range(len(parts)):
            partial = "/".join(parts[: i + 1])
            if fnmatch(partial, pattern.rstrip("/")):
                return True
            if fnmatch(partial + "/", pattern):
                return True
    return False


def _is_binary(path: Path) -> bool:
    """Heuristic: file is binary if first 8KB contains null bytes."""
    try:
        with open(path, "rb") as f:
            chunk = f.read(8192)
        return b"\x00" in chunk
    except OSError:
        return True


def _parse_with_treesitter(
    content: str,
    rel_path: str,
    language: str,
    config: IndexSettings,
) -> list[CodeChunk]:
    """Parse a file using tree-sitter and extract semantic chunks."""
    try:
        from tree_sitter_language_pack import get_parser
    except ImportError:
        return []

    try:
        parser = get_parser(language)  # type: ignore[arg-type]
    except Exception:  # noqa: BLE001
        logging.getLogger(__name__).warning(
            "Tree-sitter parser unavailable for language '%s'", language
        )
        return []

    tree = parser.parse(content.encode("utf-8"))
    node_types = CHUNK_NODE_TYPES.get(language, set())
    lines = content.split("\n")
    char_budget = config.chunk_size * CHARS_PER_TOKEN
    chunks: list[CodeChunk] = []
    covered: list[tuple[int, int]] = []  # 1-indexed inclusive line ranges

    for node in _collect_definitions(tree.root_node, node_types):
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        covered.append((start_line, end_line))
        node_content = "\n".join(lines[start_line - 1 : end_line])
        symbol = _extract_symbol(node)
        ctype = _classify(node.type)

        if len(node_content) > char_budget:
            chunks.extend(
                _split_large_content(
                    node_content, start_line, rel_path, language, symbol, ctype, config
                )
            )
        else:
            chunks.append(
                _make_chunk(rel_path, start_line, end_line, node_content, language, symbol, ctype)
            )

    # Capture content NOT covered by any definition (imports, module constants, top-level
    # config tables) so it is indexed rather than silently dropped (finding C6).
    chunks.extend(_capture_gaps(lines, covered, rel_path, language, config))
    chunks.sort(key=lambda c: (c.start_line, c.end_line))
    return chunks


def _make_chunk(
    rel_path: str,
    start_line: int,
    end_line: int,
    text: str,
    language: str,
    symbol: str | None,
    chunk_type: str,
) -> CodeChunk:
    return CodeChunk(
        file_path=rel_path,
        start_line=start_line,
        end_line=end_line,
        content=text,
        language=language,
        symbol=symbol,
        symbol_path=symbol or "",
        chunk_type=chunk_type,
        content_hash=content_hash(text),
    )


def _collect_definitions(root: Any, node_types: set[str]) -> list[Any]:
    """Collect the OUTERMOST definition nodes anywhere in the tree.

    Descends through non-definition wrappers (export_statement, if/try blocks) so a
    top-level def nested inside a block is still found, but does NOT recurse into a
    captured definition — a class's methods stay part of the class chunk (no overlap).
    """
    result: list[Any] = []

    def visit(node: Any) -> None:
        for child in getattr(node, "children", []):
            if child.type in node_types:
                result.append(child)
            else:
                visit(child)

    visit(root)
    return result


def _extract_symbol(node: Any) -> str | None:
    """Extract the name of a definition node using the tree-sitter field API."""
    name = _name_of(node)
    if name:
        return name
    # Wrappers (decorated_definition, export_statement) hold the real def as a child.
    for child in getattr(node, "children", []):
        name = _name_of(child)
        if name:
            return name
    return None


def _name_of(node: Any) -> str | None:
    """Resolve a node's name via the `name` field, or the `declarator` chain for C/C++."""
    name_node = _field(node, "name")
    if name_node is not None:
        return str(name_node.text.decode("utf-8", errors="replace"))

    decl = _field(node, "declarator")
    for _ in range(8):  # bounded declarator recursion
        if decl is None:
            break
        if decl.type in ("identifier", "field_identifier", "type_identifier"):
            return str(decl.text.decode("utf-8", errors="replace"))
        nxt = _field(decl, "declarator")
        if nxt is None:
            for ch in getattr(decl, "children", []):
                if ch.type in ("identifier", "field_identifier", "type_identifier"):
                    return str(ch.text.decode("utf-8", errors="replace"))
            break
        decl = nxt
    return None


def _field(node: Any, field: str) -> Any:
    """child_by_field_name that tolerates nodes/grammars without the field."""
    try:
        return node.child_by_field_name(field)
    except (AttributeError, TypeError):
        return None


def _capture_gaps(
    lines: list[str],
    covered: list[tuple[int, int]],
    rel_path: str,
    language: str,
    config: IndexSettings,
) -> list[CodeChunk]:
    """Chunk contiguous line ranges not covered by any definition chunk."""
    covered_lines: set[int] = set()
    for start, end in covered:
        covered_lines.update(range(start, end + 1))

    char_budget = config.chunk_size * CHARS_PER_TOKEN
    overlap_chars = config.chunk_overlap * CHARS_PER_TOKEN
    chunks: list[CodeChunk] = []

    def flush(gap_start: int, gap_end: int) -> None:
        gap_lines = lines[gap_start - 1 : gap_end]
        if not "\n".join(gap_lines).strip():
            return
        for ws, we in _compute_line_windows(gap_lines, char_budget, overlap_chars):
            seg = "\n".join(gap_lines[ws:we])
            if seg.strip():
                chunks.append(
                    _make_chunk(
                        rel_path, gap_start + ws, gap_start + we - 1, seg, language, None, "module"
                    )
                )

    run_start: int | None = None
    for ln in range(1, len(lines) + 1):
        if ln not in covered_lines:
            if run_start is None:
                run_start = ln
        elif run_start is not None:
            flush(run_start, ln - 1)
            run_start = None
    if run_start is not None:
        flush(run_start, len(lines))

    return chunks


def _compute_line_windows(
    lines: list[str],
    char_budget: int,
    overlap_chars: int,
) -> list[tuple[int, int]]:
    """Compute overlapping (start, end) line windows for chunking.

    Returns a list of (start, end) tuples where lines[start:end] is one window.
    """
    windows: list[tuple[int, int]] = []
    start = 0
    while start < len(lines):
        end = start
        total_chars = 0
        while end < len(lines) and total_chars < char_budget:
            total_chars += len(lines[end]) + 1
            end += 1

        windows.append((start, end))

        if end >= len(lines):
            break

        overlap_lines = 0
        overlap_total = 0
        for i in range(end - 1, start, -1):
            overlap_total += len(lines[i]) + 1
            overlap_lines += 1
            if overlap_total >= overlap_chars:
                break

        new_start = end - overlap_lines
        if new_start <= start:
            break
        start = new_start

    return windows


def _split_large_content(
    content: str,
    base_start_line: int,
    rel_path: str,
    language: str,
    symbol: str | None,
    chunk_type: str,
    config: IndexSettings,
) -> list[CodeChunk]:
    """Split a large chunk into overlapping sub-chunks."""
    lines = content.split("\n")
    char_budget = config.chunk_size * CHARS_PER_TOKEN
    overlap_chars = config.chunk_overlap * CHARS_PER_TOKEN
    chunks: list[CodeChunk] = []

    for start, end in _compute_line_windows(lines, char_budget, overlap_chars):
        chunk_content = "\n".join(lines[start:end])
        chunks.append(
            _make_chunk(
                rel_path,
                base_start_line + start,
                base_start_line + end - 1,
                chunk_content,
                language,
                symbol,
                chunk_type,
            )
        )

    return chunks


def _chunk_by_lines(
    content: str,
    rel_path: str,
    language: str,
    config: IndexSettings,
) -> list[CodeChunk]:
    """Fallback: chunk file by line windows (files with no tree-sitter grammar)."""
    lines = content.split("\n")
    char_budget = config.chunk_size * CHARS_PER_TOKEN
    overlap_chars = config.chunk_overlap * CHARS_PER_TOKEN
    chunks: list[CodeChunk] = []

    for start, end in _compute_line_windows(lines, char_budget, overlap_chars):
        chunk_content = "\n".join(lines[start:end])
        if chunk_content.strip():
            chunks.append(
                _make_chunk(rel_path, start + 1, end, chunk_content, language, None, "module")
            )

    return chunks
