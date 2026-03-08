"""Tree-sitter code parsing and chunking for codebase indexing."""

from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from mita.config.schema import IndexSettings
from mita.index.store import CodeChunk

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
    "ruby": {"method", "class", "module"},
}


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
    """Walk the project and yield non-excluded files."""
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = str(path.relative_to(root))
        if _matches_exclude(rel, exclude_patterns):
            continue
        files.append(path)
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
        return []

    tree = parser.parse(content.encode("utf-8"))
    node_types = CHUNK_NODE_TYPES.get(language, set())
    lines = content.split("\n")
    chunks: list[CodeChunk] = []

    for node in _walk_top_level(tree.root_node):
        if node.type not in node_types:
            continue

        start_line = node.start_point[0] + 1  # 1-indexed
        end_line = node.end_point[0] + 1
        node_content = "\n".join(lines[start_line - 1 : end_line])
        symbol = _extract_symbol(node, language)

        # Split large nodes
        char_budget = config.chunk_size * 4  # ~4 chars per token
        if len(node_content) > char_budget:
            sub_chunks = _split_large_content(
                node_content, start_line, rel_path, language, symbol, config
            )
            chunks.extend(sub_chunks)
        else:
            chunks.append(
                CodeChunk(
                    file_path=rel_path,
                    start_line=start_line,
                    end_line=end_line,
                    content=node_content,
                    language=language,
                    symbol=symbol,
                )
            )

    return chunks


def _walk_top_level(node: Any) -> list[Any]:
    """Get top-level children of the root node (non-recursive)."""
    children = getattr(node, "children", [])
    return list(children)


def _extract_symbol(node: Any, language: str) -> str | None:
    """Extract the name of a function/class node."""
    # Most languages store the name in a child node of type "identifier" or "name"
    for child in getattr(node, "children", []):
        child_type = getattr(child, "type", "")
        if child_type in ("identifier", "name", "property_identifier"):
            return getattr(child, "text", b"").decode("utf-8", errors="replace")
    return None


def _split_large_content(
    content: str,
    base_start_line: int,
    rel_path: str,
    language: str,
    symbol: str | None,
    config: IndexSettings,
) -> list[CodeChunk]:
    """Split a large chunk into overlapping sub-chunks."""
    lines = content.split("\n")
    char_budget = config.chunk_size * 4
    overlap_chars = config.chunk_overlap * 4
    chunks: list[CodeChunk] = []

    start = 0
    while start < len(lines):
        # Accumulate lines until we hit the budget
        end = start
        total_chars = 0
        while end < len(lines) and total_chars < char_budget:
            total_chars += len(lines[end]) + 1
            end += 1

        chunk_content = "\n".join(lines[start:end])
        chunks.append(
            CodeChunk(
                file_path=rel_path,
                start_line=base_start_line + start,
                end_line=base_start_line + end - 1,
                content=chunk_content,
                language=language,
                symbol=symbol,
            )
        )

        # If we consumed all lines, we're done
        if end >= len(lines):
            break

        # Advance with overlap
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

    return chunks


def _chunk_by_lines(
    content: str,
    rel_path: str,
    language: str,
    config: IndexSettings,
) -> list[CodeChunk]:
    """Fallback: chunk file by line windows."""
    lines = content.split("\n")
    char_budget = config.chunk_size * 4
    overlap_chars = config.chunk_overlap * 4
    chunks: list[CodeChunk] = []

    start = 0
    while start < len(lines):
        end = start
        total_chars = 0
        while end < len(lines) and total_chars < char_budget:
            total_chars += len(lines[end]) + 1
            end += 1

        chunk_content = "\n".join(lines[start:end])
        if chunk_content.strip():
            chunks.append(
                CodeChunk(
                    file_path=rel_path,
                    start_line=start + 1,
                    end_line=end,
                    content=chunk_content,
                    language=language,
                )
            )

        # If we consumed all lines, we're done
        if end >= len(lines):
            break

        # Advance with overlap
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

    return chunks
