# Codebase Indexing

Mita can index your codebase for retrieval-augmented generation (RAG). When enabled, the agent automatically retrieves relevant code snippets to provide better, context-aware responses.

## How It Works

1. **Parse** — Tree-sitter extracts semantic chunks (functions, classes) from your code
2. **Embed** — Ollama generates vector embeddings using `nomic-embed-text`
3. **Store** — Chunks are stored in a local LanceDB database (`.mita/index/`)
4. **Retrieve** — When you ask a question, relevant chunks are injected into the LLM context

## Build the Index

```bash
mita index build
```

If the embedding model isn't installed, Mita will prompt you to pull it.

Use `--force` to rebuild from scratch:

```bash
mita index build --force
```

## Search

```bash
mita index search "database connection pooling"
mita index search "authentication" --top-k 5
```

## Status

```bash
mita index status
```

Shows chunk count, index size, and last build time.

## Clear

```bash
mita index clear
```

Deletes the index. Rebuild with `mita index build`.

## Configuration

```toml
[index]
enabled = true
chunk_size = 512
chunk_overlap = 64
top_k = 10
exclude_patterns = [
    "*.lock",
    ".mita/**",
    "node_modules/**",
    ".git/**",
    "*.min.js",
    "*.min.css",
    "dist/**",
    "build/**",
    "__pycache__/**",
]
```

## Supported Languages

Tree-sitter parsing supports semantic chunking for: Python, JavaScript, TypeScript, Rust, Go, Java, C, C++, Ruby, PHP, and more. Files in unsupported languages fall back to line-based chunking.
