# Mita Code

**Local-first, terminal-native agentic coding assistant powered by Ollama.**

No API keys. No cloud. No telemetry. Your code never leaves your machine.

---

## What is Mita Code?

Mita Code (`mita`) is a CLI coding assistant that runs large language models entirely on your hardware via [Ollama](https://ollama.com). It gives you an agentic tool loop — reading files, writing code, running shell commands, searching your codebase — all from the terminal, all locally.

## Key Features

<div class="grid cards" markdown>

- :material-server-network: **100% Local** — All inference runs on your hardware via Ollama.
- :material-tools: **Agentic Tool Loop** — Read/write files, run shell commands, git operations with confirmation for destructive actions.
- :material-memory: **Layered Memory** — `MITA.md` files at global, project, and directory scope are automatically injected into context.
- :material-magnify: **Codebase Indexing** — Local vector search (LanceDB + Tree-sitter) for RAG over your codebase.
- :material-lightning-bolt: **Skills** — Reusable prompt templates stored as Markdown (e.g., `/commit`, `/review`).
- :material-chip: **Hardware-Aware** — Detects your RAM, VRAM, and GPU to recommend models that run well on your machine.
- :material-puzzle: **MCP Plugins** — Compatible with the [Model Context Protocol](https://modelcontextprotocol.io/) ecosystem.
- :material-cog: **Layered Config** — TOML configuration cascades from global to project level.

</div>

## Quick Start

```bash
# Install
pipx install mita-code

# Start Ollama
ollama serve

# Pull a coding model
mita models pull qwen2.5-coder:7b

# Start an interactive session
mita chat
```

See the [Installation](getting-started/installation.md) and [Quick Start](getting-started/quickstart.md) guides for details.

## Tech Stack

| Component | Library |
|---|---|
| CLI | [Typer](https://typer.tiangolo.com/) |
| Terminal UI | [Rich](https://rich.readthedocs.io/) |
| LLM Runtime | [Ollama](https://ollama.com) |
| LLM Client | [LangChain](https://www.langchain.com/) (`ChatOllama` / `ChatOpenAI`) |
| Agent orchestration | [LangGraph](https://www.langchain.com/langgraph) |
| Vector Store | [LanceDB](https://lancedb.com/) |
| Code Parsing | [Tree-sitter](https://tree-sitter.github.io/) |
| Config | TOML (stdlib `tomllib`) |
| Plugins | [MCP](https://modelcontextprotocol.io/) via `langchain-mcp-adapters` |
