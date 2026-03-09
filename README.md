# Mita Code

[![CI](https://github.com/jtdub/mita-code/actions/workflows/ci.yml/badge.svg)](https://github.com/jtdub/mita-code/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/mita-code)](https://pypi.org/project/mita-code/)
[![Python](https://img.shields.io/pypi/pyversions/mita-code)](https://pypi.org/project/mita-code/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Docs](https://readthedocs.org/projects/mita-code/badge/?version=latest)](https://mita-code.readthedocs.io/)

A local-first, terminal-native agentic coding assistant that runs LLMs entirely on your machine via [Ollama](https://ollama.com). No API keys. No cloud. No telemetry.

## Features

- **100% Local** — All inference runs on your hardware via Ollama. Your code never leaves your machine.
- **Agentic Tool Loop** — Read/write files, run shell commands, git operations — with confirmation for destructive actions.
- **Hardware-Aware Model Recommendations** — Detects your RAM, VRAM, and GPU to recommend models that will actually run well.
- **MCP Plugin System** — Compatible with the existing [Model Context Protocol](https://modelcontextprotocol.io/) ecosystem (stdio and SSE transport).
- **Skills** — Reusable, parameterized prompt templates stored as Markdown files (e.g., `/commit`, `/review`).
- **Layered Memory** — `MITA.md` files at global, project, and directory scope are automatically injected into context.
- **Layered Config** — TOML configuration cascades from global to project level.
- **Hooks** — Lifecycle shell commands that fire on events like file writes or tool calls.
- **Codebase Indexing** — Local vector search (LanceDB + Tree-sitter) for RAG over your codebase.
- **Unix Philosophy** — Composable, pipeable, scriptable.

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) installed and running

## Installation

```bash
pipx install mita-code
```

Or for development:

```bash
git clone https://github.com/jtdub/mita-code.git
cd mita-code
poetry install
```

## Quick Start

```bash
# Start Ollama (if not already running)
ollama serve

# Pull a coding model
mita models pull qwen2.5-coder:7b

# Start an interactive session
mita chat

# Or ask a single question
mita ask "explain the auth module in this project"
```

## Usage

### Interactive Chat

```bash
mita chat                           # Start agentic chat session
mita chat --model deepseek-coder-v2:16b  # Use a specific model
mita chat --no-tools                # Pure chat, no tool execution
```

### Single-Shot Prompts

```bash
mita ask "refactor this function to use async"
cat error.log | mita ask "what went wrong?"
```

### Model Management

```bash
mita models recommend               # See what fits your hardware
mita models list                     # List installed models
mita models pull qwen2.5-coder:14b  # Pull a model
mita models default qwen2.5-coder:14b  # Set as default
```

### Memory

```bash
mita memory show                     # View all active memory
mita memory add "Always use pytest" --project  # Add project-level memory
mita memory edit                     # Edit nearest MITA.md
```

### Codebase Indexing

```bash
mita index build                     # Index the current project
mita index search "database connection"  # Search the index
```

### Skills

```bash
mita skills list                     # List available skills
# In chat, use /skill_name to invoke:
# mita> /commit
# mita> /review
```

### Plugins (MCP)

```bash
mita plugins add filesystem --command "npx @modelcontextprotocol/server-filesystem ."
mita plugins list                    # List plugins and their tools
```

### Configuration

```bash
mita config show                     # Show merged configuration
mita config edit --global            # Edit global config
mita config set model.default "qwen2.5-coder:14b"
```

### Diagnostics

```bash
mita doctor                          # Check Ollama, models, config health
```

## Configuration

Global config lives at `~/.config/mita/config.toml`. Project-level overrides go in `.mita/settings.toml`.

```toml
[model]
default = "qwen2.5-coder:7b"
temperature = 0.1

[tools]
auto_approve = ["file_read", "glob", "grep"]
confirm_destructive = true

[index]
enabled = true
top_k = 10
```

See [PLANNING.md](PLANNING.md) for the full configuration schema.

## Memory System

Mita uses layered `MITA.md` files that are automatically discovered and injected into context:

| Scope | Location | Purpose |
|---|---|---|
| Global | `~/.config/mita/MITA.md` | Preferences across all projects |
| Project | `<project_root>/MITA.md` | Project-specific conventions |
| Directory | `<subdir>/MITA.md` | Directory-specific context |

Higher-specificity files take priority. Each file is capped at 200 lines.

## Tech Stack

| Component | Library |
|---|---|
| CLI | [Typer](https://typer.tiangolo.com/) |
| Terminal UI | [Rich](https://rich.readthedocs.io/) |
| LLM Runtime | [Ollama](https://ollama.com) |
| LLM Client | [LiteLLM](https://github.com/BerriAI/litellm) |
| Structured Output | [Instructor](https://github.com/instructor-ai/instructor) |
| Vector Store | [LanceDB](https://lancedb.com/) |
| Code Parsing | [Tree-sitter](https://tree-sitter.github.io/) |
| Config | TOML (stdlib `tomllib`) |
| Plugins | [MCP](https://modelcontextprotocol.io/) |

## Contributing

See [PLANNING.md](PLANNING.md) for the full project plan, architecture, and build phases.

Full documentation is available at [mita-code.readthedocs.io](https://mita-code.readthedocs.io/).

## License

Apache 2.0 — see [LICENSE](LICENSE).
