# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Session persistence**: chat sessions auto-save to `.mita/sessions/` after each turn.
  Resume with `mita chat --resume <id>` (or `--continue` for the most recent), or with
  the `/resume` command in the REPL and the TUI. Manage saved sessions with
  `mita sessions list` and `mita sessions clear`. Sessions older than
  `sessions.max_age_days` (default 7) are pruned at chat start; configure via the new
  `[sessions]` config section (`autosave`, `max_age_days`).

## [0.1.0] - 2026-03-09

### Added

- **CLI commands**: `mita chat`, `mita ask`, `mita doctor`, `mita config`, `mita memory`,
  `mita models`, `mita index`, `mita skills`, `mita plugins`, `mita hooks`, `mita ollama`.
- **Layered TOML configuration**: global (`~/.config/mita/config.toml`) and project
  (`.mita/settings.toml`) configs with deep merge at leaf level; lists are appended.
- **Layered MITA.md memory**: walk-up-tree discovery of `MITA.md` files from directory
  to project root to global, with CRUD operations.
- **Ollama integration**: model pull, list, delete, and hardware-aware model
  recommendations based on available VRAM and system RAM.
- **LLM client**: chat completions via LiteLLM with streaming support; structured
  tool-call output via Instructor in JSON mode for broad local model compatibility.
- **Built-in tools**: `file_read`, `file_write`, `file_edit`, `shell`, `glob`, `grep`
  with one-tool-per-file edit design for reliable local model usage.
- **Agentic tool loop**: async agent loop that parses model responses, dispatches tool
  calls, feeds results back, and streams output to the terminal.
- **Codebase indexing**: Tree-sitter code parsing with LanceDB vector storage in
  `.mita/index/`; supports incremental re-indexing and RAG-assisted context retrieval.
- **Skills system**: Markdown skill files with YAML frontmatter; discoverable from
  project and global skills paths; injected into agent context on demand.
- **MCP plugin system**: stdio and SSE transports; plugins registered via CLI or TOML
  config; tools from plugins are available to the agent during sessions.
- **Hooks system**: shell commands fired on lifecycle events (`session_start`,
  `session_end`, `pre_tool_call`, `post_tool_call`, `on_file_write`) with glob-based
  match patterns and context variable substitution.
- **Non-interactive mode**: stdin piping support, `--output text/json` flag, and
  `--no-tools` flag for scripting and CI usage.
- **Signal handling**: graceful Ctrl-C handling to interrupt generation or exit cleanly.
- **MkDocs documentation site**: project documentation built with MkDocs for
  ReadTheDocs hosting.
