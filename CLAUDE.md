# Mita Code — Development Conventions

## What This Project Is

Mita Code (`mita`) is a local-first, terminal-native agentic coding assistant.
It runs LLMs entirely on the user's machine — no cloud. Ollama is the default
backend; any local OpenAI-compatible server (llama.cpp `llama-server`, vLLM,
LM Studio, TGI) is also supported via the `[llm]` config section.
Think "Claude Code but 100% local."

## Architecture at a Glance

- **Language**: Python 3.11+ with async (`asyncio`)
- **Package layout**: `src/mita/` (src layout)
- **CLI**: Typer. Entry point: `mita = "mita.cli:app"`
- **LLM calls**: LiteLLM (chat + embeddings), routed per provider via `llm/providers.py`;
  `ollama` client only for Ollama model management (pull/list/delete)
- **Tool calls**: native OpenAI-style function calling, with a text-JSON fallback for
  models/backends that don't support it (`agent/loop.py`)
- **Config**: Layered TOML — global (`~/.config/mita/config.toml`) → project (`.mita/settings.toml`)
- **Memory**: Layered Markdown — global (`~/.config/mita/MITA.md`) → project → directory (`MITA.md`)
- **Plugins**: MCP protocol (stdio + SSE transport)
- **Vector store**: LanceDB (local, embedded)
- **Code parsing**: Tree-sitter
- **UI**: Rich + prompt_toolkit (REPL); Textual (`mita chat --tui`). The agent loop
  renders through a `UISink` protocol (`ui/sink.py`), so both frontends share it.

## Key Design Decisions

1. **Native function calling is the primary tool-call path**, with a text-JSON fallback
   for backends without it. (Instructor JSON mode was removed — it was unused dead code.)

2. **One tool call per file** for edits — no batch multi-file tool. Simpler schema = better
   reliability on local models.

3. **Config merge**: deep merge at leaf level. Lists (hooks, plugins, skills_paths)
   are appended, not replaced.

4. **Multi-backend via a provider registry** (`llm/providers.py`) — one place maps each
   provider to its LiteLLM prefix, base URL, api_key rule, and context-probe strategy.
   Ollama-specific options (the runtime `options` blob, `pull`/`list`) are gated behind
   `provider == ollama`. No `[llm]` section defaults to Ollama, unchanged for existing users.

5. **Async architecture** — the agent loop, MCP lifecycle, and streaming are all async.

6. **Index stored in `.mita/index/`** (gitignored) — not in a global cache.

7. **Chat sessions stored in `.mita/sessions/`** (gitignored) — one JSON file per session,
   non-system messages only; the system prompt is regenerated on resume.

## Coding Standards

- **Package manager**: Poetry
- **Formatting**: Ruff (format + lint)
- **Linting**: Ruff + pylint
- **Type checking**: mypy (strict)
- **Tests**: pytest + pytest-asyncio
- **Task runner**: Invoke (`tasks.py`) — run `invoke test`, `invoke lint`, `invoke check`
- **Models**: Pydantic v2 for all data models
- **Imports**: Use absolute imports (`from mita.config.schema import MitaConfig`)
- **No wildcard imports**
- **Docstrings**: Only on public API functions/classes. Keep them short.
- **Error handling**: Raise specific exceptions. Don't catch broad `Exception` unless at the top-level agent loop.

## Module Dependency Rules

- `config` depends on nothing internal
- `memory` depends on `config`
- `models` depends on `config`
- `llm` depends on `config`
- `tools` depends on `config`, `ui`
- `index` depends on `config`, `llm`
- `skills` depends on `config`
- `sessions` depends on `config` (plus the `agent` conversation/context modules; `agent` must never import `sessions`)
- `plugins` depends on `config`, `tools`
- `hooks` depends on `config`
- `agent` depends on everything (it's the orchestrator)
- `ui` depends on `config` (the TUI frontend additionally uses `agent` conversation models, `tools` schemas, and `sessions`)
- `cli` depends on everything (it's the entry point)

**No circular dependencies.** If module A imports from module B, module B must not import from A.

## File Naming

- Module files: `snake_case.py`
- Test files: `test_<module>.py` mirroring `src/` structure
- Config files: TOML
- Memory files: `MITA.md`
- Skill files: `<name>.md` with YAML frontmatter

## Common Patterns

### Adding a new built-in tool
1. Create `src/mita/tools/builtins/<tool_name>.py`
2. Implement the handler function with signature `async def execute(args: dict) -> ToolResult`
3. Define a `TOOL_DEF: ToolDefinition` constant in the module
4. Register in `src/mita/tools/builtins/__init__.py`

### Adding a CLI command
1. Add a new Typer sub-app or command in `src/mita/cli.py` (or a dedicated `manager.py` in the relevant subpackage)
2. Keep CLI handlers thin — delegate to module functions

### Running the project locally during development
```bash
poetry install
poetry run mita --version
poetry run mita doctor
```

## Important File Paths

- `PLANNING.md` — Full project plan with all architectural decisions
- `src/mita/config/schema.py` — All Pydantic config models
- `src/mita/tools/schema.py` — Tool definition and result models
- `src/mita/agent/loop.py` — Core agent loop
- `src/mita/cli.py` — CLI command definitions
