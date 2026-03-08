# Mita Code — Development Conventions

## What This Project Is

Mita Code (`mita`) is a local-first, terminal-native agentic coding assistant.
It runs LLMs entirely on the user's machine via Ollama — no API keys, no cloud.
Think "Claude Code but 100% local."

## Architecture at a Glance

- **Language**: Python 3.11+ with async (`asyncio`)
- **Package layout**: `src/mita/` (src layout)
- **CLI**: Typer. Entry point: `mita = "mita.cli:app"`
- **LLM calls**: LiteLLM (for chat completions) + `ollama` client (for model management)
- **Structured output**: Instructor in JSON mode (widest model compatibility)
- **Config**: Layered TOML — global (`~/.config/mita/config.toml`) → project (`.mita/settings.toml`)
- **Memory**: Layered Markdown — global (`~/.config/mita/MITA.md`) → project → directory (`MITA.md`)
- **Plugins**: MCP protocol (stdio + SSE transport)
- **Vector store**: LanceDB (local, embedded)
- **Code parsing**: Tree-sitter
- **UI**: Rich + prompt_toolkit

## Key Design Decisions

1. **Instructor JSON mode is the primary tool-call path** — not native function calling.
   This gives the widest local model compatibility. See PLANNING.md Q4.

2. **One tool call per file** for edits — no batch multi-file tool. Simpler schema = better
   reliability on local models.

3. **Config merge**: deep merge at leaf level. Lists (hooks, plugins, skills_paths)
   are appended, not replaced.

4. **LiteLLM for chat, `ollama` for model management** — LiteLLM abstracts the completion
   API; `ollama` client handles pull/list/delete which LiteLLM doesn't cover.

5. **Async architecture** — the agent loop, MCP lifecycle, and streaming are all async.

6. **Index stored in `.mita/index/`** (gitignored) — not in a global cache.

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
- `plugins` depends on `config`, `tools`
- `hooks` depends on `config`
- `agent` depends on everything (it's the orchestrator)
- `ui` depends on `config`
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
