# Mita Code — Project Plan

> Canonical reference for the Mita Code (`mita`) build.
> Local-first, agentic coding assistant powered by Ollama.

---

## 1. Project Structure

```
mita-code/
├── pyproject.toml                  # Package metadata, deps, entry points
├── LICENSE                         # Apache 2.0
├── README.md                       # User-facing docs
├── PLANNING.md                     # This file
├── CLAUDE.md                       # Conventions for AI-assisted development
├── MITA.md                         # Project-level memory for mita itself
│
├── src/
│   └── mita/
│       ├── __init__.py             # Package version, top-level exports
│       ├── __main__.py             # `python -m mita` entry point
│       ├── cli.py                  # Typer app: top-level command routing
│       │
│       ├── config/
│       │   ├── __init__.py
│       │   ├── schema.py           # Pydantic models for config/settings
│       │   ├── loader.py           # TOML loading with global→project layering
│       │   └── defaults.py         # Default config values
│       │
│       ├── memory/
│       │   ├── __init__.py
│       │   ├── discovery.py        # Walk-up-tree MITA.md file discovery
│       │   ├── loader.py           # Read, merge, and truncate memory files
│       │   └── manager.py          # CLI commands for memory CRUD
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   ├── hardware.py         # Detect RAM, VRAM, CPU, GPU
│       │   ├── registry.py         # Curated model list + metadata
│       │   ├── recommender.py      # Filter models by hardware capability
│       │   ├── ollama_client.py    # Ollama pull/list/delete/status wrappers
│       │   └── manager.py          # CLI commands for model management
│       │
│       ├── llm/
│       │   ├── __init__.py
│       │   ├── client.py           # LiteLLM client setup (Ollama backend)
│       │   ├── instructor.py       # Instructor wrapper for structured output
│       │   └── streaming.py        # Streaming token handler for Rich
│       │
│       ├── tools/
│       │   ├── __init__.py
│       │   ├── registry.py         # Tool registry: name→handler mapping
│       │   ├── schema.py           # Tool definition & result Pydantic models
│       │   ├── executor.py         # Tool dispatch, confirmation, execution
│       │   ├── builtins/
│       │   │   ├── __init__.py
│       │   │   ├── file_read.py    # Read file contents
│       │   │   ├── file_write.py   # Write/edit files with diff display
│       │   │   ├── file_edit.py    # String-replace edit tool
│       │   │   ├── glob_tool.py    # Glob-based file search
│       │   │   ├── grep_tool.py    # Ripgrep-style content search
│       │   │   ├── shell.py        # Shell command execution
│       │   │   └── git.py          # Git operations (status, diff, commit)
│       │   └── safety.py           # Destructive action detection & confirmation
│       │
│       ├── agent/
│       │   ├── __init__.py
│       │   ├── loop.py             # Core agent loop: prompt→tool→loop
│       │   ├── context.py          # Context assembly (memory + files + RAG)
│       │   ├── conversation.py     # Message history management
│       │   └── system_prompt.py    # System prompt template construction
│       │
│       ├── index/
│       │   ├── __init__.py
│       │   ├── parser.py           # Tree-sitter code parsing (symbols, chunks)
│       │   ├── embeddings.py       # Embedding generation via Ollama
│       │   ├── store.py            # LanceDB vector store operations
│       │   └── retriever.py        # RAG retrieval: query→relevant chunks
│       │
│       ├── skills/
│       │   ├── __init__.py
│       │   ├── loader.py           # Discover and parse skill Markdown files
│       │   ├── executor.py         # Render skill template → prompt
│       │   └── manager.py          # CLI commands for skill CRUD
│       │
│       ├── plugins/
│       │   ├── __init__.py
│       │   ├── mcp_client.py       # MCP client: spawn/connect, list tools
│       │   ├── mcp_transport.py    # stdio and SSE transport handling
│       │   ├── lifecycle.py        # Start/stop/health-check MCP servers
│       │   └── manager.py          # CLI commands for plugin management
│       │
│       ├── hooks/
│       │   ├── __init__.py
│       │   ├── schema.py           # Hook definition models
│       │   ├── runner.py           # Execute hooks at lifecycle points
│       │   └── manager.py          # CLI commands for hook management
│       │
│       └── ui/
│           ├── __init__.py
│           ├── repl.py             # Interactive REPL input loop
│           ├── display.py          # Rich rendering: markdown, diffs, panels
│           ├── spinner.py          # Thinking/loading indicators
│           └── theme.py            # Color scheme and style constants
│
├── tests/
│   ├── conftest.py                 # Shared fixtures
│   ├── test_config/
│   ├── test_memory/
│   ├── test_models/
│   ├── test_llm/
│   ├── test_tools/
│   ├── test_agent/
│   ├── test_index/
│   ├── test_skills/
│   ├── test_plugins/
│   ├── test_hooks/
│   └── test_ui/
│
├── skills/                         # Built-in skill templates
│   ├── commit.md                   # Git commit skill
│   ├── review.md                   # Code review skill
│   └── explain.md                  # Explain code skill
│
└── docs/
    ├── getting-started.md
    ├── configuration.md
    ├── skills.md
    ├── plugins.md
    └── hooks.md
```

---

## 2. Module Responsibility Map

### `mita.cli`
- **Responsibility**: Define all Typer commands and route to submodules
- **Public interface**: `app` (Typer instance)
- **Internal deps**: all `*.manager` modules, `agent.loop`, `ui.repl`
- **External deps**: `typer`, `rich`

### `mita.config`
- **Responsibility**: Load, merge, and validate layered TOML configuration
- **Public interface**: `load_config() → MitaConfig`, `MitaConfig`, `ProjectSettings`
- **Internal deps**: none
- **External deps**: `pydantic`, `tomli`/`tomllib`

### `mita.memory`
- **Responsibility**: Discover, load, and manage MITA.md memory files
- **Public interface**: `discover_memory_files(cwd) → list[Path]`, `load_memory(cwd) → str`, `MemoryManager`
- **Internal deps**: `config` (for size limits)
- **External deps**: none (stdlib only)

### `mita.models`
- **Responsibility**: Hardware detection, model registry, recommendations, Ollama model management
- **Public interface**: `detect_hardware() → HardwareInfo`, `recommend_models(hw) → list[ModelRec]`, `OllamaClient`
- **Internal deps**: `config`
- **External deps**: `ollama`, `psutil`

### `mita.llm`
- **Responsibility**: LLM client abstraction via LiteLLM + Instructor
- **Public interface**: `get_client(config) → LLMClient`, `chat(messages, tools) → Response`, `stream_chat(...)`
- **Internal deps**: `config`
- **External deps**: `litellm`, `instructor`

### `mita.tools`
- **Responsibility**: Define, register, and execute tools; handle safety confirmation
- **Public interface**: `ToolRegistry`, `execute_tool(call) → ToolResult`, `BUILTIN_TOOLS`
- **Internal deps**: `tools.safety`, `ui.display`
- **External deps**: `gitpython`, `subprocess`

### `mita.agent`
- **Responsibility**: Core agent loop — assemble context, call LLM, parse tool calls, execute, repeat
- **Public interface**: `run_agent(prompt, config) → None`, `AgentLoop`
- **Internal deps**: `llm`, `tools`, `memory`, `index`, `plugins`, `skills`, `ui`
- **External deps**: none (coordinates other modules)

### `mita.index`
- **Responsibility**: Codebase indexing — parse code, generate embeddings, store/retrieve vectors
- **Public interface**: `index_project(path)`, `search(query, k) → list[CodeChunk]`
- **Internal deps**: `llm` (for embeddings), `config`
- **External deps**: `tree_sitter`, `lancedb`

### `mita.skills`
- **Responsibility**: Discover, parse, and render skill Markdown templates
- **Public interface**: `load_skill(name) → Skill`, `render_skill(skill, args) → str`, `list_skills()`
- **Internal deps**: `config`, `memory` (for skill search paths)
- **External deps**: none (stdlib only)

### `mita.plugins`
- **Responsibility**: MCP client lifecycle — spawn stdio servers, connect SSE, list/call tools
- **Public interface**: `MCPClient`, `PluginManager`, `list_plugin_tools() → list[ToolDef]`
- **Internal deps**: `config`, `tools.schema`
- **External deps**: `mcp` (MCP SDK)

### `mita.hooks`
- **Responsibility**: Run lifecycle shell hooks at defined trigger points
- **Public interface**: `run_hooks(event, context)`, `HookRunner`
- **Internal deps**: `config`
- **External deps**: `subprocess`

### `mita.ui`
- **Responsibility**: Rich terminal rendering — REPL, streaming output, diffs, spinners
- **Public interface**: `REPL`, `display_markdown()`, `display_diff()`, `Spinner`
- **Internal deps**: `config` (for theme prefs)
- **External deps**: `rich`, `prompt_toolkit`

---

## 3. Core Data Models

```python
# ── config/schema.py ──────────────────────────────────────────────

from pydantic import BaseModel, Field
from typing import Optional
from pathlib import Path

class OllamaSettings(BaseModel):
    host: str = "http://localhost:11434"
    timeout: int = 120  # seconds

class ModelSettings(BaseModel):
    default: str = "qwen2.5-coder:7b"
    embedding: str = "nomic-embed-text"
    temperature: float = 0.1
    max_tokens: int = 4096
    context_window: int = 32768

class ToolSettings(BaseModel):
    auto_approve: list[str] = Field(default_factory=lambda: ["file_read", "glob", "grep"])
    confirm_destructive: bool = True
    shell_timeout: int = 120  # seconds
    banned_commands: list[str] = Field(default_factory=lambda: ["rm -rf /", "mkfs", "dd"])

class MemorySettings(BaseModel):
    max_lines_per_file: int = 200
    max_total_tokens: int = 4000

class IndexSettings(BaseModel):
    enabled: bool = True
    chunk_size: int = 512  # tokens
    chunk_overlap: int = 64
    top_k: int = 10
    exclude_patterns: list[str] = Field(
        default_factory=lambda: ["*.lock", "node_modules/**", ".git/**", "*.min.js"]
    )

class HookDefinition(BaseModel):
    event: str  # "pre_tool_call", "post_tool_call", "on_file_write", "session_start", "session_end"
    command: str
    match: Optional[str] = None  # optional tool name filter

class PluginDefinition(BaseModel):
    name: str
    transport: str = "stdio"  # "stdio" | "sse"
    command: Optional[str] = None  # for stdio
    args: list[str] = Field(default_factory=list)
    url: Optional[str] = None  # for sse
    env: dict[str, str] = Field(default_factory=dict)

class UISettings(BaseModel):
    theme: str = "auto"  # "auto", "dark", "light"
    show_token_count: bool = True
    stream: bool = True
    markdown: bool = True

class MitaConfig(BaseModel):
    """Root configuration model — result of merging global + project TOML."""
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    tools: ToolSettings = Field(default_factory=ToolSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    index: IndexSettings = Field(default_factory=IndexSettings)
    ui: UISettings = Field(default_factory=UISettings)
    hooks: list[HookDefinition] = Field(default_factory=list)
    plugins: list[PluginDefinition] = Field(default_factory=list)
    skills_paths: list[str] = Field(default_factory=lambda: ["~/.config/mita/skills", ".mita/skills"])


# ── tools/schema.py ───────────────────────────────────────────────

from pydantic import BaseModel, Field
from typing import Any, Optional
from enum import Enum

class ToolParameter(BaseModel):
    name: str
    type: str  # "string", "integer", "boolean", "array", "object"
    description: str
    required: bool = True
    default: Optional[Any] = None

class ToolDefinition(BaseModel):
    """Schema sent to the LLM so it knows what tools are available."""
    name: str
    description: str
    parameters: list[ToolParameter]
    destructive: bool = False  # triggers confirmation flow
    source: str = "builtin"  # "builtin" | "mcp:<plugin_name>"

class ToolCall(BaseModel):
    """Parsed from LLM output."""
    id: str
    name: str
    arguments: dict[str, Any]

class ToolResult(BaseModel):
    """Returned after executing a tool."""
    tool_call_id: str
    success: bool
    output: str = ""
    error: Optional[str] = None
    truncated: bool = False


# ── models/hardware.py ────────────────────────────────────────────

from pydantic import BaseModel
from typing import Optional

class GPUInfo(BaseModel):
    name: str
    vram_gb: float
    vendor: str  # "nvidia", "amd", "apple"

class HardwareInfo(BaseModel):
    ram_gb: float
    cpu_cores: int
    cpu_name: str
    gpus: list[GPUInfo]
    os: str  # "darwin", "linux", "windows"
    apple_silicon: bool = False
    unified_memory: bool = False  # Apple Silicon uses unified RAM as VRAM


# ── models/registry.py ───────────────────────────────────────────

class ModelCard(BaseModel):
    """Metadata for a coding-focused model in the curated registry."""
    name: str  # Ollama model tag, e.g. "qwen2.5-coder:7b"
    family: str  # "qwen", "codellama", "deepseek", etc.
    param_count: str  # "7B", "14B", "32B"
    min_ram_gb: float
    min_vram_gb: float
    context_window: int
    quantization: str = "Q4_K_M"
    tool_call_support: bool = True
    description: str = ""
    tags: list[str] = Field(default_factory=list)  # ["code", "chat", "instruct"]

class ModelRecommendation(BaseModel):
    model: ModelCard
    fit_score: float  # 0.0–1.0, how well it fits the hardware
    notes: str  # e.g. "Will use ~6GB of your 16GB RAM"


# ── agent/conversation.py ────────────────────────────────────────

from enum import Enum

class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

class Message(BaseModel):
    role: Role
    content: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: Optional[str] = None  # for tool-result messages
    name: Optional[str] = None  # tool name for tool-result messages

class Conversation(BaseModel):
    messages: list[Message] = Field(default_factory=list)
    total_tokens: int = 0

    def add(self, msg: Message) -> None: ...
    def get_messages_for_api(self) -> list[dict]: ...
    def truncate_to_fit(self, max_tokens: int) -> None: ...


# ── skills/loader.py ──────────────────────────────────────────────

class SkillFrontmatter(BaseModel):
    name: str
    description: str
    args: list[ToolParameter] = Field(default_factory=list)
    trigger: Optional[str] = None  # regex or keyword trigger

class Skill(BaseModel):
    frontmatter: SkillFrontmatter
    template: str  # Markdown body with {{arg}} placeholders
    source_path: Path


# ── index/store.py ────────────────────────────────────────────────

class CodeChunk(BaseModel):
    file_path: str
    start_line: int
    end_line: int
    content: str
    language: str
    symbol: Optional[str] = None  # function/class name if applicable
    embedding: Optional[list[float]] = None

class SearchResult(BaseModel):
    chunk: CodeChunk
    score: float
```

---

## 4. Build Order & Milestones

### Phase 1 — Scaffold & Config (Week 1)
**Goal**: `pip install -e .` works, `mita --version` prints version, config loads.

- [ ] `pyproject.toml` with entry point `mita = "mita.cli:app"`
- [ ] `src/mita/__init__.py` with `__version__`
- [ ] `mita.config` — TOML loading, global→project merge, `MitaConfig` model
- [ ] `mita.memory` — MITA.md discovery and loading
- [ ] CLI commands: `mita config show`, `mita memory show`, `mita memory edit`
- [ ] Unit tests for config merging and memory discovery

**Testable**: `mita config show` prints merged config. `mita memory show` prints discovered MITA.md content.

### Phase 2 — Ollama & Hardware (Week 2)
**Goal**: Detect hardware, list/pull/recommend models.

- [ ] `mita.models.hardware` — RAM, VRAM, CPU, GPU detection (psutil + platform-specific)
- [ ] `mita.models.registry` — Curated `ModelCard` list for coding models
- [ ] `mita.models.recommender` — Filter and rank models by hardware
- [ ] `mita.models.ollama_client` — Wrap `ollama` Python client
- [ ] CLI commands: `mita models list`, `mita models pull <name>`, `mita models recommend`, `mita models info <name>`, `mita models remove <name>`

**Testable**: `mita models recommend` shows models that fit the user's machine.

### Phase 3 — LLM Client & Built-in Tools (Week 3)
**Goal**: Can send a prompt to a local model and get a response. Tools are defined and executable.

- [ ] `mita.llm.client` — LiteLLM client pointing at Ollama
- [ ] `mita.llm.instructor` — Instructor wrapper for structured tool calls
- [ ] `mita.llm.streaming` — Token-by-token streaming to terminal
- [ ] `mita.tools.schema` — Tool definition models
- [ ] `mita.tools.registry` — Tool registry
- [ ] `mita.tools.builtins` — All 7 built-in tools
- [ ] `mita.tools.safety` — Destructive action detection
- [ ] `mita.tools.executor` — Dispatch and execute

**Testable**: Each built-in tool can be called directly via unit tests. LLM client can complete a simple prompt.

### Phase 4 — Agent Loop & REPL (Week 4–5)
**Goal**: `mita chat` starts an interactive session. The agent can use tools in a loop.

- [ ] `mita.agent.system_prompt` — Build system prompt from config + memory
- [ ] `mita.agent.context` — Context assembly
- [ ] `mita.agent.conversation` — Message history with truncation
- [ ] `mita.agent.loop` — Core loop: LLM call → parse tool calls → confirm → execute → repeat
- [ ] `mita.ui.repl` — Interactive input with prompt_toolkit
- [ ] `mita.ui.display` — Rich markdown, code blocks, diffs
- [ ] `mita.ui.spinner` — Thinking indicator
- [ ] CLI commands: `mita chat`, `mita ask "<prompt>"`

**Testable**: `mita chat` opens REPL. User can ask the agent to read a file, and it does so via tool calls.

### Phase 5 — Codebase Indexing (Week 6)
**Goal**: `mita index` builds a vector index. Agent uses RAG to answer questions about codebases.

- [ ] `mita.index.parser` — Tree-sitter code chunking
- [ ] `mita.index.embeddings` — Ollama embedding generation
- [ ] `mita.index.store` — LanceDB operations (upsert, query)
- [ ] `mita.index.retriever` — RAG retrieval pipeline
- [ ] Wire retriever into `agent.context`
- [ ] CLI commands: `mita index build`, `mita index status`, `mita index search "<query>"`

**Testable**: `mita index build` indexes a project. `mita index search "auth"` returns relevant code chunks.

### Phase 6 — Skills System (Week 7)
**Goal**: Users can invoke `/commit` or `/review` and it expands into a rich prompt.

- [ ] `mita.skills.loader` — Parse skill Markdown with YAML frontmatter
- [ ] `mita.skills.executor` — Template rendering with arg substitution
- [ ] `mita.skills.manager` — CRUD operations
- [ ] CLI commands: `mita skills list`, `mita skills show <name>`, `mita skills create <name>`
- [ ] `/` prefix triggers skill lookup in REPL
- [ ] Ship 3 built-in skills: `commit`, `review`, `explain`

**Testable**: `/commit` in the REPL produces a well-formatted commit message prompt.

### Phase 7 — MCP Plugin System (Week 8)
**Goal**: Users can configure MCP servers and the agent can call their tools.

- [ ] `mita.plugins.mcp_transport` — stdio and SSE transport
- [ ] `mita.plugins.mcp_client` — Protocol handshake, tool listing, tool calling
- [ ] `mita.plugins.lifecycle` — Process management, health checks
- [ ] `mita.plugins.manager` — Plugin CRUD
- [ ] Wire MCP tools into `tools.registry` so agent sees them
- [ ] CLI commands: `mita plugins list`, `mita plugins add <name>`, `mita plugins remove <name>`

**Testable**: Configure a test MCP server, `mita plugins list` shows its tools, agent can call them.

### Phase 8 — Hooks System (Week 9)
**Goal**: Users can configure shell hooks that fire at lifecycle events.

- [ ] `mita.hooks.schema` — Hook definitions
- [ ] `mita.hooks.runner` — Execute hooks with context injection
- [ ] Wire hooks into agent loop at all trigger points
- [ ] CLI commands: `mita hooks list`, `mita hooks add`

**Testable**: Configure an `on_file_write` hook, write a file via agent, hook fires.

### Phase 9 — Polish & Non-interactive Mode (Week 10)
**Goal**: Piping support, single-shot mode, output formatting.

- [ ] `mita ask "prompt"` works non-interactively (reads stdin, writes stdout)
- [ ] `echo "explain this" | mita` piping support
- [ ] `--output json` flag for structured output
- [ ] `--no-tools` flag for pure chat mode
- [ ] Proper signal handling (Ctrl-C, Ctrl-D)
- [ ] Progress bars for indexing and model pulls

**Testable**: `echo "what is 2+2" | mita ask --no-tools` outputs an answer.

### Phase 10 — Distribution & Docs (Week 11)
**Goal**: `pipx install mita-code` works. Docs are complete.

- [ ] Final `pyproject.toml` polish (classifiers, URLs, etc.)
- [ ] PyPI publishing workflow (GitHub Actions)
- [ ] `docs/` — Getting started, configuration, skills, plugins, hooks
- [ ] `README.md` — Full user-facing readme
- [ ] Shell completion generation (`mita --install-completion`)
- [ ] `mita doctor` — Diagnostics command (check Ollama, models, config)

---

## 5. Key Technical Risks

### Risk 1: Tool Call Reliability on Local Models
**Problem**: Local models (especially <14B) produce malformed tool calls — missing arguments, hallucinated tool names, invalid JSON.
**Mitigation**:
- Use **Instructor** with retry logic and Pydantic validation to coerce outputs.
- Provide explicit JSON schema in the system prompt in addition to function-calling format.
- Implement a "repair" pass: if JSON parsing fails, attempt regex extraction of tool name + args.
- Maintain a tested compatibility matrix of models × tool call reliability.
- Default to models with strong tool-call benchmarks (Qwen 2.5 Coder, DeepSeek Coder V2).

### Risk 2: Context Window Management
**Problem**: Large codebases + conversation history + memory + RAG results can overflow the context window (often 8K–32K for local models).
**Mitigation**:
- Token counting via `tiktoken` (approximation) before each LLM call.
- Aggressive summarization: after N turns, summarize older turns into a compressed message.
- RAG results are injected on-demand, not preloaded — only add chunks when relevant.
- Memory files have hard line limits (200 lines) with truncation warnings.
- System prompt is kept lean; verbose instructions are only loaded for relevant tools.

### Risk 3: MCP Subprocess Lifecycle
**Problem**: MCP stdio servers are child processes that can crash, hang, or leak. Stale processes on exit.
**Mitigation**:
- Use `asyncio.subprocess` with proper signal handling.
- Implement a health-check ping (MCP `ping` method) with configurable timeout.
- Register an `atexit` handler that sends SIGTERM then SIGKILL to all MCP children.
- Set a per-request timeout on tool calls to MCP servers.
- Use process groups so child trees are killed together.

### Risk 4: Cross-Platform Shell Execution
**Problem**: Shell commands behave differently on macOS, Linux, and Windows. Path separators, available commands, and permissions differ.
**Mitigation**:
- Default shell: `$SHELL` on Unix, `cmd.exe` on Windows (with `powershell` option).
- Use `shlex.quote()` on Unix, proper escaping on Windows.
- Never inject unsanitized user input into shell strings — always pass as args list.
- Test on macOS and Linux in CI. Windows is best-effort in early phases.
- Built-in tools (file_read, glob, grep) use Python stdlib, avoiding shell entirely.

### Risk 5: Ollama Process Management
**Problem**: Should mita manage the Ollama server process, or expect it to be running?
**Mitigation**:
- **Detection first**: Check if Ollama is already running on the configured host/port.
- **Do not auto-start Ollama** in Phase 1 — require user to have it running. Print a clear diagnostic message if not found.
- In Phase 9+, optionally add `mita ollama start/stop` commands that manage it as a subprocess.
- If Ollama dies mid-session, catch connection errors and prompt user to restart.

### Risk 6: Embedding Quality for RAG
**Problem**: Local embedding models produce lower-quality vectors than cloud APIs, leading to poor retrieval.
**Mitigation**:
- Default to `nomic-embed-text` (strong local embedding model).
- Combine vector search with keyword search (BM25 via LanceDB full-text search) for hybrid retrieval.
- Chunk code by semantic units (functions/classes via Tree-sitter) rather than fixed token windows.
- Allow users to re-index with a different embedding model if quality is poor.

### Risk 7: Instructor / Structured Output Compatibility
**Problem**: Instructor's structured output relies on specific model output formats. Not all Ollama models support the same format.
**Mitigation**:
- Use Instructor's `Mode.JSON` mode (most compatible) rather than `Mode.TOOLS` for models that don't support native tool calling.
- Implement a model capability detection step: test if a model supports tool-call format on first use, cache the result.
- Fall back to regex-based JSON extraction from raw text as a last resort.
- Document which models work best with structured output.

---

## 6. CLI Command Surface

### `mita chat`
Start an interactive agentic chat session.
```
mita chat [--model MODEL] [--no-tools] [--no-memory] [--no-index]
```
- `--model`: Override the default model for this session
- `--no-tools`: Pure chat mode, no tool execution
- `--no-memory`: Don't load MITA.md files
- `--no-index`: Don't use RAG retrieval
- **Example**: `mita chat --model deepseek-coder-v2:16b`

### `mita ask`
Single-shot prompt (non-interactive). Reads from argument or stdin.
```
mita ask "<prompt>" [--model MODEL] [--no-tools] [--output json|text|markdown]
```
- **Example**: `mita ask "explain the auth module"`
- **Example**: `cat error.log | mita ask "what went wrong?"`

### `mita models`
Model management commands.
```
mita models list                          # List installed Ollama models
mita models pull <name>                   # Pull a model from Ollama registry
mita models remove <name>                 # Remove an installed model
mita models recommend                     # Recommend models for your hardware
mita models info <name>                   # Show model details and compatibility
mita models default <name>               # Set the default model
```
- **Example**: `mita models recommend`
- **Example**: `mita models pull qwen2.5-coder:14b`

### `mita config`
Configuration management.
```
mita config show                          # Show merged config
mita config edit [--global]               # Open config in $EDITOR
mita config set <key> <value> [--global]  # Set a config value
mita config path                          # Show config file paths
```
- **Example**: `mita config set model.default "qwen2.5-coder:14b"`
- **Example**: `mita config edit --global`

### `mita memory`
MITA.md memory management.
```
mita memory show                          # Show all discovered memory content
mita memory edit [--global|--project]     # Open relevant MITA.md in $EDITOR
mita memory add "<text>" [--global|--project]  # Append to memory file
mita memory path                          # Show discovered memory file paths
```
- **Example**: `mita memory add "Always use pytest for tests" --project`

### `mita index`
Codebase indexing.
```
mita index build [--force]                # Index the current project
mita index status                         # Show index stats (chunks, last updated)
mita index search "<query>" [--top-k N]   # Search the index
mita index clear                          # Delete the index
```
- **Example**: `mita index build`
- **Example**: `mita index search "database connection" --top-k 5`

### `mita skills`
Skill management.
```
mita skills list                          # List available skills
mita skills show <name>                   # Show skill details and template
mita skills create <name>                 # Create a new skill from template
mita skills path                          # Show skill search paths
```
- **Example**: `mita skills list`
- **Example**: `mita skills show commit`

### `mita plugins`
MCP plugin management.
```
mita plugins list                         # List configured plugins and their tools
mita plugins add <name> --command "<cmd>" # Add a stdio MCP server
mita plugins add <name> --url "<url>"     # Add an SSE MCP server
mita plugins remove <name>                # Remove a plugin
mita plugins test <name>                  # Test plugin connectivity
```
- **Example**: `mita plugins add filesystem --command "npx @modelcontextprotocol/server-filesystem /home/user"`

### `mita hooks`
Hook management.
```
mita hooks list                           # List configured hooks
mita hooks add <event> "<command>"        # Add a hook
mita hooks remove <event>                 # Remove a hook
```
- **Example**: `mita hooks add on_file_write "eslint --fix {file_path}"`

### `mita doctor`
Diagnostics.
```
mita doctor                               # Check Ollama, models, config, system health
```
- **Example**: `mita doctor`

### `mita version`
```
mita version                              # Print version
mita --version                            # Same
```

---

## 7. Configuration Schema

### `~/.config/mita/config.toml` (Global)

```toml
# ─── Ollama Connection ────────────────────────────────────────────
[ollama]
host = "http://localhost:11434"   # Ollama server URL
timeout = 120                     # Request timeout in seconds

# ─── Model Settings ──────────────────────────────────────────────
[model]
default = "qwen2.5-coder:7b"     # Default model for chat/agent
embedding = "nomic-embed-text"    # Model used for embeddings
temperature = 0.1                 # LLM temperature (0.0–1.0)
max_tokens = 4096                 # Max tokens in LLM response
context_window = 32768            # Context window size for the default model

# ─── Tool Settings ────────────────────────────────────────────────
[tools]
# Tools in this list execute without asking for user confirmation
auto_approve = ["file_read", "glob", "grep"]
confirm_destructive = true        # Ask before destructive actions (rm, git push, etc.)
shell_timeout = 120               # Max seconds for shell command execution
# Commands that are always blocked
banned_commands = ["rm -rf /", "mkfs", "dd if=/dev/zero"]

# ─── Memory Settings ─────────────────────────────────────────────
[memory]
max_lines_per_file = 200          # Max lines loaded per MITA.md file
max_total_tokens = 4000           # Max tokens of memory injected into context

# ─── Index / RAG Settings ────────────────────────────────────────
[index]
enabled = true                    # Enable codebase indexing
chunk_size = 512                  # Tokens per code chunk
chunk_overlap = 64                # Overlap between chunks
top_k = 10                        # Number of chunks retrieved per query
exclude_patterns = [              # Patterns to skip during indexing
    "*.lock",
    "node_modules/**",
    ".git/**",
    "*.min.js",
    "*.min.css",
    "dist/**",
    "build/**",
    "__pycache__/**",
]

# ─── UI Settings ──────────────────────────────────────────────────
[ui]
theme = "auto"                    # "auto", "dark", "light"
show_token_count = true           # Show token usage after each response
stream = true                     # Stream LLM output token-by-token
markdown = true                   # Render markdown in responses

# ─── Skills Search Paths ─────────────────────────────────────────
# Directories to search for skill files (Markdown)
skills_paths = [
    "~/.config/mita/skills",
]

# ─── Hooks ────────────────────────────────────────────────────────
# [[hooks]]
# event = "on_file_write"         # Trigger event
# command = "eslint --fix {file_path}"  # Shell command to run
# match = "*.js"                  # Optional: only trigger for matching files/tools

# ─── Plugins (MCP Servers) ───────────────────────────────────────
# [[plugins]]
# name = "filesystem"
# transport = "stdio"
# command = "npx"
# args = ["@modelcontextprotocol/server-filesystem", "/home/user"]
#
# [[plugins]]
# name = "web-search"
# transport = "sse"
# url = "http://localhost:3001/sse"
```

### `.mita/settings.toml` (Project-level Override)

```toml
# Project-level overrides. Only specify keys you want to override.
# All keys follow the same schema as the global config.

[model]
default = "qwen2.5-coder:14b"    # This project needs a bigger model
context_window = 65536

[tools]
auto_approve = ["file_read", "glob", "grep", "git"]  # Trust git in this project
shell_timeout = 300               # Long-running builds

[index]
exclude_patterns = [
    "*.lock",
    "vendor/**",
    ".git/**",
    "test/fixtures/**",
]

# Project-specific skills
skills_paths = [".mita/skills"]

# Project-specific hooks
[[hooks]]
event = "on_file_write"
command = "ruff check --fix {file_path}"
match = "*.py"

# Project-specific MCP plugins
[[plugins]]
name = "project-db"
transport = "stdio"
command = "python"
args = [".mita/db_server.py"]
```

### Config Merge Strategy

1. Load global config from `~/.config/mita/config.toml`
2. Load project config from `.mita/settings.toml` (if exists)
3. Deep merge: project values override global values at the leaf level
4. For lists (`hooks`, `plugins`, `skills_paths`): project values are **appended** to global values
5. Validate merged result against `MitaConfig` Pydantic model

---

## 8. MITA.md Memory System

### File Locations & Lookup Order

Memory files are named `MITA.md` and discovered by walking up the directory tree from the current working directory:

```
Lookup order (highest priority first):
1. ./<current_dir>/MITA.md          — Directory-level memory
2. ./<project_root>/MITA.md         — Project-level memory (root = .git or .mita dir)
3. ~/.config/mita/MITA.md           — Global memory
```

Additionally, `.mita/MITA.md` inside a project is equivalent to `./MITA.md` at project root.

### Discovery Algorithm

```python
def discover_memory_files(cwd: Path) -> list[Path]:
    files = []
    # 1. Walk up from cwd to filesystem root
    current = cwd
    while True:
        candidate = current / "MITA.md"
        if candidate.is_file():
            files.append(candidate)
        # Also check .mita/ subdirectory
        dotmita = current / ".mita" / "MITA.md"
        if dotmita.is_file() and dotmita not in files:
            files.append(dotmita)
        # Stop at project root or filesystem root
        if (current / ".git").exists() or (current / ".mita").exists():
            break
        parent = current.parent
        if parent == current:
            break
        current = parent
    # 2. Always add global memory
    global_mem = Path.home() / ".config" / "mita" / "MITA.md"
    if global_mem.is_file() and global_mem not in files:
        files.append(global_mem)
    # 3. Reverse so global is first (lowest priority → injected first)
    return list(reversed(files))
```

### Context Injection

Memory content is injected into the system prompt in a clearly delimited block:

```
<memory>
<!-- Source: ~/.config/mita/MITA.md (global) -->
{global memory content}

<!-- Source: /path/to/project/MITA.md (project) -->
{project memory content}

<!-- Source: /path/to/project/src/MITA.md (directory) -->
{directory memory content}
</memory>
```

### Size Limits & Truncation

- Each MITA.md file is truncated to `memory.max_lines_per_file` (default 200) lines.
- If a file exceeds the limit, the first 200 lines are kept, and a note is appended:
  `[... truncated at 200 lines. Keep MITA.md concise.]`
- Total memory tokens are capped at `memory.max_total_tokens` (default 4000).
  If exceeded, files are truncated proportionally starting from the lowest-priority (global) file.

### CLI Commands

| Command | Behavior |
|---|---|
| `mita memory show` | Print all discovered memory with source annotations |
| `mita memory edit --global` | Open `~/.config/mita/MITA.md` in `$EDITOR` |
| `mita memory edit --project` | Open `<project_root>/MITA.md` in `$EDITOR` |
| `mita memory edit` | Open the nearest MITA.md (or create at project root) in `$EDITOR` |
| `mita memory add "text"` | Append a line to the nearest MITA.md |
| `mita memory add "text" --global` | Append to global MITA.md |
| `mita memory path` | Print all discovered MITA.md paths |

---

## 9. Agent Loop Pseudocode

```python
async def run_agent(user_prompt: str, config: MitaConfig) -> None:
    # ── 1. Initialize ────────────────────────────────────────
    conversation = Conversation()
    tool_registry = ToolRegistry()
    tool_registry.register_builtins()

    # Load MCP plugin tools
    plugin_manager = PluginManager(config.plugins)
    await plugin_manager.start_all()
    for tool_def in plugin_manager.list_tools():
        tool_registry.register(tool_def)

    # ── 2. Assemble system prompt ─────────────────────────────
    memory_content = load_memory(cwd=Path.cwd())
    system_prompt = build_system_prompt(
        config=config,
        memory=memory_content,
        tools=tool_registry.get_definitions(),
    )
    conversation.add(Message(role=Role.SYSTEM, content=system_prompt))

    # ── 3. Inject RAG context if index exists ─────────────────
    if config.index.enabled and index_exists():
        retriever = Retriever(config)
        relevant_chunks = retriever.search(user_prompt, top_k=config.index.top_k)
        if relevant_chunks:
            rag_context = format_chunks(relevant_chunks)
            conversation.add(Message(
                role=Role.SYSTEM,
                content=f"<codebase_context>\n{rag_context}\n</codebase_context>"
            ))

    # ── 4. Add user message ───────────────────────────────────
    #    Check if it's a skill invocation
    if user_prompt.startswith("/"):
        skill = skills_loader.find(user_prompt)
        if skill:
            user_prompt = render_skill(skill, user_prompt)

    conversation.add(Message(role=Role.USER, content=user_prompt))

    # ── 5. Run hooks ──────────────────────────────────────────
    await run_hooks("session_start", config.hooks)

    # ── 6. Agent loop ─────────────────────────────────────────
    MAX_ITERATIONS = 25
    iteration = 0

    while iteration < MAX_ITERATIONS:
        iteration += 1

        # ── 6a. Truncate conversation if needed ───────────────
        conversation.truncate_to_fit(config.model.context_window)

        # ── 6b. Call LLM ─────────────────────────────────────
        display_spinner("Thinking...")
        try:
            response = await stream_chat(
                messages=conversation.get_messages_for_api(),
                model=config.model.default,
                tools=tool_registry.get_openai_schema(),
                temperature=config.model.temperature,
                max_tokens=config.model.max_tokens,
                stream=config.ui.stream,
                on_token=display_streaming_token,  # Rich live display
            )
        except ConnectionError:
            display_error("Cannot connect to Ollama. Is it running?")
            return
        except TimeoutError:
            display_error("LLM request timed out.")
            return

        hide_spinner()

        # ── 6c. Display assistant text ────────────────────────
        if response.content:
            display_markdown(response.content)

        # ── 6d. Process tool calls ────────────────────────────
        if not response.tool_calls:
            # No tool calls → agent is done, break
            conversation.add(Message(
                role=Role.ASSISTANT,
                content=response.content or "",
            ))
            break

        # Add assistant message with tool calls
        conversation.add(Message(
            role=Role.ASSISTANT,
            content=response.content or "",
            tool_calls=response.tool_calls,
        ))

        # Execute each tool call
        for tool_call in response.tool_calls:
            # ── Validate tool exists ──────────────────────────
            if not tool_registry.has(tool_call.name):
                result = ToolResult(
                    tool_call_id=tool_call.id,
                    success=False,
                    error=f"Unknown tool: {tool_call.name}",
                )
            else:
                tool_def = tool_registry.get(tool_call.name)

                # ── Pre-tool hook ─────────────────────────────
                await run_hooks("pre_tool_call", config.hooks, context={
                    "tool": tool_call.name,
                    "args": tool_call.arguments,
                })

                # ── Safety check ──────────────────────────────
                needs_confirm = (
                    tool_def.destructive
                    and config.tools.confirm_destructive
                    and tool_call.name not in config.tools.auto_approve
                )

                if needs_confirm:
                    display_tool_call(tool_call)  # Show what will be executed
                    approved = prompt_user_confirm(
                        f"Allow {tool_call.name}?"
                    )
                    if not approved:
                        result = ToolResult(
                            tool_call_id=tool_call.id,
                            success=False,
                            error="User denied this action.",
                        )
                        conversation.add(Message(
                            role=Role.TOOL,
                            content=result.model_dump_json(),
                            tool_call_id=tool_call.id,
                            name=tool_call.name,
                        ))
                        continue

                # ── Execute tool ──────────────────────────────
                display_tool_executing(tool_call)
                try:
                    result = await tool_registry.execute(tool_call)
                except Exception as e:
                    result = ToolResult(
                        tool_call_id=tool_call.id,
                        success=False,
                        error=f"Tool execution error: {str(e)}",
                    )

                # ── Truncate large outputs ────────────────────
                if len(result.output) > 10000:
                    result.output = result.output[:10000] + "\n[... output truncated]"
                    result.truncated = True

                # ── Post-tool hook ────────────────────────────
                await run_hooks("post_tool_call", config.hooks, context={
                    "tool": tool_call.name,
                    "result": result,
                })

            # ── Add tool result to conversation ───────────────
            display_tool_result(result)
            conversation.add(Message(
                role=Role.TOOL,
                content=result.output if result.success else f"ERROR: {result.error}",
                tool_call_id=tool_call.id,
                name=tool_call.name,
            ))

    # ── 7. Max iterations guard ───────────────────────────────
    if iteration >= MAX_ITERATIONS:
        display_warning(f"Agent reached max iterations ({MAX_ITERATIONS}). Stopping.")

    # ── 8. Session end hooks ──────────────────────────────────
    await run_hooks("session_end", config.hooks)

    # ── 9. Display token usage ────────────────────────────────
    if config.ui.show_token_count:
        display_token_usage(conversation.total_tokens)

    # ── 10. Cleanup ───────────────────────────────────────────
    await plugin_manager.stop_all()
```

### REPL Wrapper

```python
async def repl(config: MitaConfig) -> None:
    """Interactive REPL that wraps the agent loop."""
    display_welcome(config)

    while True:
        try:
            user_input = await prompt_input("mita> ")
        except (EOFError, KeyboardInterrupt):
            display_goodbye()
            break

        user_input = user_input.strip()
        if not user_input:
            continue

        if user_input.lower() in ("/quit", "/exit", "/q"):
            display_goodbye()
            break

        if user_input.lower() == "/clear":
            # Reset conversation but keep config/memory
            continue

        await run_agent(user_input, config)
```

---

## 10. Open Questions

### Q1: Async vs Sync Architecture
**Options**:
- (A) Fully async (`asyncio`) — natural for streaming, MCP subprocess management, concurrent tool execution
- (B) Sync with threading for I/O — simpler, but harder to manage MCP lifecycles

**Recommendation**: **(A) Async.** The agent loop is inherently I/O-bound (LLM calls, subprocess management, MCP communication). `asyncio` is the right fit. Use `anyio` for portability if needed. LiteLLM supports async natively.

### Q2: Conversation Persistence
**Options**:
- (A) No persistence — conversations are ephemeral per session
- (B) Auto-save conversation history to `.mita/history/`
- (C) Optional save with `mita history save` / `mita history load`

**Recommendation**: **(A) for Phase 1, then (C) later.** Keep Phase 1 simple. Conversation persistence is a nice-to-have that can be layered on without architectural changes.

### Q3: Multi-file Edit Strategy
**Options**:
- (A) One tool call per file (simpler, works with all models)
- (B) Batch edit tool that accepts multiple files (fewer round trips)

**Recommendation**: **(A) One tool call per file.** Local models handle simpler tool schemas better. The agent loop naturally handles multi-step edits through iteration. Batch editing adds complexity to tool parsing for marginal gain.

### Q4: How to Handle Models Without Tool-Call Support
**Options**:
- (A) Require tool-call support, refuse to run models without it
- (B) Fall back to prompt-based tool calling (inject JSON schema in system prompt, parse from raw output)
- (C) Use Instructor's JSON mode as the universal path, bypassing native tool calling entirely

**Recommendation**: **(C) Instructor JSON mode as primary, with native tool calling as optional optimization.** This gives the widest model compatibility. Instructor handles the parsing and validation. For models that support native tool calling well, we can add a fast path later.

### Q5: Index Storage Location
**Options**:
- (A) `.mita/index/` in the project (gitignored)
- (B) `~/.cache/mita/<project-hash>/` in user cache
- (C) User choice via config

**Recommendation**: **(A) `.mita/index/` in the project.** Keep it local and discoverable. Add `.mita/index/` to the default `.gitignore` template. This is simplest and avoids orphaned caches.

### Q6: Embedding Model Bundling
**Options**:
- (A) Auto-pull the embedding model on first `mita index build`
- (B) Require user to manually pull it
- (C) Prompt user to confirm the pull

**Recommendation**: **(C) Prompt to confirm.** Embedding models are ~250MB. Don't silently download. Show the model name and size, ask for confirmation, then pull.

### Q7: LiteLLM vs Direct Ollama Client
**Options**:
- (A) Use LiteLLM for all LLM calls (adds abstraction, supports future backends)
- (B) Use `ollama` Python client directly (simpler, fewer deps)
- (C) Use LiteLLM but keep `ollama` client for model management (pull/list/delete)

**Recommendation**: **(C) Both.** LiteLLM for chat completions (OpenAI-compatible interface, easy to add LM Studio or other backends later). `ollama` client for model management operations (pull, list, delete, show) which LiteLLM doesn't cover. This gives us the best of both.

### Q8: Python Version Floor
**Options**:
- (A) Python 3.11+ (tomllib in stdlib, nice asyncio improvements)
- (B) Python 3.10+ (wider compatibility, use `tomli` for TOML)
- (C) Python 3.12+ (newest features, smallest audience)

**Recommendation**: **(A) Python 3.11+.** Good balance. `tomllib` is in stdlib (no extra dep). Task groups in asyncio are stable. Wide enough adoption via pipx.

---

## Appendix: Dependency List

```
# Core
typer >= 0.9.0
rich >= 13.0.0
litellm >= 1.0.0
instructor >= 1.0.0
ollama >= 0.3.0
pydantic >= 2.0.0
prompt-toolkit >= 3.0.0

# Indexing
lancedb >= 0.4.0
tree-sitter >= 0.21.0
tree-sitter-languages >= 1.10.0  # pre-built grammars

# Git
gitpython >= 3.1.0

# System
psutil >= 5.9.0

# MCP
mcp >= 1.0.0  # MCP Python SDK

# Dev
pytest >= 8.0.0
pytest-asyncio >= 0.23.0
ruff >= 0.3.0
mypy >= 1.8.0
```
