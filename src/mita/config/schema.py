"""Pydantic models for Mita configuration."""

from __future__ import annotations

import enum
import os

from pydantic import BaseModel, Field, field_validator, model_validator


class LLMProvider(enum.StrEnum):
    """Local inference backend that serves the chat/embeddings API.

    All except ``ollama`` are OpenAI-compatible HTTP servers. ``ollama`` keeps its
    native API (and daemon management via the ``[ollama]`` section).
    """

    OLLAMA = "ollama"
    LLAMACPP = "llamacpp"
    VLLM = "vllm"
    LMSTUDIO = "lmstudio"
    TGI = "tgi"
    OPENAI_COMPATIBLE = "openai_compatible"


class LLMSettings(BaseModel):
    """Backend selection for chat and embeddings (audit finding C5).

    Backward compatible: with no ``[llm]`` section the provider defaults to Ollama and
    the existing ``[ollama] host`` is used as the base URL.
    """

    provider: LLMProvider = LLMProvider.OLLAMA
    base_url: str = ""  # empty → provider default (Ollama uses [ollama] host)
    api_key: str = ""  # empty → placeholder for openai-routed providers
    context_probe: str = "auto"  # auto | off | <int>

    @field_validator("api_key", "base_url")
    @classmethod
    def _expand_env(cls, v: str) -> str:
        """Expand ${VAR}/$VAR from the environment so secrets aren't stored in TOML."""
        return os.path.expandvars(v)

    @field_validator("context_probe")
    @classmethod
    def _valid_probe(cls, v: str) -> str:
        if v in ("auto", "off"):
            return v
        try:
            if int(v) > 0:
                return v
        except ValueError:
            pass
        raise ValueError('context_probe must be "auto", "off", or a positive integer')


class PermissionMode(enum.StrEnum):
    """Permission mode controlling which tools are auto-approved.

    - ask: only read-only tools auto-approved (safest)
    - auto_edit: file reads/writes/edits and git auto-approved, shell requires confirmation
    - trust: all tools auto-approved (dangerous)
    """

    ASK = "ask"
    AUTO_EDIT = "auto_edit"
    TRUST = "trust"


# Tools auto-approved in each permission mode
PERMISSION_MODE_TOOLS: dict[PermissionMode, list[str]] = {
    PermissionMode.ASK: ["file_read", "glob", "grep"],
    PermissionMode.AUTO_EDIT: ["file_read", "glob", "grep", "file_write", "file_edit", "git"],
    PermissionMode.TRUST: [
        "file_read",
        "glob",
        "grep",
        "file_write",
        "file_edit",
        "git",
        "shell",
    ],
}


class OllamaSettings(BaseModel):
    """Ollama server connection settings."""

    host: str = "http://localhost:11434"
    timeout: int = 120
    auto_manage: bool = True

    @field_validator("timeout")
    @classmethod
    def timeout_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("timeout must be positive")
        return v


class OllamaRuntimeOptions(BaseModel):
    """Ollama model runtime options passed via the API.

    These map to Ollama's /api/chat 'options' field and let users
    tune inference for their hardware.
    """

    num_gpu: int | None = None
    num_thread: int | None = None
    num_ctx: int | None = None
    num_batch: int | None = None
    use_mmap: bool | None = None
    use_mlock: bool | None = None
    num_keep: int | None = None
    main_gpu: int | None = None
    low_vram: bool | None = None
    flash_attention: bool | None = None

    def to_api_dict(self) -> dict[str, int | bool]:
        """Return only non-None options for the Ollama API."""
        return {k: v for k, v in self.model_dump().items() if v is not None}


class ModelSettings(BaseModel):
    """LLM model settings."""

    default: str = "qwen2.5-coder:7b"
    embedding: str = "nomic-embed-text"
    temperature: float = 0.1
    max_tokens: int = 4096
    context_window: int = 32768
    ollama_options: OllamaRuntimeOptions = Field(default_factory=OllamaRuntimeOptions)

    @field_validator("max_tokens")
    @classmethod
    def max_tokens_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("max_tokens must be positive")
        return v

    @field_validator("context_window")
    @classmethod
    def context_window_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("context_window must be positive")
        return v

    @model_validator(mode="after")
    def context_window_gte_max_tokens(self) -> ModelSettings:
        if self.context_window < self.max_tokens:
            raise ValueError(
                f"context_window ({self.context_window}) must be >= max_tokens ({self.max_tokens})"
            )
        return self


class ToolSettings(BaseModel):
    """Tool execution settings."""

    permission_mode: PermissionMode = PermissionMode.ASK
    auto_approve: list[str] = Field(
        default_factory=lambda: list(PERMISSION_MODE_TOOLS[PermissionMode.ASK])
    )
    confirm_destructive: bool = True
    shell_timeout: int = 120
    glob_max_results: int = 500
    grep_max_matches: int = 200
    banned_commands: list[str] = Field(
        default_factory=lambda: ["rm -rf /", "mkfs", "dd if=/dev/zero"]
    )

    @property
    def effective_auto_approve(self) -> set[str]:
        """Return the effective set of auto-approved tools.

        Merges the permission mode's default tools with any explicit auto_approve overrides.
        """
        mode_tools = set(PERMISSION_MODE_TOOLS[self.permission_mode])
        return mode_tools | set(self.auto_approve)

    @field_validator("shell_timeout")
    @classmethod
    def shell_timeout_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("shell_timeout must be positive")
        return v


class MemorySettings(BaseModel):
    """MITA.md memory system settings."""

    max_lines_per_file: int = 200
    max_total_tokens: int = 4000


class IndexSettings(BaseModel):
    """Codebase indexing settings."""

    enabled: bool = True
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 10
    max_file_size: int = 1_000_000  # bytes; skip files larger than this (finding C6)
    respect_gitignore: bool = True
    context_token_budget: int = 2000  # retrieval budget, NOT the whole context window
    relevance_floor: float = 0.0  # drop hybrid results below this score (0 = keep all)
    exclude_patterns: list[str] = Field(
        default_factory=lambda: [
            "*.lock",
            ".mita/**",
            "node_modules/**",
            ".git/**",
            "*.min.js",
            "*.min.css",
            "dist/**",
            "build/**",
            "__pycache__/**",
            # Virtualenvs / vendored / caches — omitting these embedded whole
            # virtualenvs (finding C6).
            ".venv/**",
            "venv/**",
            ".tox/**",
            "site-packages/**",
            "target/**",
            "vendor/**",
            ".next/**",
            ".mypy_cache/**",
            ".pytest_cache/**",
            ".ruff_cache/**",
            "htmlcov/**",
            # Likely-secret files — never embed these.
            ".env",
            ".env.*",
            "*.pem",
            "*.key",
            "id_rsa",
            "id_dsa",
            "credentials*",
            ".netrc",
            "*.p12",
            "*.pfx",
            "*.keystore",
        ]
    )

    @model_validator(mode="after")
    def chunk_size_gt_overlap(self) -> IndexSettings:
        if self.chunk_size <= self.chunk_overlap:
            raise ValueError(
                f"chunk_size ({self.chunk_size}) must be > chunk_overlap ({self.chunk_overlap})"
            )
        return self

    @field_validator("top_k")
    @classmethod
    def top_k_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("top_k must be positive")
        return v


class HookSettings(BaseModel):
    """Hook execution settings."""

    timeout: int = 30

    @field_validator("timeout")
    @classmethod
    def timeout_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("hook timeout must be positive")
        return v


class HookDefinition(BaseModel):
    """A lifecycle hook definition."""

    event: str
    command: str
    match: str | None = None


class PluginDefinition(BaseModel):
    """An MCP plugin server definition."""

    name: str
    transport: str = "stdio"
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    url: str | None = None
    env: dict[str, str] = Field(default_factory=dict)

    @field_validator("url")
    @classmethod
    def validate_url_format(cls, v: str | None) -> str | None:
        if v is not None and not v.startswith(("http://", "https://")):
            raise ValueError(f"Plugin URL must start with http:// or https://, got: {v}")
        return v


class UISettings(BaseModel):
    """Terminal UI settings."""

    theme: str = "auto"
    show_token_count: bool = True
    stream: bool = True
    markdown: bool = True


class MitaConfig(BaseModel):
    """Root configuration model — result of merging global + project TOML."""

    llm: LLMSettings = Field(default_factory=LLMSettings)
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    tools: ToolSettings = Field(default_factory=ToolSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    index: IndexSettings = Field(default_factory=IndexSettings)
    ui: UISettings = Field(default_factory=UISettings)
    hook_settings: HookSettings = Field(default_factory=HookSettings)
    max_iterations: int = 25
    hooks: list[HookDefinition] = Field(default_factory=list)
    plugins: list[PluginDefinition] = Field(default_factory=list)
    skills_paths: list[str] = Field(
        default_factory=lambda: ["~/.config/mita/skills", ".mita/skills"]
    )

    @field_validator("max_iterations")
    @classmethod
    def max_iterations_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("max_iterations must be positive")
        return v
