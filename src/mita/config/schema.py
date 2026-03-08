"""Pydantic models for Mita configuration."""

from __future__ import annotations

from pydantic import BaseModel, Field


class OllamaSettings(BaseModel):
    """Ollama server connection settings."""

    host: str = "http://localhost:11434"
    timeout: int = 120


class ModelSettings(BaseModel):
    """LLM model settings."""

    default: str = "qwen2.5-coder:7b"
    embedding: str = "nomic-embed-text"
    temperature: float = 0.1
    max_tokens: int = 4096
    context_window: int = 32768


class ToolSettings(BaseModel):
    """Tool execution settings."""

    auto_approve: list[str] = Field(default_factory=lambda: ["file_read", "glob", "grep"])
    confirm_destructive: bool = True
    shell_timeout: int = 120
    banned_commands: list[str] = Field(
        default_factory=lambda: ["rm -rf /", "mkfs", "dd if=/dev/zero"]
    )


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
        ]
    )


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


class UISettings(BaseModel):
    """Terminal UI settings."""

    theme: str = "auto"
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
    skills_paths: list[str] = Field(default_factory=lambda: ["~/.config/mita/skills"])
