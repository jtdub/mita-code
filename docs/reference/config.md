# Configuration Schema

Full reference for all configuration options in `config.toml` / `.mita/settings.toml`.

## `[ollama]`

| Key | Type | Default | Description |
|---|---|---|---|
| `host` | string | `"http://localhost:11434"` | Ollama server URL |
| `timeout` | int | `120` | Request timeout in seconds |

## `[model]`

| Key | Type | Default | Description |
|---|---|---|---|
| `default` | string | `"qwen2.5-coder:7b"` | Default LLM for chat/ask |
| `embedding` | string | `"nomic-embed-text"` | Embedding model for codebase indexing |
| `temperature` | float | `0.1` | Sampling temperature |
| `max_tokens` | int | `4096` | Maximum tokens per response |
| `context_window` | int | `32768` | Context window size |

## `[tools]`

| Key | Type | Default | Description |
|---|---|---|---|
| `auto_approve` | list[str] | `["file_read", "glob", "grep"]` | Tools that run without confirmation |
| `confirm_destructive` | bool | `true` | Prompt before destructive actions |
| `shell_timeout` | int | `120` | Shell command timeout in seconds |
| `banned_commands` | list[str] | `["rm -rf /", ...]` | Commands that are always blocked |

## `[memory]`

| Key | Type | Default | Description |
|---|---|---|---|
| `max_lines_per_file` | int | `200` | Max lines per MITA.md file |
| `max_total_tokens` | int | `4000` | Max total tokens for all memory |

## `[index]`

| Key | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `true` | Enable RAG indexing |
| `chunk_size` | int | `512` | Lines per chunk |
| `chunk_overlap` | int | `64` | Overlap between chunks |
| `top_k` | int | `10` | Number of chunks to retrieve |
| `exclude_patterns` | list[str] | See below | Glob patterns to exclude |

Default exclude patterns:

```toml
exclude_patterns = [
    "*.lock", ".mita/**", "node_modules/**", ".git/**",
    "*.min.js", "*.min.css", "dist/**", "build/**", "__pycache__/**",
]
```

## `[ui]`

| Key | Type | Default | Description |
|---|---|---|---|
| `theme` | string | `"auto"` | Color theme |
| `show_token_count` | bool | `true` | Show token usage after responses |
| `stream` | bool | `true` | Stream responses token-by-token |
| `markdown` | bool | `true` | Render Markdown in responses |

## `skills_paths`

| Key | Type | Default | Description |
|---|---|---|---|
| `skills_paths` | list[str] | `["~/.config/mita/skills", ".mita/skills"]` | Directories to search for skill files |

## `[[hooks]]`

| Key | Type | Required | Description |
|---|---|---|---|
| `event` | string | yes | Lifecycle event name |
| `command` | string | yes | Shell command to run |
| `match` | string | no | Glob pattern to filter (e.g., `"*.py"`) |

## `[[plugins]]`

| Key | Type | Required | Description |
|---|---|---|---|
| `name` | string | yes | Plugin name |
| `transport` | string | no | `"stdio"` or `"sse"` (default: `"stdio"`) |
| `command` | string | no | Command to start the plugin |
| `args` | list[str] | no | Arguments for the command |
| `url` | string | no | URL for SSE transport |
| `env` | dict | no | Environment variables |
