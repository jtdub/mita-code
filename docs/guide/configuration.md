# Configuration

Mita uses layered TOML configuration. Global settings cascade into project-level overrides.

## Config Files

| Scope | Location |
|---|---|
| Global | `~/.config/mita/config.toml` |
| Project | `.mita/settings.toml` |

Project settings override global settings.

## View Configuration

```bash
mita config show    # Show merged configuration
mita config path    # Show config file paths
```

## Full Schema

```toml
[ollama]
host = "http://localhost:11434"
timeout = 120

[model]
default = "qwen2.5-coder:7b"
embedding = "nomic-embed-text"
temperature = 0.1
max_tokens = 4096
context_window = 32768

[tools]
auto_approve = ["file_read", "glob", "grep"]
confirm_destructive = true
shell_timeout = 120
banned_commands = ["rm -rf /", "mkfs", "dd if=/dev/zero"]

[memory]
max_lines_per_file = 200
max_total_tokens = 4000

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

[ui]
theme = "auto"
show_token_count = true
stream = true
markdown = true

# Skills search paths
skills_paths = ["~/.config/mita/skills", ".mita/skills"]

# Hooks — lifecycle shell commands
# [[hooks]]
# event = "after_file_write"
# command = "ruff format {file}"
# match = "*.py"

# MCP Plugins
# [[plugins]]
# name = "filesystem"
# transport = "stdio"
# command = "npx"
# args = ["@modelcontextprotocol/server-filesystem", "."]
```
