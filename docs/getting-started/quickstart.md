# Quick Start

## Interactive Chat

Start an agentic chat session:

```bash
mita chat
```

Inside the session, type natural language prompts. The agent can read files, write code, run commands, and search your codebase.

```
mita> explain the auth module in this project
mita> refactor the database connection to use async
mita> /commit
```

### REPL Commands

| Command | Action |
|---|---|
| `/quit`, `/exit`, `/q` | Exit the session |
| `/clear` | Clear conversation history |
| `/<skill>` | Invoke a skill (e.g., `/commit`, `/review`) |

## Single-Shot Prompts

For one-off questions without entering an interactive session:

```bash
mita ask "explain the auth module in this project"
```

Pipe input from other commands:

```bash
cat error.log | mita ask "what went wrong?"
git diff | mita ask "review these changes"
```

## Model Management

```bash
# See what fits your hardware
mita models recommend

# List installed models
mita models list

# Pull a model
mita models pull qwen2.5-coder:14b

# Set as default
mita models default qwen2.5-coder:14b
```

## Next Steps

- [Memory System](../guide/memory.md) — Add persistent context with `MITA.md` files
- [Codebase Indexing](../guide/indexing.md) — Enable RAG over your codebase
- [Skills](../guide/skills.md) — Create reusable prompt templates
- [Configuration](../guide/configuration.md) — Customize Mita to your workflow
