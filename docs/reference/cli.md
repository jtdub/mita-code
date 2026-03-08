# CLI Commands

## Top-Level

| Command | Description |
|---|---|
| `mita chat` | Open an interactive chat session |
| `mita ask <prompt>` | Send a single prompt (non-interactive) |
| `mita --version` | Show version |

## Config

| Command | Description |
|---|---|
| `mita config show` | Show merged configuration |
| `mita config path` | Show config file paths |

## Memory

| Command | Description |
|---|---|
| `mita memory show` | Show all discovered memory content |
| `mita memory path` | Show memory file paths |
| `mita memory edit [--global\|--project]` | Open MITA.md in `$EDITOR` |
| `mita memory add <text> [--global\|--project]` | Append text to a MITA.md file |

## Models

| Command | Description |
|---|---|
| `mita models list` | List installed Ollama models |
| `mita models pull <name>` | Pull a model from the registry |
| `mita models remove <name>` | Remove an installed model |
| `mita models recommend` | Recommend models for your hardware |
| `mita models info <name>` | Show model details |
| `mita models default <name>` | Set the default model |
| `mita models hardware` | Show detected hardware info |

## Index

| Command | Description |
|---|---|
| `mita index build [--force]` | Build the codebase index |
| `mita index status` | Show index statistics |
| `mita index search <query> [--top-k N]` | Search the index |
| `mita index clear` | Delete the index |

## Skills

| Command | Description |
|---|---|
| `mita skills list` | List available skills |
| `mita skills show <name>` | Show skill details |
| `mita skills create <name>` | Create a new skill from template |
| `mita skills path` | Show skill search paths |
