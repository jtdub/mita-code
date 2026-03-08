# Memory System

Mita uses layered `MITA.md` files that are automatically discovered and injected into the LLM context. This gives the assistant persistent knowledge about your preferences, project conventions, and directory-specific context.

## Memory Scopes

| Scope | Location | Purpose |
|---|---|---|
| Global | `~/.config/mita/MITA.md` | Preferences across all projects |
| Project | `<project_root>/MITA.md` | Project-specific conventions |
| Directory | `<subdir>/MITA.md` | Directory-specific context |

Higher-specificity files take priority. Each file is capped at 200 lines.

## Commands

### View Memory

```bash
mita memory show
```

Displays all discovered memory content that would be injected into the context.

### Add Memory

```bash
mita memory add "Always use pytest for testing" --project
mita memory add "Prefer async/await patterns" --global
```

### Edit Memory

```bash
mita memory edit              # Edit nearest MITA.md
mita memory edit --global     # Edit global MITA.md
mita memory edit --project    # Edit project MITA.md
```

Opens the file in `$EDITOR`.

### Show Paths

```bash
mita memory path
```

Shows discovered memory file paths and whether they exist.

## Configuration

```toml
[memory]
max_lines_per_file = 200
max_total_tokens = 4000
```
