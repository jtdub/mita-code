# Hooks

Hooks are shell commands that Mita fires automatically on lifecycle events. Use them to enforce project conventions, run linters, trigger builds, or log activity -- without modifying the agent itself.

## Lifecycle Events

| Event             | When it fires                          |
|-------------------|----------------------------------------|
| `session_start`   | A chat or ask session begins           |
| `session_end`     | A session ends                         |
| `pre_tool_call`   | Before a built-in tool executes        |
| `post_tool_call`  | After a built-in tool executes         |
| `on_file_write`   | After a file is written or edited      |

## Adding a Hook

```bash
mita hooks add on_file_write "ruff check --fix {file_path}" --match "*.py"
```

This registers a hook that runs `ruff check --fix` on any Python file after Mita writes to it.

The `--match` flag is optional. When provided, the hook only fires if the relevant file path matches the glob pattern.

## Listing Hooks

```bash
mita hooks list
```

Displays all registered hooks grouped by event.

## Removing a Hook

Remove all hooks for a given event:

```bash
mita hooks remove on_file_write
```

## Context Variables

Hook commands can reference context variables that Mita substitutes at runtime:

| Variable      | Available in                        | Description                        |
|---------------|-------------------------------------|------------------------------------|
| `{file_path}` | `on_file_write`, `pre_tool_call`, `post_tool_call` | Path of the affected file |
| `{tool}`      | `pre_tool_call`, `post_tool_call`   | Name of the tool being called      |
| `{result}`    | `post_tool_call`                    | Output of the tool call            |

Variables that are not available for a given event are left as-is in the command string.

## Match Patterns

The `match` field accepts glob-style patterns:

- `*.py` -- all Python files
- `src/**/*.ts` -- TypeScript files under `src/`
- `*.{js,jsx}` -- JavaScript and JSX files

When no match pattern is set, the hook fires unconditionally for its event.

## TOML Configuration

Hooks are stored in your project or global config as `[[hooks]]` entries:

```toml
[[hooks]]
event = "on_file_write"
command = "ruff check --fix {file_path}"
match = "*.py"

[[hooks]]
event = "session_start"
command = "echo 'Session started at $(date)' >> .mita/session.log"

[[hooks]]
event = "post_tool_call"
command = "echo '{tool}: {result}' >> .mita/tool.log"
```

### Fields

| Field     | Required | Description                                    |
|-----------|----------|------------------------------------------------|
| `event`   | Yes      | One of the lifecycle events listed above       |
| `command` | Yes      | Shell command to execute (supports variables)  |
| `match`   | No       | Glob pattern to filter by file path            |

## Example: Auto-lint on File Write

Run `ruff` automatically whenever Mita writes a Python file:

```bash
mita hooks add on_file_write "ruff check --fix {file_path}" --match "*.py"
```

Equivalent TOML:

```toml
[[hooks]]
event = "on_file_write"
command = "ruff check --fix {file_path}"
match = "*.py"
```

If the hook command exits with a non-zero status, Mita logs a warning but continues the session.
