# Chat & Ask

## Interactive Chat

```bash
mita chat
```

Opens an interactive REPL with the agentic assistant. The agent has access to tools for reading/writing files, running shell commands, searching code, and more.

### Tool Execution

The agent can invoke tools automatically. Destructive actions (file writes, shell commands) require your confirmation before executing. Read-only tools like file reads, glob, and grep are auto-approved by default.

Configure auto-approval in your config:

```toml
[tools]
auto_approve = ["file_read", "glob", "grep"]
confirm_destructive = true
```

### Streaming

Responses stream token-by-token by default. Disable with:

```toml
[ui]
stream = false
```

## Single-Shot Ask

```bash
mita ask "your prompt here"
```

Sends a single prompt and prints the response. Useful for scripting and piping:

```bash
cat error.log | mita ask "what went wrong?"
git diff | mita ask "review these changes"
mita ask "write a pytest for src/auth.py" > test_auth.py
```
