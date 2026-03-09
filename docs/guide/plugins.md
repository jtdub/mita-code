# MCP Plugins

Mita supports extending its capabilities through the **Model Context Protocol (MCP)**. MCP is an open standard that lets AI assistants connect to external tools and data sources via lightweight server processes. Plugins run locally alongside Mita and expose additional tools that the agent can invoke during a session.

Mita supports two MCP transports:

- **stdio** -- the plugin runs as a subprocess, communicating over stdin/stdout (most common)
- **SSE** -- the plugin runs as an HTTP server using Server-Sent Events

## Adding a Plugin

Use `mita plugins add` to register a new stdio plugin:

```bash
mita plugins add filesystem --command "npx @modelcontextprotocol/server-filesystem /tmp"
```

This registers a plugin named `filesystem` that spawns the given command when activated.

For SSE-based plugins, provide a URL instead:

```bash
mita plugins add remote-db --url "http://localhost:8080/sse"
```

## Listing Plugins

```bash
mita plugins list
```

Displays all registered plugins, their transport type, and connection status.

## Testing a Plugin

Verify that a plugin starts correctly and exposes tools:

```bash
mita plugins test filesystem
```

This spawns the plugin process (or connects to the SSE endpoint), lists the tools it advertises, and shuts down.

## Removing a Plugin

```bash
mita plugins remove filesystem
```

## TOML Configuration

Plugins are stored in your project or global config as `[[plugins]]` entries:

```toml
[[plugins]]
name = "filesystem"
transport = "stdio"
command = "npx"
args = ["@modelcontextprotocol/server-filesystem", "/tmp"]

[[plugins]]
name = "remote-db"
transport = "sse"
url = "http://localhost:8080/sse"
```

### Fields

| Field       | Required | Description                                      |
|-------------|----------|--------------------------------------------------|
| `name`      | Yes      | Unique identifier for the plugin                 |
| `transport` | Yes      | `stdio` or `sse`                                 |
| `command`   | stdio    | Executable to run                                |
| `args`      | No       | List of arguments passed to the command           |
| `url`       | sse      | URL of the SSE endpoint                          |
| `env`       | No       | Table of environment variables set for the process|

### Environment Variables

Pass secrets or configuration to a plugin process via `env`:

```toml
[[plugins]]
name = "github"
transport = "stdio"
command = "npx"
args = ["@modelcontextprotocol/server-github"]
env = { GITHUB_TOKEN = "ghp_xxxx" }
```

## Example: Filesystem Plugin

The MCP filesystem server gives Mita tools for reading, writing, and searching files inside a sandboxed directory.

```bash
# Install and register
mita plugins add filesystem --command "npx @modelcontextprotocol/server-filesystem /tmp"

# Verify it works
mita plugins test filesystem
```

Once registered, start a chat session and the agent will automatically have access to the filesystem tools provided by the plugin.
