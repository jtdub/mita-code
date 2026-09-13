"""Plugin manager — connects MCP servers and registers their tools as LangChain tools."""

from __future__ import annotations

import hashlib
import logging
import os
import re
from typing import Any, cast

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from rich.console import Console

from mita.config.schema import PluginDefinition
from mita.tools.registry import ToolHandler, ToolRegistry
from mita.tools.schema import ToolDefinition, ToolParameter, ToolResult

_logger = logging.getLogger(__name__)

_mcp_default_env: Any | None
try:
    # Minimal, safe environment allowlist (PATH/HOME/etc.) provided by the MCP SDK.
    from mcp.client.stdio import get_default_environment as _mcp_default_env
except ImportError:  # pragma: no cover - depends on mcp version
    _mcp_default_env = None

# Only these variables are passed through to plugin subprocesses when the SDK
# helper is unavailable. Everything else (tokens, keys, agent sockets) is dropped.
_ENV_ALLOWLIST: frozenset[str] = frozenset(
    {"PATH", "HOME", "USER", "LOGNAME", "SHELL", "LANG", "LC_ALL", "TERM", "TMPDIR", "PATHEXT"}
)


def mcp_tool_name(plugin: str, tool: str) -> str:
    """Build a registry/function name for an MCP tool.

    OpenAI-compatible function names must match ^[a-zA-Z0-9_-]{1,64}$, so the old
    ``mcp:{plugin}/{tool}`` scheme (with ':' and '/') was rejected by strict
    providers and failed the whole request. Sanitize to underscores and, if the
    name would exceed 64 chars, append a short hash of the original (finding: MCP
    tool-name charset).
    """
    raw = f"mcp_{plugin}_{tool}"
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", raw)
    if len(safe) > 64:
        digest = hashlib.sha1(raw.encode()).hexdigest()[:8]
        safe = f"{safe[:55]}_{digest}"
    return safe


def _schema_to_parameters(input_schema: dict[str, Any]) -> list[ToolParameter]:
    """Convert a JSON Schema inputSchema to a list of ToolParameter."""
    properties = input_schema.get("properties", {})
    required = set(input_schema.get("required", []))
    params: list[ToolParameter] = []

    for name, prop in properties.items():
        params.append(
            ToolParameter(
                name=name,
                type=prop.get("type", "string"),
                description=prop.get("description", ""),
                required=name in required,
                default=prop.get("default"),
                # Preserve the full property schema (enum/items/nested) for MCP tools.
                json_schema=prop if isinstance(prop, dict) else None,
            )
        )
    return params


class PluginManager:
    """Manages MCP plugin connections and exposes their tools."""

    def __init__(self, plugins: list[PluginDefinition]) -> None:
        self._plugins = plugins
        self._client: MultiServerMCPClient | None = None
        self._tools: dict[str, list[BaseTool]] = {}

    @property
    def plugin_names(self) -> list[str]:
        return [p.name for p in self._plugins]

    async def start_all(self, console: Console | None = None) -> list[str]:
        """Start all configured plugins. Returns list of successfully started plugin names."""
        connections: dict[str, dict[str, Any]] = {}
        for plugin in self._plugins:
            try:
                connections[plugin.name] = _build_connection(plugin)
            except Exception as e:  # noqa: BLE001 - untrusted plugin boundary
                # A misconfigured plugin must never abort startup.
                if console:
                    console.print(f"[yellow]Plugin '{plugin.name}' failed to start: {e}[/yellow]")

        if not connections:
            self._client = None
            return []

        self._client = MultiServerMCPClient(cast(Any, connections), handle_tool_errors=False)

        started: list[str] = []
        for name in connections:
            try:
                self._tools[name] = await self._client.get_tools(server_name=name)
                started.append(name)
            except Exception as e:  # noqa: BLE001 - untrusted plugin boundary
                if console:
                    console.print(f"[yellow]Plugin '{name}' failed to start: {e}[/yellow]")
        return started

    async def stop_all(self) -> None:
        """Disconnect all running plugins."""
        self._tools.clear()
        self._client = None

    async def start_plugin(self, name: str, console: Console | None = None) -> bool:
        """Start a single plugin by name."""
        plugin = next((p for p in self._plugins if p.name == name), None)
        if plugin is None:
            if console:
                console.print(f"[red]Plugin '{name}' not found in configuration.[/red]")
            return False

        try:
            connection = _build_connection(plugin)
            if self._client is None:
                self._client = MultiServerMCPClient({}, handle_tool_errors=False)
            self._client.connections[name] = cast(Any, connection)
            self._tools[name] = await self._client.get_tools(server_name=name)
            return True
        except Exception as e:  # noqa: BLE001 - untrusted plugin boundary
            if console:
                console.print(f"[red]Plugin '{name}' failed to start: {e}[/red]")
            return False

    async def stop_plugin(self, name: str) -> None:
        """Stop a single plugin by name."""
        self._tools.pop(name, None)

    async def list_tools(self, name: str | None = None) -> dict[str, list[dict[str, Any]]]:
        """List tools from connected plugins.

        Args:
            name: If provided, list tools from this plugin only.

        Returns:
            Dict mapping plugin name to list of tool dicts.
        """
        targets = [name] if name and name in self._tools else list(self._tools)
        return {pname: [_tool_info(tool) for tool in self._tools[pname]] for pname in targets}

    async def register_tools(self, registry: ToolRegistry) -> int:
        """Register all MCP plugin tools into a ToolRegistry.

        Must be called after start_all(). Returns the number of tools registered.
        """
        count = 0
        for pname, tools in self._tools.items():
            for tool in tools:
                tool_name = mcp_tool_name(pname, tool.name)
                # A third-party plugin tool is treated as destructive (requires
                # confirmation) UNLESS it explicitly declares readOnlyHint=True.
                # Without this, plugin tools defaulted to non-destructive and ran
                # with no confirmation at all (audit finding C2).
                read_only = _read_only(tool)
                definition = ToolDefinition(
                    name=tool_name,
                    description=tool.description or "",
                    parameters=_schema_to_parameters(_tool_schema(tool)),
                    source=f"mcp:{pname}",
                    destructive=not read_only,
                )
                handler = _make_mcp_handler(tool, tool.name)
                registry.register(definition, handler)
                count += 1
        return count

    async def test_plugin(self, name: str) -> dict[str, Any]:
        """Test plugin connectivity. Returns status dict."""
        if name not in self._tools or self._client is None:
            return {"name": name, "connected": False, "error": "Not started"}

        try:
            tools = await self._client.get_tools(server_name=name)
            return {
                "name": name,
                "connected": True,
                "ping": True,
                "tools": len(tools),
                "tool_names": [t.name for t in tools],
            }
        except Exception as e:  # noqa: BLE001 - untrusted plugin boundary
            return {"name": name, "connected": True, "ping": False, "error": str(e)}


def _build_connection(plugin: PluginDefinition) -> dict[str, Any]:
    """Build a langchain-mcp-adapters connection dict for a plugin."""
    if plugin.transport == "stdio":
        if not plugin.command:
            raise ValueError(f"Plugin '{plugin.name}' requires a command for stdio transport")
        return {
            "transport": "stdio",
            "command": plugin.command,
            "args": list(plugin.args),
            "env": _plugin_env(plugin),
        }
    if plugin.transport in ("sse", "streamable_http"):
        if not plugin.url:
            raise ValueError(
                f"Plugin '{plugin.name}' requires a url for {plugin.transport} transport"
            )
        return {
            "transport": plugin.transport,
            "url": plugin.url,
            "headers": plugin.headers or None,
        }
    raise ValueError(f"Unsupported transport: {plugin.transport}")


def _plugin_env(plugin: PluginDefinition) -> dict[str, str]:
    """Return a safe env allowlist for a plugin subprocess, plus its declared vars.

    Do NOT inherit the full parent environment: that leaks credentials
    (GITHUB_TOKEN, AWS_*, SSH sockets) to every third-party plugin subprocess
    (audit finding C1).
    """
    base = _mcp_default_env() if _mcp_default_env is not None else _minimal_env()
    return {**base, **plugin.env}


def _minimal_env() -> dict[str, str]:
    """Return a minimal environment allowlist for plugin subprocesses."""
    return {key: os.environ[key] for key in _ENV_ALLOWLIST if key in os.environ}


def _read_only(tool: BaseTool) -> bool:
    """Return True when the MCP server declared the tool read-only."""
    metadata = tool.metadata or {}
    return metadata.get("readOnlyHint") is True


def _tool_schema(tool: BaseTool) -> dict[str, Any]:
    """Return the JSON Schema for a LangChain tool's arguments."""
    args_schema: Any = tool.args_schema
    if args_schema is None:
        return {}
    schema: dict[str, Any] = args_schema.model_json_schema()
    return schema


def _tool_info(tool: BaseTool) -> dict[str, Any]:
    """Describe a tool for `mita plugins list`."""
    return {
        "name": tool.name,
        "description": tool.description or "",
        "inputSchema": _tool_schema(tool),
        "readOnlyHint": _read_only(tool),
    }


def _make_mcp_handler(tool: BaseTool, remote_tool_name: str) -> ToolHandler:
    """Create a tool handler that dispatches to a LangChain MCP tool."""

    async def handler(args: dict[str, Any]) -> ToolResult:
        try:
            raw = await tool.ainvoke(args)
            return ToolResult(tool_call_id="", success=True, output=_output_text(raw))
        except TimeoutError:
            return ToolResult(
                tool_call_id="",
                success=False,
                error=f"MCP tool '{remote_tool_name}' timed out",
            )
        except Exception as e:  # noqa: BLE001 - a failing plugin tool must not abort the turn
            return ToolResult(
                tool_call_id="",
                success=False,
                error=f"MCP tool '{remote_tool_name}' failed: {e}",
            )

    return handler


def _output_text(raw: Any) -> str:
    """Convert a LangChain tool result (content or content/artifact tuple) to text."""
    if isinstance(raw, tuple):
        raw = raw[0]
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        parts = [block.get("text", "") for block in raw if isinstance(block, dict)]
        if parts:
            return "\n".join(parts)
    return str(raw)
