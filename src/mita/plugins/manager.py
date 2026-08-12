"""Plugin manager — manages all MCP plugin connections and tool registration."""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any

from rich.console import Console

from mita.config.schema import PluginDefinition
from mita.plugins.client import MCPPluginClient
from mita.tools.registry import ToolHandler, ToolRegistry
from mita.tools.schema import ToolDefinition, ToolParameter, ToolResult

_logger = logging.getLogger(__name__)


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
            )
        )
    return params


class PluginManager:
    """Manages MCP plugin connections and exposes their tools."""

    def __init__(self, plugins: list[PluginDefinition]) -> None:
        self._plugins = plugins
        self._clients: dict[str, MCPPluginClient] = {}

    @property
    def plugin_names(self) -> list[str]:
        return [p.name for p in self._plugins]

    def get_client(self, name: str) -> MCPPluginClient | None:
        return self._clients.get(name)

    async def start_all(self, console: Console | None = None) -> list[str]:
        """Start all configured plugins. Returns list of successfully started plugin names."""
        started: list[str] = []
        for plugin in self._plugins:
            try:
                client = MCPPluginClient(plugin)
                await client.connect()
                self._clients[plugin.name] = client
                started.append(plugin.name)
            except Exception as e:  # noqa: BLE001 - untrusted plugin boundary
                # The MCP SDK/anyio raise McpError, BrokenResourceError, and
                # ExceptionGroup, none of which subclass the connection errors we
                # used to catch. A misbehaving plugin must never abort startup.
                if console:
                    console.print(f"[yellow]Plugin '{plugin.name}' failed to start: {e}[/yellow]")
        return started

    async def stop_all(self) -> None:
        """Disconnect all running plugins."""
        for client in self._clients.values():
            try:
                await client.disconnect()
            except Exception:  # noqa: BLE001 - cleanup must not raise
                # stop_all runs in finally blocks; one failing disconnect must not
                # skip the others, leak subprocesses, or mask the original error.
                pass
        self._clients.clear()

    async def start_plugin(self, name: str, console: Console | None = None) -> bool:
        """Start a single plugin by name."""
        plugin = next((p for p in self._plugins if p.name == name), None)
        if plugin is None:
            if console:
                console.print(f"[red]Plugin '{name}' not found in configuration.[/red]")
            return False

        try:
            client = MCPPluginClient(plugin)
            await client.connect()
            self._clients[name] = client
            return True
        except Exception as e:  # noqa: BLE001 - untrusted plugin boundary
            if console:
                console.print(f"[red]Plugin '{name}' failed to start: {e}[/red]")
            return False

    async def stop_plugin(self, name: str) -> None:
        """Stop a single plugin by name."""
        client = self._clients.pop(name, None)
        if client is not None:
            await client.disconnect()

    async def list_tools(self, name: str | None = None) -> dict[str, list[dict[str, Any]]]:
        """List tools from connected plugins.

        Args:
            name: If provided, list tools from this plugin only.

        Returns:
            Dict mapping plugin name to list of tool dicts.
        """
        result: dict[str, list[dict[str, Any]]] = {}
        clients = {name: self._clients[name]} if name and name in self._clients else self._clients
        for pname, client in clients.items():
            try:
                result[pname] = await client.list_tools()
            except Exception:  # noqa: BLE001 - untrusted plugin boundary
                result[pname] = []
        return result

    async def register_tools(self, registry: ToolRegistry) -> int:
        """Register all MCP plugin tools into a ToolRegistry.

        Must be called after start_all(). Returns the number of tools registered.
        """
        count = 0

        for pname, client in self._clients.items():
            try:
                tools = await client.list_tools()
            except Exception:  # noqa: BLE001 - untrusted plugin boundary
                continue

            for tool in tools:
                tool_name = mcp_tool_name(pname, tool["name"])
                # A third-party plugin tool is treated as destructive (requires
                # confirmation) UNLESS it explicitly declares readOnlyHint=True.
                # Without this, plugin tools defaulted to non-destructive and ran
                # with no confirmation at all (audit finding C2).
                read_only = tool.get("readOnlyHint") is True
                definition = ToolDefinition(
                    name=tool_name,
                    description=tool["description"],
                    parameters=_schema_to_parameters(tool.get("inputSchema", {})),
                    source=f"mcp:{pname}",
                    destructive=not read_only,
                )
                handler = _make_mcp_handler(client, tool["name"])
                registry.register(definition, handler)
                count += 1

        return count

    async def test_plugin(self, name: str) -> dict[str, Any]:
        """Test plugin connectivity. Returns status dict."""
        client = self._clients.get(name)
        if client is None:
            return {"name": name, "connected": False, "error": "Not started"}

        alive = await client.ping()
        if not alive:
            return {"name": name, "connected": True, "ping": False, "error": "Ping failed"}

        try:
            tools = await client.list_tools()
            return {
                "name": name,
                "connected": True,
                "ping": True,
                "tools": len(tools),
                "tool_names": [t["name"] for t in tools],
            }
        except Exception as e:  # noqa: BLE001 - untrusted plugin boundary
            return {"name": name, "connected": True, "ping": True, "error": str(e)}


def _make_mcp_handler(client: MCPPluginClient, remote_tool_name: str) -> ToolHandler:
    """Create a tool handler that dispatches to an MCP plugin."""

    async def handler(args: dict[str, Any]) -> ToolResult:
        try:
            output = await client.call_tool(remote_tool_name, args)
            return ToolResult(tool_call_id="", success=True, output=output)
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
