"""Plugin manager — connects MCP servers and registers their tools as LangChain tools."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
from contextlib import AsyncExitStack
from typing import Any, cast

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from rich.console import Console

from mita.config.schema import PluginDefinition
from mita.tools.registry import ToolHandler, ToolRegistry
from mita.tools.schema import ToolDefinition, ToolParameter, ToolResult

_logger = logging.getLogger(__name__)

_MCP_CALL_TIMEOUT = 120.0
"""Seconds a single MCP tool call may take before it is abandoned."""

_MCP_PING_TIMEOUT = 10.0
"""Seconds a plugin health-check ping may take before it is abandoned."""

_ENV_ALLOWLIST: frozenset[str] = frozenset(
    {"PATH", "HOME", "USER", "LOGNAME", "SHELL", "LANG", "LC_ALL", "TERM", "TMPDIR", "PATHEXT"}
)
"""The only variables a plugin subprocess inherits.

Everything else (tokens, keys, agent sockets) is dropped. Mita owns this list so
that the set does not change with the version of a transitive package.
"""


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
    """Convert a JSON Schema inputSchema to a list of ToolParameter.

    Each parameter keeps its full property schema, so enum, items, and nested
    objects survive.
    """
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
                json_schema=prop if isinstance(prop, dict) else None,
            )
        )
    return params


class PluginManager:
    """Manages MCP plugin connections and exposes their tools.

    One MCP session is held open per plugin for its lifetime, so a stdio server
    keeps its process and any in-process state between tool calls. Sessions are
    closed by ``stop_all()`` and ``stop_plugin()``.
    """

    def __init__(self, plugins: list[PluginDefinition]) -> None:
        self._plugins = plugins
        self._client: MultiServerMCPClient | None = None
        self._exit_stacks: dict[str, AsyncExitStack] = {}
        self._sessions: dict[str, Any] = {}
        self._tools: dict[str, list[BaseTool]] = {}

    @property
    def plugin_names(self) -> list[str]:
        return [p.name for p in self._plugins]

    async def start_all(self, console: Console | None = None) -> list[str]:
        """Start every configured plugin. Return the names that started.

        One misconfigured or unreachable plugin must never abort startup.
        """
        started: list[str] = []
        for plugin in self._plugins:
            if await self.start_plugin(plugin.name, console):
                started.append(plugin.name)
        return started

    async def stop_all(self) -> None:
        """Disconnect all running plugins.

        A failing close must not mask the error a caller was propagating, so each
        plugin's session is closed in its own guarded block.
        """
        self._tools.clear()
        self._sessions.clear()
        self._client = None
        stacks = self._exit_stacks
        self._exit_stacks = {}
        for name, stack in stacks.items():
            try:
                await stack.aclose()
            except Exception:  # noqa: BLE001 - cleanup must not mask the original error
                _logger.warning("Error closing plugin '%s' session", name, exc_info=True)

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
            await self._connect_plugin(name)
            return True
        except Exception as e:  # noqa: BLE001 - untrusted plugin boundary
            # A failed start must not leave a connection with no session.
            if self._client is not None:
                self._client.connections.pop(name, None)
            if console:
                console.print(f"[yellow]Plugin '{name}' failed to start: {e}[/yellow]")
            return False

    async def stop_plugin(self, name: str) -> None:
        """Stop a single plugin by name."""
        self._tools.pop(name, None)
        self._sessions.pop(name, None)
        stack = self._exit_stacks.pop(name, None)
        if stack is not None:
            try:
                await stack.aclose()
            except Exception:  # noqa: BLE001 - cleanup must not mask the original error
                _logger.warning("Error closing plugin '%s' session", name, exc_info=True)

    async def _connect_plugin(self, name: str) -> None:
        """Open a session for one plugin and load its tools bound to that session."""
        if self._client is None:
            raise RuntimeError("a client must exist before connecting a plugin")
        stack = AsyncExitStack()
        session = await stack.enter_async_context(self._client.session(name))
        self._sessions[name] = session
        self._exit_stacks[name] = stack
        self._tools[name] = await load_mcp_tools(
            session,
            callbacks=self._client.callbacks,
            server_name=name,
            handle_tool_errors=False,
        )

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

        A third-party plugin tool counts as destructive, and so needs confirmation,
        unless the server declares readOnlyHint=True (audit finding C2).
        """
        count = 0
        for pname, tools in self._tools.items():
            for tool in tools:
                definition = ToolDefinition(
                    name=mcp_tool_name(pname, tool.name),
                    description=tool.description or "",
                    parameters=_schema_to_parameters(_tool_schema(tool)),
                    source=f"mcp:{pname}",
                    destructive=not _read_only(tool),
                )
                registry.register(definition, _make_mcp_handler(tool))
                count += 1
        return count

    async def test_plugin(self, name: str) -> dict[str, Any]:
        """Test plugin connectivity. Returns status dict."""
        session = self._sessions.get(name)
        if session is None:
            return {"name": name, "connected": False, "error": "Not started"}

        try:
            await asyncio.wait_for(session.send_ping(), timeout=_MCP_PING_TIMEOUT)
        except Exception as e:  # noqa: BLE001 - untrusted plugin boundary
            return {"name": name, "connected": True, "ping": False, "error": str(e)}

        tools = self._tools.get(name, [])
        return {
            "name": name,
            "connected": True,
            "ping": True,
            "tools": len(tools),
            "tool_names": [t.name for t in tools],
        }


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
    """Return the allowed environment for a plugin subprocess, plus its declared vars.

    A plugin subprocess must not inherit the full parent environment. That leaks
    credentials (GITHUB_TOKEN, AWS_*, SSH sockets) to every third-party plugin
    (audit finding C1).
    """
    return {**_minimal_env(), **plugin.env}


def _minimal_env() -> dict[str, str]:
    """Return a minimal environment allowlist for plugin subprocesses."""
    return {key: os.environ[key] for key in _ENV_ALLOWLIST if key in os.environ}


def _read_only(tool: BaseTool) -> bool:
    """Return True when the MCP server declared the tool read-only."""
    metadata = tool.metadata or {}
    return metadata.get("readOnlyHint") is True


def _tool_schema(tool: BaseTool) -> dict[str, Any]:
    """Return the JSON Schema for a LangChain tool's arguments.

    langchain-mcp-adapters passes the raw JSON-Schema dict through, not a Pydantic
    model, so both shapes must work.
    """
    args_schema: Any = tool.args_schema
    if args_schema is None:
        return {}
    if isinstance(args_schema, dict):
        return args_schema
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


def _make_mcp_handler(tool: BaseTool) -> ToolHandler:
    """Create a tool handler that dispatches to a LangChain MCP tool.

    A failing plugin tool must not abort the turn, so every error becomes a
    failed ToolResult.
    """

    async def handler(args: dict[str, Any]) -> ToolResult:
        try:
            raw = await asyncio.wait_for(tool.ainvoke(args), timeout=_MCP_CALL_TIMEOUT)
            return ToolResult(tool_call_id="", success=True, output=_output_text(raw))
        except TimeoutError:
            return ToolResult(
                tool_call_id="",
                success=False,
                error=f"MCP tool '{tool.name}' timed out",
            )
        except Exception as e:  # noqa: BLE001
            return ToolResult(
                tool_call_id="",
                success=False,
                error=f"MCP tool '{tool.name}' failed: {e}",
            )

    return handler


def _output_text(raw: Any) -> str:
    """Convert a LangChain tool result (content or content/artifact tuple) to text.

    A block without text (an image, a resource link) becomes a marker so the
    agent still sees that the tool produced output.
    """
    if isinstance(raw, tuple):
        raw = raw[0]
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        parts: list[str] = []
        for block in raw:
            if not isinstance(block, dict):
                parts.append(str(block))
                continue
            text = block.get("text")
            if text:
                parts.append(text)
            elif block.get("type") == "base64":
                parts.append(f"[binary data: {block.get('mime_type', 'unknown')}]")
            elif block.get("type") == "url":
                parts.append(f"[resource: {block.get('url', '')}]")
            else:
                parts.append(str(block))
        return "\n".join(parts) if parts else str(raw)
    return str(raw)
