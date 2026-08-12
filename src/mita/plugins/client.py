"""MCP client wrapper for a single plugin server."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from mita.config.schema import PluginDefinition

_mcp_default_env: Callable[[], dict[str, str]] | None
try:
    # Minimal, safe environment allowlist (PATH/HOME/etc.) provided by the SDK.
    from mcp.client.stdio import get_default_environment as _mcp_default_env
except ImportError:  # pragma: no cover - depends on mcp version
    _mcp_default_env = None

_logger = logging.getLogger(__name__)

# Only these variables are passed through to plugin subprocesses when the SDK
# helper is unavailable. Everything else (tokens, keys, agent sockets) is dropped.
_ENV_ALLOWLIST: frozenset[str] = frozenset(
    {"PATH", "HOME", "USER", "LOGNAME", "SHELL", "LANG", "LC_ALL", "TERM", "TMPDIR", "PATHEXT"}
)


def _minimal_env() -> dict[str, str]:
    """Return a minimal environment allowlist for plugin subprocesses."""
    return {key: os.environ[key] for key in _ENV_ALLOWLIST if key in os.environ}


class MCPPluginClient:
    """Manages connection to a single MCP server."""

    def __init__(self, plugin: PluginDefinition) -> None:
        self._plugin = plugin
        self._session: ClientSession | None = None
        self._exit_stack: AsyncExitStack | None = None

    @property
    def name(self) -> str:
        return self._plugin.name

    @property
    def transport(self) -> str:
        return self._plugin.transport

    @property
    def connected(self) -> bool:
        return self._session is not None

    async def connect(self, timeout: float = 30.0) -> None:
        """Connect to the MCP server and initialize the session."""
        if self._session is not None:
            return

        self._exit_stack = AsyncExitStack()

        try:
            if self._plugin.transport == "stdio":
                await self._connect_stdio(timeout)
            elif self._plugin.transport == "sse":
                await self._connect_sse(timeout)
            else:
                raise ValueError(f"Unsupported transport: {self._plugin.transport}")
        except Exception:
            # Clean up resources on connection failure
            await self._exit_stack.aclose()
            self._exit_stack = None
            raise

    async def _connect_stdio(self, timeout: float) -> None:
        """Connect via stdio transport (subprocess)."""
        if self._exit_stack is None:
            raise RuntimeError("connect() must be called before _connect_stdio()")

        if not self._plugin.command:
            raise ValueError(f"Plugin '{self.name}' requires a command for stdio transport")

        # Build environment from a minimal safe allowlist, then overlay the
        # plugin's own declared vars. Do NOT inherit the full parent environment:
        # that leaked credentials (GITHUB_TOKEN, AWS_*, SSH sockets) to every
        # third-party plugin subprocess (audit finding C1).
        base_env = _mcp_default_env() if _mcp_default_env is not None else _minimal_env()
        env: dict[str, str] = {**base_env, **self._plugin.env}

        server_params = StdioServerParameters(
            command=self._plugin.command,
            args=self._plugin.args,
            env=env,
        )

        transport = await self._exit_stack.enter_async_context(stdio_client(server_params))
        read_stream, write_stream = transport

        session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await asyncio.wait_for(session.initialize(), timeout=timeout)
        self._session = session

    async def _connect_sse(self, timeout: float) -> None:
        """Connect via SSE transport (HTTP)."""
        if self._exit_stack is None:
            raise RuntimeError("connect() must be called before _connect_sse()")

        if not self._plugin.url:
            raise ValueError(f"Plugin '{self.name}' requires a url for SSE transport")

        from mcp.client.sse import sse_client

        transport = await self._exit_stack.enter_async_context(sse_client(self._plugin.url))
        read_stream, write_stream = transport

        session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await asyncio.wait_for(session.initialize(), timeout=timeout)
        self._session = session

    async def disconnect(self) -> None:
        """Disconnect from the MCP server and clean up resources."""
        if self._exit_stack is not None:
            await self._exit_stack.aclose()
            self._exit_stack = None
        self._session = None

    async def list_tools(self) -> list[dict[str, Any]]:
        """List tools provided by this MCP server.

        Returns a list of dicts with name, description, and inputSchema.
        """
        if self._session is None:
            raise RuntimeError(f"Plugin '{self.name}' is not connected")

        response = await self._session.list_tools()
        tools: list[dict[str, Any]] = []
        for tool in response.tools:
            annotations = getattr(tool, "annotations", None)
            read_only = getattr(annotations, "readOnlyHint", None) if annotations else None
            tools.append(
                {
                    "name": tool.name,
                    "description": tool.description or "",
                    "inputSchema": tool.inputSchema,
                    "readOnlyHint": read_only,
                }
            )
        return tools

    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any], timeout: float = 120.0
    ) -> str:
        """Call a tool on this MCP server.

        Returns the text content from the tool result.
        """
        if self._session is None:
            raise RuntimeError(f"Plugin '{self.name}' is not connected")

        result = await asyncio.wait_for(
            self._session.call_tool(tool_name, arguments),
            timeout=timeout,
        )

        # Extract text content from result
        parts: list[str] = []
        for item in result.content:
            if hasattr(item, "text"):
                parts.append(item.text)
            elif hasattr(item, "data"):
                parts.append(f"[binary data: {getattr(item, 'mimeType', 'unknown')}]")
            else:
                parts.append(str(item))
        text = "\n".join(parts)

        # Honor the MCP isError flag: a tool that failed must not be reported to the
        # agent as a success. Raise so the handler produces a failed ToolResult.
        # `is True` (not truthiness) — isError is a spec bool, and this stays correct
        # when the result is a test MagicMock whose attributes auto-create as truthy.
        if getattr(result, "isError", False) is True:
            raise RuntimeError(text or f"MCP tool '{tool_name}' reported an error")
        return text

    async def ping(self, timeout: float = 10.0) -> bool:
        """Ping the MCP server to check connectivity."""
        if self._session is None:
            return False
        try:
            await asyncio.wait_for(self._session.send_ping(), timeout=timeout)
            return True
        except (TimeoutError, ConnectionError, OSError):
            return False
