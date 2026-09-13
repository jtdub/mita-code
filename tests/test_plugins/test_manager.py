"""Tests for MCP plugin manager (LangChain-backed)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field

from mita.config.schema import PluginDefinition
from mita.plugins.manager import (
    PluginManager,
    _make_mcp_handler,
    _schema_to_parameters,
    mcp_tool_name,
)
from mita.tools.registry import ToolRegistry
from mita.tools.schema import ToolResult


class _Args(BaseModel):
    path: str = Field(description="File path")


class _AnyArgs(BaseModel):
    """Accept any keyword input, so a tool can run with arbitrary arguments."""

    model_config = ConfigDict(extra="allow")


def _tool(
    name: str = "read_file",
    description: str = "Read a file",
    output: str = "done",
    read_only: bool | None = None,
    args_schema: type[BaseModel] | None = None,
) -> StructuredTool:
    """Build a StructuredTool with the surface mita reads."""

    async def _run(**kwargs: Any) -> str:
        return output

    kwargs: dict[str, Any] = {
        "name": name,
        "description": description,
        "args_schema": args_schema or _AnyArgs,
        "coroutine": _run,
    }
    if read_only is not None:
        kwargs["metadata"] = {"readOnlyHint": read_only}
    return StructuredTool(**kwargs)


def _mock_client() -> MagicMock:
    """Return a MultiServerMCPClient mock whose sessions yield mock sessions."""
    client = AsyncMock()

    def _session_cm(_name: str) -> AsyncMock:
        cm = AsyncMock()
        cm.__aenter__ = AsyncMock(return_value=AsyncMock())
        cm.__aexit__ = AsyncMock(return_value=False)
        return cm

    client.session = MagicMock(side_effect=_session_cm)
    return client


class TestSchemaToParameters:
    def test_empty_schema(self) -> None:
        params = _schema_to_parameters({})
        assert params == []

    def test_basic_properties(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path"},
                "lines": {"type": "integer", "description": "Number of lines"},
            },
            "required": ["path"],
        }
        params = _schema_to_parameters(schema)
        assert len(params) == 2
        path_param = next(p for p in params if p.name == "path")
        assert path_param.type == "string"
        assert path_param.required is True
        lines_param = next(p for p in params if p.name == "lines")
        assert lines_param.required is False

    def test_default_values(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "encoding": {"type": "string", "description": "Encoding", "default": "utf-8"},
            },
        }
        params = _schema_to_parameters(schema)
        assert params[0].default == "utf-8"


class TestMakeMCPHandler:
    @pytest.mark.asyncio
    async def test_success(self) -> None:
        tool = _tool(name="read_file", output="tool output")
        handler = _make_mcp_handler(tool)
        result = await handler({"path": "/tmp/test"})

        assert isinstance(result, ToolResult)
        assert result.success is True
        assert result.output == "tool output"

    @pytest.mark.asyncio
    async def test_timeout(self) -> None:
        async def _boom(**kwargs: Any) -> str:
            raise TimeoutError

        tool = StructuredTool(
            name="slow_tool", description="s", args_schema=_AnyArgs, coroutine=_boom
        )
        handler = _make_mcp_handler(tool)
        result = await handler({})

        assert result.success is False
        assert "timed out" in (result.error or "")

    @pytest.mark.asyncio
    async def test_connection_error(self) -> None:
        async def _boom(**kwargs: Any) -> str:
            raise ConnectionError("dead")

        tool = StructuredTool(
            name="broken_tool", description="b", args_schema=_AnyArgs, coroutine=_boom
        )
        handler = _make_mcp_handler(tool)
        result = await handler({})

        assert result.success is False
        assert "failed" in (result.error or "")


class TestPluginManager:
    @pytest.fixture
    def plugins(self) -> list[PluginDefinition]:
        return [
            PluginDefinition(name="fs", transport="stdio", command="npx", args=["@mcp/fs"]),
            PluginDefinition(name="web", transport="sse", url="http://localhost:3001/sse"),
        ]

    def test_plugin_names(self, plugins: list[PluginDefinition]) -> None:
        mgr = PluginManager(plugins)
        assert mgr.plugin_names == ["fs", "web"]

    @pytest.mark.asyncio
    async def test_start_all(self) -> None:
        plugins = [PluginDefinition(name="fs", transport="stdio", command="echo")]
        mgr = PluginManager(plugins)
        with (
            patch("mita.plugins.manager.MultiServerMCPClient", return_value=_mock_client()),
            patch(
                "mita.plugins.manager.load_mcp_tools",
                new_callable=AsyncMock,
                return_value=[_tool(name="read", output="x")],
            ),
        ):
            started = await mgr.start_all()
        assert started == ["fs"]
        assert len(mgr._tools["fs"]) == 1
        assert "fs" in mgr._sessions

    @pytest.mark.asyncio
    async def test_start_all_with_failures(self) -> None:
        plugins = [PluginDefinition(name="bad", transport="stdio")]
        mgr = PluginManager(plugins)
        started = await mgr.start_all()
        assert started == []
        assert mgr._tools == {}

    @pytest.mark.asyncio
    async def test_start_all_tool_load_failure(self) -> None:
        plugins = [PluginDefinition(name="fs", transport="stdio", command="echo")]
        mgr = PluginManager(plugins)
        with (
            patch("mita.plugins.manager.MultiServerMCPClient", return_value=_mock_client()),
            patch(
                "mita.plugins.manager.load_mcp_tools",
                new_callable=AsyncMock,
                side_effect=ConnectionError("down"),
            ),
        ):
            started = await mgr.start_all()
        assert started == []
        assert "fs" not in mgr._tools

    @pytest.mark.asyncio
    async def test_stop_all(self) -> None:
        mgr = PluginManager([])
        mgr._tools["test"] = [_tool(name="x")]
        mgr._sessions["test"] = AsyncMock()
        exit_stack = AsyncMock()
        exit_stack.aclose = AsyncMock()
        mgr._exit_stack = exit_stack

        await mgr.stop_all()
        assert mgr._tools == {}
        assert mgr._sessions == {}
        assert mgr._client is None
        assert mgr._exit_stack is None
        exit_stack.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_list_tools(self) -> None:
        mgr = PluginManager([])
        mgr._tools["fs"] = [_tool(name="read", description="Read file")]

        result = await mgr.list_tools()
        assert "fs" in result
        assert len(result["fs"]) == 1
        assert result["fs"][0]["name"] == "read"

    @pytest.mark.asyncio
    async def test_list_tools_single_plugin(self) -> None:
        mgr = PluginManager([])
        mgr._tools["fs"] = [_tool(name="read")]
        mgr._tools["web"] = [_tool(name="search")]

        result = await mgr.list_tools("fs")
        assert "fs" in result
        assert "web" not in result

    @pytest.mark.asyncio
    async def test_register_tools(self) -> None:
        mgr = PluginManager([])
        mgr._tools["fs"] = [_tool(name="read_file", description="Read a file", args_schema=_Args)]

        registry = ToolRegistry()
        count = await mgr.register_tools(registry)

        assert count == 1
        assert registry.has_tool("mcp_fs_read_file")
        defn = registry.get_definition("mcp_fs_read_file")
        assert defn is not None
        assert defn.source == "mcp:fs"
        assert len(defn.parameters) == 1

    @pytest.mark.asyncio
    async def test_register_tools_dict_args_schema(self) -> None:
        """langchain-mcp-adapters passes the inputSchema dict through (finding #2)."""

        async def _run(**kwargs: Any) -> str:
            return "ok"

        dict_tool = StructuredTool(
            name="read",
            description="Read",
            args_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
            coroutine=_run,
        )
        mgr = PluginManager([])
        mgr._tools["fs"] = [dict_tool]

        registry = ToolRegistry()
        count = await mgr.register_tools(registry)

        assert count == 1
        defn = registry.get_definition("mcp_fs_read")
        assert defn is not None
        assert len(defn.parameters) == 1
        assert defn.parameters[0].name == "path"

    @pytest.mark.asyncio
    async def test_test_plugin_not_started(self) -> None:
        mgr = PluginManager([])
        result = await mgr.test_plugin("missing")
        assert result["connected"] is False

    @pytest.mark.asyncio
    async def test_test_plugin_healthy(self) -> None:
        mgr = PluginManager([])
        mgr._tools["fs"] = [_tool(name="tool1")]
        session = AsyncMock()
        session.send_ping = AsyncMock(return_value=None)
        mgr._sessions["fs"] = session

        result = await mgr.test_plugin("fs")
        assert result["ping"] is True
        assert result["tools"] == 1
        session.send_ping.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_test_plugin_ping_failed(self) -> None:
        mgr = PluginManager([])
        mgr._tools["fs"] = [_tool(name="tool1")]
        session = AsyncMock()
        session.send_ping = AsyncMock(side_effect=ConnectionError("down"))
        mgr._sessions["fs"] = session

        result = await mgr.test_plugin("fs")
        assert result["ping"] is False

    @pytest.mark.asyncio
    async def test_start_plugin(self) -> None:
        plugin = PluginDefinition(name="test", transport="stdio", command="echo")
        mgr = PluginManager([plugin])
        with (
            patch("mita.plugins.manager.MultiServerMCPClient", return_value=_mock_client()),
            patch(
                "mita.plugins.manager.load_mcp_tools",
                new_callable=AsyncMock,
                return_value=[_tool(name="x")],
            ),
        ):
            result = await mgr.start_plugin("test")
        assert result is True
        assert "test" in mgr._tools
        assert "test" in mgr._sessions

    @pytest.mark.asyncio
    async def test_start_plugin_not_configured(self) -> None:
        mgr = PluginManager([])
        result = await mgr.start_plugin("missing")
        assert result is False

    @pytest.mark.asyncio
    async def test_stop_plugin(self) -> None:
        mgr = PluginManager([])
        mgr._tools["test"] = [_tool(name="x")]
        mgr._sessions["test"] = AsyncMock()

        await mgr.stop_plugin("test")
        assert "test" not in mgr._tools
        assert "test" not in mgr._sessions

    @pytest.mark.asyncio
    async def test_tool_execution_via_registry(self) -> None:
        """End-to-end: register MCP tool, then execute via registry."""
        mgr = PluginManager([])
        mgr._tools["greeter"] = [
            _tool(name="greet", description="Greet someone", output="Hello, World!")
        ]

        registry = ToolRegistry()
        await mgr.register_tools(registry)

        from mita.tools.schema import ToolCall

        result = await registry.execute(
            ToolCall(id="tc1", name="mcp_greeter_greet", arguments={"name": "World"})
        )
        assert result.success is True
        assert result.output == "Hello, World!"


class TestMCPToolConfirmation:
    """Audit finding C2: plugin tools must pass through the confirmation gate."""

    @pytest.mark.asyncio
    async def test_plugin_tool_defaults_destructive_and_confirms(self) -> None:
        from mita.config.schema import ToolSettings
        from mita.tools.executor import execute_tool
        from mita.tools.schema import ToolCall

        mgr = PluginManager([])
        mgr._tools["srv"] = [_tool(name="do_thing", output="done", read_only=None)]
        registry = ToolRegistry()
        await mgr.register_tools(registry)

        definition = registry.get_definition("mcp_srv_do_thing")
        assert definition is not None
        assert definition.destructive is True

        prompts: list[str] = []

        async def confirm(prompt: str) -> bool:
            prompts.append(prompt)
            return False

        result = await execute_tool(
            ToolCall(id="1", name="mcp_srv_do_thing", arguments={}),
            registry,
            ToolSettings(),
            confirm_fn=confirm,
        )
        assert prompts, "plugin tool must ask for confirmation"
        assert result.success is False
        assert "denied" in (result.error or "").lower()

    @pytest.mark.asyncio
    async def test_read_only_plugin_tool_skips_confirmation(self) -> None:
        from mita.config.schema import ToolSettings
        from mita.tools.executor import execute_tool
        from mita.tools.schema import ToolCall

        mgr = PluginManager([])
        mgr._tools["srv"] = [_tool(name="do_thing", output="done", read_only=True)]
        registry = ToolRegistry()
        await mgr.register_tools(registry)

        definition = registry.get_definition("mcp_srv_do_thing")
        assert definition is not None
        assert definition.destructive is False

        # No confirm_fn: a read-only tool must still run (not be auto-denied).
        result = await execute_tool(
            ToolCall(id="1", name="mcp_srv_do_thing", arguments={}),
            registry,
            ToolSettings(),
            confirm_fn=None,
        )
        assert result.success is True
        assert result.output == "done"


class TestMcpToolName:
    """Audit finding: mcp: tool names must be OpenAI-function-name safe."""

    def test_sanitizes_colon_and_slash(self) -> None:
        name = mcp_tool_name("my-server", "read/file")
        assert name == "mcp_my-server_read_file"
        assert all(c.isalnum() or c in "_-" for c in name)

    def test_truncates_and_hashes_long_names(self) -> None:
        name = mcp_tool_name("p" * 50, "t" * 50)
        assert len(name) <= 64
        assert all(c.isalnum() or c in "_-" for c in name)
