"""Tests for MCP plugin manager."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from mita.config.schema import PluginDefinition
from mita.plugins.manager import PluginManager, _make_mcp_handler, _schema_to_parameters
from mita.tools.registry import ToolRegistry
from mita.tools.schema import ToolResult


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
        mock_client = AsyncMock()
        mock_client.call_tool = AsyncMock(return_value="tool output")

        handler = _make_mcp_handler(mock_client, "read_file")
        result = await handler({"path": "/tmp/test"})

        assert isinstance(result, ToolResult)
        assert result.success is True
        assert result.output == "tool output"

    @pytest.mark.asyncio
    async def test_timeout(self) -> None:
        mock_client = AsyncMock()
        mock_client.call_tool = AsyncMock(side_effect=TimeoutError)

        handler = _make_mcp_handler(mock_client, "slow_tool")
        result = await handler({})

        assert result.success is False
        assert "timed out" in (result.error or "")

    @pytest.mark.asyncio
    async def test_connection_error(self) -> None:
        mock_client = AsyncMock()
        mock_client.call_tool = AsyncMock(side_effect=ConnectionError("dead"))

        handler = _make_mcp_handler(mock_client, "broken_tool")
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

    def test_get_client_none(self, plugins: list[PluginDefinition]) -> None:
        mgr = PluginManager(plugins)
        assert mgr.get_client("fs") is None

    @pytest.mark.asyncio
    async def test_start_all_with_failures(self) -> None:
        plugins = [
            PluginDefinition(name="good", transport="stdio", command="echo"),
            PluginDefinition(name="bad", transport="stdio"),  # missing command
        ]
        mgr = PluginManager(plugins)

        with patch.object(mgr, "start_all", new_callable=AsyncMock, return_value=["good"]):
            started = await mgr.start_all()
            assert "good" in started

    @pytest.mark.asyncio
    async def test_stop_all(self) -> None:
        mgr = PluginManager([])
        mock_client = AsyncMock()
        mock_client.disconnect = AsyncMock()
        mgr._clients["test"] = mock_client

        await mgr.stop_all()
        assert len(mgr._clients) == 0
        mock_client.disconnect.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_list_tools(self) -> None:
        mgr = PluginManager([])
        mock_client = AsyncMock()
        mock_client.list_tools = AsyncMock(
            return_value=[{"name": "read", "description": "Read file", "inputSchema": {}}]
        )
        mgr._clients["fs"] = mock_client

        result = await mgr.list_tools()
        assert "fs" in result
        assert len(result["fs"]) == 1
        assert result["fs"][0]["name"] == "read"

    @pytest.mark.asyncio
    async def test_list_tools_single_plugin(self) -> None:
        mgr = PluginManager([])
        mock_client = AsyncMock()
        mock_client.list_tools = AsyncMock(return_value=[])
        mgr._clients["fs"] = mock_client
        mgr._clients["web"] = AsyncMock()

        result = await mgr.list_tools("fs")
        assert "fs" in result
        assert "web" not in result

    @pytest.mark.asyncio
    async def test_register_tools(self) -> None:
        mgr = PluginManager([])
        mock_client = AsyncMock()
        mock_client.list_tools = AsyncMock(
            return_value=[
                {
                    "name": "read_file",
                    "description": "Read a file",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"path": {"type": "string", "description": "path"}},
                        "required": ["path"],
                    },
                }
            ]
        )
        mgr._clients["fs"] = mock_client

        registry = ToolRegistry()
        count = await mgr.register_tools(registry)

        assert count == 1
        assert registry.has_tool("mcp_fs_read_file")
        defn = registry.get_definition("mcp_fs_read_file")
        assert defn is not None
        assert defn.source == "mcp:fs"
        assert len(defn.parameters) == 1

    @pytest.mark.asyncio
    async def test_test_plugin_not_started(self) -> None:
        mgr = PluginManager([])
        result = await mgr.test_plugin("missing")
        assert result["connected"] is False

    @pytest.mark.asyncio
    async def test_test_plugin_healthy(self) -> None:
        mgr = PluginManager([])
        mock_client = AsyncMock()
        mock_client.ping = AsyncMock(return_value=True)
        mock_client.list_tools = AsyncMock(
            return_value=[{"name": "tool1", "description": "desc", "inputSchema": {}}]
        )
        mgr._clients["fs"] = mock_client

        result = await mgr.test_plugin("fs")
        assert result["ping"] is True
        assert result["tools"] == 1

    @pytest.mark.asyncio
    async def test_test_plugin_ping_failed(self) -> None:
        mgr = PluginManager([])
        mock_client = AsyncMock()
        mock_client.ping = AsyncMock(return_value=False)
        mgr._clients["fs"] = mock_client

        result = await mgr.test_plugin("fs")
        assert result["ping"] is False

    @pytest.mark.asyncio
    async def test_start_plugin(self) -> None:
        plugin = PluginDefinition(name="test", transport="stdio", command="echo")
        mgr = PluginManager([plugin])

        with patch("mita.plugins.manager.MCPPluginClient") as mock_cls:
            mock_instance = AsyncMock()
            mock_instance.connect = AsyncMock()
            mock_cls.return_value = mock_instance

            result = await mgr.start_plugin("test")
            assert result is True
            assert mgr.get_client("test") is mock_instance

    @pytest.mark.asyncio
    async def test_start_plugin_not_configured(self) -> None:
        mgr = PluginManager([])
        result = await mgr.start_plugin("missing")
        assert result is False

    @pytest.mark.asyncio
    async def test_stop_plugin(self) -> None:
        mgr = PluginManager([])
        mock_client = AsyncMock()
        mock_client.disconnect = AsyncMock()
        mgr._clients["test"] = mock_client

        await mgr.stop_plugin("test")
        assert mgr.get_client("test") is None
        mock_client.disconnect.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_tool_execution_via_registry(self) -> None:
        """End-to-end: register MCP tool, then execute via registry."""
        mgr = PluginManager([])
        mock_client = AsyncMock()
        mock_client.list_tools = AsyncMock(
            return_value=[
                {
                    "name": "greet",
                    "description": "Greet someone",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"name": {"type": "string", "description": "Name"}},
                    },
                }
            ]
        )
        mock_client.call_tool = AsyncMock(return_value="Hello, World!")
        mgr._clients["greeter"] = mock_client

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

    @staticmethod
    def _client_with_tool(read_only_hint: object) -> AsyncMock:
        tool: dict[str, object] = {
            "name": "do_thing",
            "description": "Does a thing",
            "inputSchema": {"type": "object", "properties": {}},
        }
        if read_only_hint is not None:
            tool["readOnlyHint"] = read_only_hint
        client = AsyncMock()
        client.list_tools = AsyncMock(return_value=[tool])
        client.call_tool = AsyncMock(return_value="done")
        return client

    @pytest.mark.asyncio
    async def test_plugin_tool_defaults_destructive_and_confirms(self) -> None:
        from mita.config.schema import ToolSettings
        from mita.tools.executor import execute_tool
        from mita.tools.schema import ToolCall

        mgr = PluginManager([])
        mgr._clients["srv"] = self._client_with_tool(read_only_hint=None)
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
        mgr._clients["srv"] = self._client_with_tool(read_only_hint=True)
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
        from mita.plugins.manager import mcp_tool_name

        name = mcp_tool_name("my-server", "read/file")
        assert name == "mcp_my-server_read_file"
        assert all(c.isalnum() or c in "_-" for c in name)

    def test_truncates_and_hashes_long_names(self) -> None:
        from mita.plugins.manager import mcp_tool_name

        name = mcp_tool_name("p" * 50, "t" * 50)
        assert len(name) <= 64
        assert all(c.isalnum() or c in "_-" for c in name)
