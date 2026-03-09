"""Tests for MCP plugin client."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mita.config.schema import PluginDefinition
from mita.plugins.client import MCPPluginClient


@pytest.fixture
def stdio_plugin() -> PluginDefinition:
    return PluginDefinition(
        name="test-plugin",
        transport="stdio",
        command="echo",
        args=["hello"],
    )


@pytest.fixture
def sse_plugin() -> PluginDefinition:
    return PluginDefinition(
        name="test-sse",
        transport="sse",
        url="http://localhost:3001/sse",
    )


class TestMCPPluginClient:
    def test_init(self, stdio_plugin: PluginDefinition) -> None:
        client = MCPPluginClient(stdio_plugin)
        assert client.name == "test-plugin"
        assert client.transport == "stdio"
        assert not client.connected

    def test_missing_command_raises(self) -> None:
        plugin = PluginDefinition(name="bad", transport="stdio")
        client = MCPPluginClient(plugin)
        with pytest.raises(ValueError, match="requires a command"):
            import asyncio

            asyncio.get_event_loop().run_until_complete(client.connect())

    def test_missing_url_raises(self) -> None:
        plugin = PluginDefinition(name="bad", transport="sse")
        client = MCPPluginClient(plugin)
        with pytest.raises(ValueError, match="requires a url"):
            import asyncio

            asyncio.get_event_loop().run_until_complete(client.connect())

    def test_unsupported_transport_raises(self) -> None:
        plugin = PluginDefinition(name="bad", transport="grpc")
        client = MCPPluginClient(plugin)
        with pytest.raises(ValueError, match="Unsupported transport"):
            import asyncio

            asyncio.get_event_loop().run_until_complete(client.connect())

    @pytest.mark.asyncio
    async def test_list_tools_not_connected(self, stdio_plugin: PluginDefinition) -> None:
        client = MCPPluginClient(stdio_plugin)
        with pytest.raises(RuntimeError, match="not connected"):
            await client.list_tools()

    @pytest.mark.asyncio
    async def test_call_tool_not_connected(self, stdio_plugin: PluginDefinition) -> None:
        client = MCPPluginClient(stdio_plugin)
        with pytest.raises(RuntimeError, match="not connected"):
            await client.call_tool("test", {})

    @pytest.mark.asyncio
    async def test_ping_not_connected(self, stdio_plugin: PluginDefinition) -> None:
        client = MCPPluginClient(stdio_plugin)
        assert not await client.ping()

    @pytest.mark.asyncio
    async def test_connect_stdio(self, stdio_plugin: PluginDefinition) -> None:
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()

        mock_transport = (MagicMock(), MagicMock())

        with (
            patch("mita.plugins.client.stdio_client") as mock_stdio,
            patch("mita.plugins.client.ClientSession", return_value=mock_session),
        ):
            # stdio_client returns an async context manager yielding transport
            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_transport)
            mock_cm.__aexit__ = AsyncMock(return_value=False)
            mock_stdio.return_value = mock_cm

            # ClientSession also an async context manager
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)

            client = MCPPluginClient(stdio_plugin)
            await client.connect()

            assert client.connected
            mock_session.initialize.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_list_tools_with_session(self, stdio_plugin: PluginDefinition) -> None:
        mock_tool = MagicMock()
        mock_tool.name = "read_file"
        mock_tool.description = "Read a file"
        mock_tool.inputSchema = {"type": "object", "properties": {"path": {"type": "string"}}}

        mock_response = MagicMock()
        mock_response.tools = [mock_tool]

        client = MCPPluginClient(stdio_plugin)
        client._session = AsyncMock()
        client._session.list_tools = AsyncMock(return_value=mock_response)

        tools = await client.list_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "read_file"
        assert tools[0]["description"] == "Read a file"

    @pytest.mark.asyncio
    async def test_call_tool_with_session(self, stdio_plugin: PluginDefinition) -> None:
        mock_content = MagicMock()
        mock_content.text = "file contents here"

        mock_result = MagicMock()
        mock_result.content = [mock_content]

        client = MCPPluginClient(stdio_plugin)
        client._session = AsyncMock()
        client._session.call_tool = AsyncMock(return_value=mock_result)

        output = await client.call_tool("read_file", {"path": "/tmp/test.txt"})
        assert output == "file contents here"
        client._session.call_tool.assert_awaited_once_with("read_file", {"path": "/tmp/test.txt"})

    @pytest.mark.asyncio
    async def test_disconnect(self, stdio_plugin: PluginDefinition) -> None:
        client = MCPPluginClient(stdio_plugin)
        client._session = AsyncMock()
        client._exit_stack = AsyncMock()
        client._exit_stack.aclose = AsyncMock()

        await client.disconnect()
        assert client._session is None
        assert client._exit_stack is None

    @pytest.mark.asyncio
    async def test_ping_success(self, stdio_plugin: PluginDefinition) -> None:
        client = MCPPluginClient(stdio_plugin)
        client._session = AsyncMock()
        client._session.send_ping = AsyncMock()

        assert await client.ping()

    @pytest.mark.asyncio
    async def test_ping_timeout(self, stdio_plugin: PluginDefinition) -> None:
        client = MCPPluginClient(stdio_plugin)
        client._session = AsyncMock()
        client._session.send_ping = AsyncMock(side_effect=TimeoutError)

        assert not await client.ping()
