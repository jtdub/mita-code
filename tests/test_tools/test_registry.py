"""Tests for the tool registry."""

from __future__ import annotations

import pytest

from mita.tools.registry import ToolRegistry, create_default_registry
from mita.tools.schema import ToolCall, ToolDefinition, ToolParameter, ToolResult


async def _dummy_handler(args: dict[str, object]) -> ToolResult:
    return ToolResult(tool_call_id="", success=True, output=f"got: {args}")


class TestToolRegistry:
    def test_register_and_get(self) -> None:
        registry = ToolRegistry()
        td = ToolDefinition(name="test", description="test", parameters=[])
        registry.register(td, _dummy_handler)
        assert registry.has_tool("test")
        assert registry.get_definition("test") == td

    def test_get_definitions(self) -> None:
        registry = ToolRegistry()
        td = ToolDefinition(name="test", description="test", parameters=[])
        registry.register(td, _dummy_handler)
        defs = registry.get_definitions()
        assert len(defs) == 1
        assert defs[0].name == "test"

    def test_get_openai_schemas(self) -> None:
        registry = ToolRegistry()
        td = ToolDefinition(
            name="test",
            description="test",
            parameters=[ToolParameter(name="x", type="string", description="x")],
        )
        registry.register(td, _dummy_handler)
        schemas = registry.get_openai_schemas()
        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "test"

    def test_unknown_tool(self) -> None:
        registry = ToolRegistry()
        assert registry.has_tool("nope") is False
        assert registry.get_definition("nope") is None

    @pytest.mark.asyncio()
    async def test_execute_known_tool(self) -> None:
        registry = ToolRegistry()
        td = ToolDefinition(name="test", description="test", parameters=[])
        registry.register(td, _dummy_handler)
        call = ToolCall(id="call_1", name="test", arguments={"x": 1})
        result = await registry.execute(call)
        assert result.success is True
        assert result.tool_call_id == "call_1"

    @pytest.mark.asyncio()
    async def test_execute_unknown_tool(self) -> None:
        registry = ToolRegistry()
        call = ToolCall(id="call_1", name="nope", arguments={})
        result = await registry.execute(call)
        assert result.success is False
        assert "Unknown tool" in (result.error or "")

    def test_tool_names(self) -> None:
        registry = ToolRegistry()
        td1 = ToolDefinition(name="a", description="a", parameters=[])
        td2 = ToolDefinition(name="b", description="b", parameters=[])
        registry.register(td1, _dummy_handler)
        registry.register(td2, _dummy_handler)
        assert set(registry.tool_names) == {"a", "b"}


class TestDefaultRegistry:
    def test_has_all_builtins(self) -> None:
        registry = create_default_registry()
        expected = {"file_read", "file_write", "file_edit", "glob", "grep", "shell", "git"}
        assert set(registry.tool_names) == expected

    def test_all_have_definitions(self) -> None:
        registry = create_default_registry()
        for name in registry.tool_names:
            td = registry.get_definition(name)
            assert td is not None
            assert td.name == name
