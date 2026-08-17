"""Tests for tool schema models."""

from __future__ import annotations

from mita.tools.schema import ToolCall, ToolDefinition, ToolParameter, ToolResult


class TestToolDefinition:
    def test_to_openai_schema(self) -> None:
        td = ToolDefinition(
            name="test_tool",
            description="A test tool.",
            parameters=[
                ToolParameter(name="path", type="string", description="A path."),
                ToolParameter(
                    name="limit", type="integer", description="Max items.", required=False
                ),
            ],
        )
        schema = td.to_openai_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "test_tool"
        props = schema["function"]["parameters"]["properties"]
        assert "path" in props
        assert "limit" in props
        assert schema["function"]["parameters"]["required"] == ["path"]

    def test_destructive_default_false(self) -> None:
        td = ToolDefinition(name="x", description="x", parameters=[])
        assert td.destructive is False

    def test_source_default_builtin(self) -> None:
        td = ToolDefinition(name="x", description="x", parameters=[])
        assert td.source == "builtin"


class TestToolCall:
    def test_create(self) -> None:
        tc = ToolCall(id="call_1", name="file_read", arguments={"path": "/tmp/test"})
        assert tc.name == "file_read"
        assert tc.arguments["path"] == "/tmp/test"


class TestToolResult:
    def test_truncate_short_output(self) -> None:
        tr = ToolResult(tool_call_id="1", success=True, output="short")
        result = tr.truncate_output()
        assert result.output == "short"
        assert result.truncated is False

    def test_truncate_long_output(self) -> None:
        tr = ToolResult(tool_call_id="1", success=True, output="x" * 20_000)
        result = tr.truncate_output()
        assert result.truncated is True
        assert len(result.output) < 20_000
        assert "truncated" in result.output

    def test_error_result(self) -> None:
        tr = ToolResult(tool_call_id="1", success=False, error="fail")
        assert tr.error == "fail"
        assert tr.success is False


class TestMCPSchemaPreservation:
    """Audit finding F4.1: full JSON Schema (enum/items/nested) survives to OpenAI schema."""

    def test_enum_and_items_preserved(self) -> None:
        from mita.tools.schema import ToolDefinition, ToolParameter

        defn = ToolDefinition(
            name="mcp_x_do",
            description="do",
            parameters=[
                ToolParameter(
                    name="mode",
                    type="string",
                    description="mode",
                    json_schema={"type": "string", "enum": ["a", "b"], "description": "mode"},
                ),
                ToolParameter(
                    name="tags",
                    type="array",
                    description="tags",
                    json_schema={"type": "array", "items": {"type": "string"}},
                ),
            ],
        )
        props = defn.to_openai_schema()["function"]["parameters"]["properties"]
        assert props["mode"]["enum"] == ["a", "b"]
        assert props["tags"]["items"] == {"type": "string"}

    def test_builtin_param_still_flat(self) -> None:
        from mita.tools.schema import ToolDefinition, ToolParameter

        defn = ToolDefinition(
            name="file_read",
            description="read",
            parameters=[ToolParameter(name="path", type="string", description="path")],
        )
        props = defn.to_openai_schema()["function"]["parameters"]["properties"]
        assert props["path"] == {"type": "string", "description": "path"}
