"""Tool registry: name → (definition, handler) mapping."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from mita.tools.schema import ToolCall, ToolDefinition, ToolResult

# Type for async tool handler functions
ToolHandler = Callable[[dict[str, Any]], Awaitable[ToolResult]]


class ToolRegistry:
    """Central registry for all available tools (built-in + plugins)."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        self._handlers: dict[str, ToolHandler] = {}

    def register(self, definition: ToolDefinition, handler: ToolHandler) -> None:
        """Register a tool with its definition and handler."""
        self._tools[definition.name] = definition
        self._handlers[definition.name] = handler

    def get_definition(self, name: str) -> ToolDefinition | None:
        """Get a tool definition by name."""
        return self._tools.get(name)

    def get_definitions(self) -> list[ToolDefinition]:
        """Get all registered tool definitions."""
        return list(self._tools.values())

    def get_openai_schemas(self) -> list[dict[str, Any]]:
        """Get all tool definitions as OpenAI-compatible schemas."""
        return [t.to_openai_schema() for t in self._tools.values()]

    def has_tool(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    async def execute(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool call and return the result."""
        handler = self._handlers.get(tool_call.name)
        if handler is None:
            return ToolResult(
                tool_call_id=tool_call.id,
                success=False,
                error=f"Unknown tool: {tool_call.name}",
            )
        result = await handler(tool_call.arguments)
        # Attach the tool_call_id
        result = result.model_copy(update={"tool_call_id": tool_call.id})
        return result.truncate_output()

    @property
    def tool_names(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())


def create_default_registry() -> ToolRegistry:
    """Create a registry with all built-in tools registered."""
    from mita.tools.builtins import BUILTIN_TOOLS

    registry = ToolRegistry()
    for _name, (definition, handler) in BUILTIN_TOOLS.items():
        registry.register(definition, handler)  # type: ignore[arg-type]
    return registry
