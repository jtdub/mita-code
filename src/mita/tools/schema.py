"""Pydantic models for tool definitions, calls, and results."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ToolParameter(BaseModel):
    """A parameter in a tool definition."""

    name: str
    type: str  # "string", "integer", "boolean", "array", "object"
    description: str
    required: bool = True
    default: Any | None = None
    # Full JSON Schema fragment for this property, when available (MCP tools). Preserves
    # enum/items/nested/format that the flat fields drop, so strict providers accept the
    # request (audit finding F4.1). None for built-in tools (the flat fields suffice).
    json_schema: dict[str, Any] | None = None


class ToolDefinition(BaseModel):
    """Schema sent to the LLM so it knows what tools are available."""

    name: str
    description: str
    parameters: list[ToolParameter]
    destructive: bool = False  # triggers confirmation flow
    source: str = "builtin"  # "builtin" | "mcp:<plugin_name>"

    def to_openai_schema(self) -> dict[str, Any]:
        """Convert to OpenAI-compatible function schema for LiteLLM."""
        properties: dict[str, Any] = {}
        required: list[str] = []

        for param in self.parameters:
            if param.json_schema is not None:
                # Pass the original property schema through verbatim (keeps enum/items/
                # nested/format), just ensuring a description is present.
                prop = dict(param.json_schema)
                prop.setdefault("description", param.description)
            else:
                prop = {"type": param.type, "description": param.description}
                if param.default is not None:
                    prop["default"] = param.default
            properties[param.name] = prop
            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


class ToolCall(BaseModel):
    """Parsed from LLM output."""

    id: str
    name: str
    arguments: dict[str, Any]


class ToolResult(BaseModel):
    """Returned after executing a tool."""

    tool_call_id: str
    success: bool
    output: str = ""
    error: str | None = None
    truncated: bool = False

    # Max output size before truncation (10K chars)
    MAX_OUTPUT_SIZE: int = Field(default=10_000, exclude=True)

    def truncate_output(self) -> ToolResult:
        """Return a copy with output truncated if too long."""
        if len(self.output) <= self.MAX_OUTPUT_SIZE:
            return self
        truncated_output = (
            self.output[: self.MAX_OUTPUT_SIZE]
            + f"\n\n... [truncated, {len(self.output):,} chars total]"
        )
        return self.model_copy(update={"output": truncated_output, "truncated": True})

    @classmethod
    def from_subprocess(
        cls,
        stdout_bytes: bytes | None,
        stderr_bytes: bytes | None,
        returncode: int | None,
        *,
        empty_message: str = "(no output)",
    ) -> ToolResult:
        """Build a ToolResult from subprocess output.

        Shared by shell and git tools to avoid duplicating the
        stdout/stderr decoding, assembly, and exit-code formatting logic.
        """
        stdout = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
        stderr = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""

        output_parts: list[str] = []
        if stdout:
            output_parts.append(stdout)
        if stderr:
            output_parts.append(f"STDERR:\n{stderr}")

        output = "\n".join(output_parts) if output_parts else empty_message
        exit_code = returncode or 0

        if exit_code != 0:
            output = f"[exit code {exit_code}]\n{output}"

        return cls(
            tool_call_id="",
            success=exit_code == 0,
            output=output,
            error=f"Command failed with exit code {exit_code}" if exit_code != 0 else None,
        )
