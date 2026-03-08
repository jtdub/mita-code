"""Build the system prompt from config, memory, and tool definitions."""

from __future__ import annotations

from mita.config.schema import MitaConfig
from mita.tools.schema import ToolDefinition

SYSTEM_PROMPT_TEMPLATE = """\
You are Mita, a local-first coding assistant running on the user's machine.
You help with software engineering tasks: writing code, debugging, refactoring, \
explaining code, and answering questions.

You have access to the following tools to interact with the local filesystem and \
run commands. Use them to help the user.

## Available Tools
{tools_section}

## Guidelines
- Read files before modifying them to understand context.
- Use file_edit for targeted changes; use file_write only for new files or full rewrites.
- Use glob and grep to explore the codebase before making changes.
- Run tests after making changes to verify correctness.
- Ask for confirmation before destructive operations.
- Be concise in your responses. Lead with the answer, not the reasoning.
- When you're done with a task, say so clearly.
{memory_section}\
"""


def build_system_prompt(
    config: MitaConfig,
    memory: str,
    tools: list[ToolDefinition],
) -> str:
    """Build the full system prompt.

    Args:
        config: The application configuration.
        memory: Loaded memory content from MITA.md files.
        tools: All available tool definitions.
    """
    tools_section = _format_tools(tools)
    memory_section = _format_memory(memory)
    return SYSTEM_PROMPT_TEMPLATE.format(
        tools_section=tools_section,
        memory_section=memory_section,
    )


def _format_tools(tools: list[ToolDefinition]) -> str:
    """Format tool definitions for the system prompt."""
    parts: list[str] = []
    for tool in tools:
        params = ", ".join(
            f"{p.name}: {p.type}" + ("" if p.required else f" = {p.default}")
            for p in tool.parameters
        )
        destructive_note = " [DESTRUCTIVE]" if tool.destructive else ""
        parts.append(f"- **{tool.name}**({params}){destructive_note}: {tool.description}")
    return "\n".join(parts)


def _format_memory(memory: str) -> str:
    """Format memory content for the system prompt."""
    if not memory.strip():
        return ""
    return f"\n## Project Memory\n{memory}\n"
