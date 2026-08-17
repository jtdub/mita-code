"""Context assembly: memory + tool definitions → system prompt."""

from __future__ import annotations

from pathlib import Path

from mita.agent.conversation import Conversation, Message, Role
from mita.agent.system_prompt import build_system_prompt
from mita.config.schema import MitaConfig
from mita.memory.loader import load_memory
from mita.tools.registry import ToolRegistry


def assemble_context(
    conversation: Conversation,
    config: MitaConfig,
    registry: ToolRegistry,
    cwd: Path | None = None,
) -> None:
    """Assemble the initial context for the agent loop.

    Builds the system prompt from config, memory, and tool definitions,
    and adds it to the conversation.

    Args:
        conversation: The conversation to add the system message to.
        config: Application configuration.
        registry: Tool registry with all available tools.
        cwd: Working directory for memory discovery. Defaults to Path.cwd().
    """
    working_dir = cwd or Path.cwd()

    # Load memory from MITA.md files, honoring the configured memory limits.
    memory_content = load_memory(cwd=working_dir, settings=config.memory)

    # Build system prompt
    system_prompt = build_system_prompt(
        config=config,
        memory=memory_content,
        tools=registry.get_definitions(),
    )

    conversation.add(Message(role=Role.SYSTEM, content=system_prompt))
