"""Rebuild a live conversation from a saved session record."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mita.agent.context import assemble_context
from mita.agent.conversation import Conversation

if TYPE_CHECKING:
    from mita.config.schema import MitaConfig
    from mita.sessions.store import SessionRecord
    from mita.tools.registry import ToolRegistry


def conversation_from_record(
    record: SessionRecord,
    config: MitaConfig,
    registry: ToolRegistry,
) -> Conversation:
    """Build a conversation with a fresh system prompt plus the saved history."""
    conversation = Conversation()
    assemble_context(conversation, config, registry)
    for msg in record.messages:
        conversation.add(msg)
    return conversation
