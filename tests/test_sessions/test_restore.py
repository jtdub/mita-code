"""Tests for restoring a conversation from a saved session."""

from __future__ import annotations

import time

from mita.agent.conversation import Conversation, Message, Role
from mita.config.schema import MitaConfig
from mita.sessions.restore import conversation_from_record
from mita.sessions.store import SessionRecord
from mita.tools.registry import ToolRegistry


def record_with_history() -> SessionRecord:
    conv = Conversation()
    conv.add(Message(role=Role.SYSTEM, content="OLD SYSTEM PROMPT"))
    conv.add(Message(role=Role.USER, content="hello"))
    conv.add(Message(role=Role.ASSISTANT, content="hi there"))
    return SessionRecord.from_conversation(
        conv,
        session_id="restore-test",
        title="hello",
        cwd="/elsewhere",
        model="test-model",
        created_at=time.time(),
    )


class TestConversationFromRecord:
    def test_regenerates_system_prompt_and_keeps_history(self) -> None:
        record = record_with_history()
        conversation = conversation_from_record(record, MitaConfig(), ToolRegistry())

        assert conversation.messages[0].role == Role.SYSTEM
        assert conversation.messages[0].content != "OLD SYSTEM PROMPT"
        non_system = [m for m in conversation.messages if m.role != Role.SYSTEM]
        assert [m.content for m in non_system] == ["hello", "hi there"]
        assert conversation.total_tokens > 0
