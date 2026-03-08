"""Tests for conversation message history."""

from __future__ import annotations

from mita.agent.conversation import Conversation, Message, Role, _estimate_tokens


class TestMessage:
    def test_to_api_dict_basic(self) -> None:
        msg = Message(role=Role.USER, content="hello")
        d = msg.to_api_dict()
        assert d["role"] == "user"
        assert d["content"] == "hello"
        assert "tool_calls" not in d
        assert "tool_call_id" not in d

    def test_to_api_dict_with_tool_call_id(self) -> None:
        msg = Message(role=Role.TOOL, content="result", tool_call_id="tc_1", name="file_read")
        d = msg.to_api_dict()
        assert d["role"] == "tool"
        assert d["tool_call_id"] == "tc_1"
        assert d["name"] == "file_read"

    def test_to_api_dict_with_tool_calls(self) -> None:
        msg = Message(
            role=Role.ASSISTANT,
            content="",
            tool_calls=[{"id": "1", "type": "function", "function": {"name": "x"}}],
        )
        d = msg.to_api_dict()
        assert len(d["tool_calls"]) == 1


class TestConversation:
    def test_add_and_get_messages(self) -> None:
        conv = Conversation()
        conv.add(Message(role=Role.USER, content="hello"))
        conv.add(Message(role=Role.ASSISTANT, content="hi there"))
        msgs = conv.get_messages_for_api()
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"

    def test_total_tokens_tracked(self) -> None:
        conv = Conversation()
        conv.add(Message(role=Role.USER, content="a" * 100))
        assert conv.total_tokens > 0

    def test_truncate_keeps_system(self) -> None:
        conv = Conversation()
        conv.add(Message(role=Role.SYSTEM, content="system prompt"))
        conv.add(Message(role=Role.USER, content="msg1" * 100))
        conv.add(Message(role=Role.ASSISTANT, content="msg2" * 100))
        conv.add(Message(role=Role.USER, content="msg3" * 100))

        # Truncate to a small budget
        conv.truncate_to_fit(50)
        roles = [m.role for m in conv.messages]
        assert Role.SYSTEM in roles

    def test_truncate_noop_when_under_budget(self) -> None:
        conv = Conversation()
        conv.add(Message(role=Role.USER, content="short"))
        original_count = len(conv.messages)
        conv.truncate_to_fit(10000)
        assert len(conv.messages) == original_count

    def test_clear_non_system(self) -> None:
        conv = Conversation()
        conv.add(Message(role=Role.SYSTEM, content="sys"))
        conv.add(Message(role=Role.USER, content="user"))
        conv.add(Message(role=Role.ASSISTANT, content="assistant"))
        conv.clear_non_system()
        assert len(conv.messages) == 1
        assert conv.messages[0].role == Role.SYSTEM

    def test_truncate_keeps_recent_messages(self) -> None:
        conv = Conversation()
        conv.add(Message(role=Role.SYSTEM, content="s"))
        for i in range(10):
            conv.add(Message(role=Role.USER, content=f"message {i}" * 50))
        conv.truncate_to_fit(200)
        # Should keep system + some recent messages
        assert conv.messages[0].role == Role.SYSTEM
        assert len(conv.messages) < 11


class TestEstimateTokens:
    def test_short_text(self) -> None:
        assert _estimate_tokens("hi") >= 1

    def test_long_text(self) -> None:
        tokens = _estimate_tokens("a" * 400)
        assert tokens == 100  # 400 / 4

    def test_empty_text(self) -> None:
        assert _estimate_tokens("") >= 1
