"""Message history management with token counting and truncation."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class Role(StrEnum):
    """Message roles in the conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(BaseModel):
    """A single message in the conversation."""

    role: Role
    content: str
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    tool_call_id: str | None = None  # for tool-result messages
    name: str | None = None  # tool name for tool-result messages

    def to_api_dict(self) -> dict[str, Any]:
        """Convert to LiteLLM/OpenAI API format."""
        msg: dict[str, Any] = {"role": self.role.value, "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        if self.tool_call_id is not None:
            msg["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            msg["name"] = self.name
        return msg


class Conversation(BaseModel):
    """Conversation history with token estimation and truncation."""

    messages: list[Message] = Field(default_factory=list)
    total_tokens: int = 0

    def add(self, msg: Message) -> None:
        """Add a message to the conversation."""
        self.messages.append(msg)
        # Rough token estimate: ~4 chars per token
        self.total_tokens += _estimate_tokens(msg.content)

    def get_messages_for_api(self) -> list[dict[str, Any]]:
        """Convert all messages to API format."""
        return [m.to_api_dict() for m in self.messages]

    def truncate_to_fit(self, max_tokens: int) -> None:
        """Truncate conversation to fit within the token budget.

        Strategy: Keep the system message(s) and last N messages.
        Remove oldest non-system messages first.
        """
        if self.total_tokens <= max_tokens:
            return

        # Separate system messages from the rest
        system_msgs = [m for m in self.messages if m.role == Role.SYSTEM]
        other_msgs = [m for m in self.messages if m.role != Role.SYSTEM]

        system_tokens = sum(_estimate_tokens(m.content) for m in system_msgs)
        budget = max_tokens - system_tokens

        if budget <= 0:
            # System prompt alone exceeds budget — keep just the last system msg
            self.messages = system_msgs[-1:] if system_msgs else []
            self.total_tokens = sum(_estimate_tokens(m.content) for m in self.messages)
            return

        # Keep messages from the end until we hit the budget
        kept: list[Message] = []
        used = 0
        for msg in reversed(other_msgs):
            msg_tokens = _estimate_tokens(msg.content)
            if used + msg_tokens > budget:
                break
            kept.append(msg)
            used += msg_tokens

        kept.reverse()
        self.messages = system_msgs + kept
        self.total_tokens = system_tokens + used

    def clear_non_system(self) -> None:
        """Clear all non-system messages (for /clear command)."""
        self.messages = [m for m in self.messages if m.role == Role.SYSTEM]
        self.total_tokens = sum(_estimate_tokens(m.content) for m in self.messages)


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 characters per token."""
    return max(1, len(text) // 4)
