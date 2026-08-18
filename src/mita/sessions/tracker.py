"""Live-session state shared by the REPL and TUI frontends."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

from mita.sessions.restore import conversation_from_record
from mita.sessions.store import SessionRecord, SessionStore, make_title, new_session_id

if TYPE_CHECKING:
    from mita.agent.conversation import Conversation
    from mita.config.schema import MitaConfig
    from mita.tools.registry import ToolRegistry


class SessionTracker:
    """Owns the current session's identity and its save/adopt operations.

    Frontends keep one tracker and render the strings it returns; all
    persistence policy (what gets saved, when the title is set, how a
    record becomes a conversation) lives here.
    """

    def __init__(self, store: SessionStore, config: MitaConfig) -> None:
        self.store = store
        self.config = config
        self.session_id = new_session_id()
        self.created_at = time.time()
        self.title = ""

    def reset(self) -> None:
        """Start a new session identity, leaving any previously saved file intact.

        Called when the conversation is cleared: without this, the next autosave
        would overwrite the resumed session's file with the emptied history.
        """
        self.session_id = new_session_id()
        self.created_at = time.time()
        self.title = ""

    def adopt(self, record: SessionRecord, registry: ToolRegistry) -> tuple[Conversation, str]:
        """Make a saved record the live session; returns (conversation, banner)."""
        conversation = conversation_from_record(record, self.config, registry)
        self.session_id = record.id
        self.created_at = record.created_at
        self.title = record.title
        banner = f'Resumed session {record.id} — "{record.title}" ({record.message_count} messages)'
        return conversation, banner

    def cwd_warning(self, record: SessionRecord) -> str | None:
        """Warning text when the record was saved in a different directory, else None."""
        if record.cwd == str(Path.cwd()):
            return None
        return (
            f"Session was saved in {record.cwd}; current directory is {Path.cwd()}. "
            "Tool results and index context may not match."
        )

    def save(self, conversation: Conversation, first_input: str = "") -> str | None:
        """Save the conversation; returns an error message on failure, else None.

        The first user input of a fresh session becomes the title.
        """
        if not self.title and first_input:
            self.title = make_title(first_input)
        try:
            self.store.save(
                SessionRecord.from_conversation(
                    conversation,
                    session_id=self.session_id,
                    title=self.title,
                    cwd=str(Path.cwd()),
                    model=self.config.model.default,
                    created_at=self.created_at,
                )
            )
        except OSError as e:
            return f"Failed to save session: {e}"
        return None
