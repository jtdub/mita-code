"""Tests for the live-session tracker shared by the REPL and TUI."""

from __future__ import annotations

from pathlib import Path

from mita.agent.conversation import Conversation, Message, Role
from mita.config.schema import MitaConfig
from mita.sessions.store import SessionStore
from mita.sessions.tracker import SessionTracker
from mita.tools.registry import ToolRegistry
from tests.test_sessions.test_restore import record_with_history


def _tracker(tmp_path: Path) -> SessionTracker:
    return SessionTracker(SessionStore(tmp_path / "sessions"), MitaConfig())


class TestAdopt:
    def test_adopt_takes_over_identity_and_returns_banner(self, tmp_path: Path) -> None:
        tracker = _tracker(tmp_path)
        record = record_with_history()

        conversation, banner = tracker.adopt(record, ToolRegistry())

        assert tracker.session_id == record.id
        assert tracker.created_at == record.created_at
        assert tracker.title == record.title
        assert record.id in banner
        assert conversation.messages[0].role == Role.SYSTEM


class TestCwdWarning:
    def test_warns_on_different_cwd(self, tmp_path: Path) -> None:
        tracker = _tracker(tmp_path)
        record = record_with_history()  # cwd="/elsewhere"
        warning = tracker.cwd_warning(record)
        assert warning is not None
        assert "/elsewhere" in warning

    def test_no_warning_on_same_cwd(self, tmp_path: Path) -> None:
        tracker = _tracker(tmp_path)
        record = record_with_history()
        record.cwd = str(Path.cwd())
        assert tracker.cwd_warning(record) is None


class TestSave:
    def test_save_sets_title_from_first_input_and_persists(self, tmp_path: Path) -> None:
        tracker = _tracker(tmp_path)
        conv = Conversation()
        conv.add(Message(role=Role.USER, content="fix the bug"))

        assert tracker.save(conv, "fix   the\nbug") is None
        assert tracker.title == "fix the bug"

        loaded = tracker.store.load(tracker.session_id)
        assert loaded.title == "fix the bug"
        assert loaded.cwd == str(Path.cwd())

    def test_save_keeps_existing_title(self, tmp_path: Path) -> None:
        tracker = _tracker(tmp_path)
        tracker.title = "original"
        conv = Conversation()
        conv.add(Message(role=Role.USER, content="second prompt"))

        assert tracker.save(conv, "second prompt") is None
        assert tracker.title == "original"

    def test_save_continues_same_session_after_adopt(self, tmp_path: Path) -> None:
        tracker = _tracker(tmp_path)
        record = record_with_history()
        conversation, _ = tracker.adopt(record, ToolRegistry())
        conversation.add(Message(role=Role.USER, content="one more"))

        assert tracker.save(conversation, "one more") is None
        loaded = tracker.store.load(record.id)
        assert loaded.message_count == 3


class TestReset:
    def test_reset_starts_a_new_session_file(self, tmp_path: Path) -> None:
        """/clear must not overwrite the resumed session with the emptied history."""
        tracker = _tracker(tmp_path)
        record = record_with_history()
        tracker.store.save(record)
        conversation, _ = tracker.adopt(record, ToolRegistry())
        original_id = tracker.session_id

        conversation.clear_non_system()
        tracker.reset()
        assert tracker.session_id != original_id
        assert tracker.title == ""

        assert tracker.save(conversation, "brand new topic") is None
        # The resumed session is untouched on disk.
        assert tracker.store.load(original_id).message_count == 2
        assert tracker.store.load(tracker.session_id).message_count == 0
