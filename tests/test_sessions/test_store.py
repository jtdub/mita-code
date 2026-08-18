"""Tests for the session store."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from mita.agent.conversation import Conversation, Message, Role
from mita.sessions.store import (
    LATEST,
    SESSION_VERSION,
    SessionRecord,
    SessionStore,
    SessionStoreError,
    make_title,
    new_session_id,
)


def _sample_conversation() -> Conversation:
    conv = Conversation()
    conv.add(Message(role=Role.SYSTEM, content="You are mita."))
    conv.add(Message(role=Role.USER, content="Refactor the parser"))
    conv.add(
        Message(
            role=Role.ASSISTANT,
            content="",
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path": "a.py"}'},
                }
            ],
        )
    )
    conv.add(
        Message(role=Role.TOOL, content="file contents", tool_call_id="call_1", name="read_file")
    )
    conv.add(Message(role=Role.ASSISTANT, content="Done."))
    return conv


def _sample_record(session_id: str = "20260816-120000-abcd1234") -> SessionRecord:
    return SessionRecord.from_conversation(
        _sample_conversation(),
        session_id=session_id,
        title="Refactor the parser",
        cwd="/some/project",
        model="qwen2.5-coder:7b",
        created_at=time.time(),
    )


class TestSessionRecord:
    def test_from_conversation_excludes_system_messages(self) -> None:
        record = _sample_record()
        assert all(m.role != Role.SYSTEM for m in record.messages)
        assert record.message_count == 4
        assert record.total_tokens > 0

    def test_from_conversation_excludes_rag_context(self) -> None:
        conv = _sample_conversation()
        conv.add(Message(role=Role.SYSTEM, content="Relevant code context:\n..."))
        record = SessionRecord.from_conversation(
            conv,
            session_id="rag-test",
            title="t",
            cwd="/p",
            model="m",
            created_at=time.time(),
        )
        assert all(m.role != Role.SYSTEM for m in record.messages)
        assert record.message_count == 4

    def test_round_trip_preserves_tool_fields(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path / "sessions")
        record = _sample_record()
        store.save(record)

        loaded = store.load(record.id)
        assert loaded.messages == record.messages
        assert loaded.messages[1].tool_calls[0]["function"]["name"] == "read_file"
        assert loaded.messages[2].tool_call_id == "call_1"
        assert loaded.messages[2].name == "read_file"
        assert loaded.title == record.title
        assert loaded.model == record.model


class TestSessionStoreLoad:
    def test_missing_session_raises(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path / "sessions")
        with pytest.raises(SessionStoreError, match="not found"):
            store.load("20260816-000000-deadbeef")

    def test_invalid_id_raises(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path / "sessions")
        with pytest.raises(SessionStoreError, match="Invalid session id"):
            store.load("../evil")

    def test_corrupt_json_raises(self, tmp_path: Path) -> None:
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()
        (sessions_dir / "bad.json").write_text("{not json")
        with pytest.raises(SessionStoreError, match="corrupt"):
            SessionStore(sessions_dir).load("bad")

    def test_wrong_version_raises(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path / "sessions")
        record = _sample_record()
        data = record.model_dump(mode="json")
        data["version"] = SESSION_VERSION + 1
        store._dir.mkdir(parents=True)
        (store._dir / f"{record.id}.json").write_text(json.dumps(data))
        with pytest.raises(SessionStoreError, match="unsupported version"):
            store.load(record.id)


class TestSessionStoreListing:
    def test_list_metas_sorted_and_skips_corrupt(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path / "sessions")
        old = _sample_record("session-old")
        new = _sample_record("session-new")
        old.updated_at = 100.0
        new.updated_at = 200.0
        store.save(old)
        store.save(new)
        (store._dir / "corrupt.json").write_text("{nope")

        metas = store.list_metas()
        assert [m.id for m in metas] == ["session-new", "session-old"]

    def test_latest_id(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path / "sessions")
        assert store.latest_id() is None
        record = _sample_record()
        store.save(record)
        assert store.latest_id() == record.id

    def test_list_metas_empty_dir(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path / "missing")
        assert store.list_metas() == []

    def test_resolve_explicit_id_passes_through(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path / "sessions")
        assert store.resolve("some-id") == "some-id"

    def test_resolve_latest_returns_most_recent(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path / "sessions")
        record = _sample_record()
        store.save(record)
        assert store.resolve(LATEST) == record.id

    def test_resolve_latest_with_no_sessions_raises(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path / "sessions")
        with pytest.raises(SessionStoreError, match="No saved sessions"):
            store.resolve(LATEST)


def _backdate(store: SessionStore, session_id: str, age_days: float) -> None:
    """Set a session file's mtime age_days into the past."""
    path = store._dir / f"{session_id}.json"
    stamp = time.time() - age_days * 86400
    os.utime(path, (stamp, stamp))


class TestSessionStorePruneAndClear:
    def test_prune_deletes_old_sessions(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path / "sessions")
        store.save(_sample_record("session-old"))
        store.save(_sample_record("session-fresh"))
        _backdate(store, "session-old", age_days=30)

        assert store.prune(7) == 1
        assert [m.id for m in store.list_metas()] == ["session-fresh"]

    def test_prune_zero_is_noop(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path / "sessions")
        store.save(_sample_record("session-old"))
        _backdate(store, "session-old", age_days=365)
        assert store.prune(0) == 0
        assert len(store.list_metas()) == 1

    def test_clear_returns_count(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path / "sessions")
        store.save(_sample_record("a1"))
        store.save(_sample_record("b2"))
        assert store.clear() == 2
        assert store.list_metas() == []


class TestHelpers:
    def test_new_session_id_is_valid_and_unique(self) -> None:
        a, b = new_session_id(), new_session_id()
        assert a != b
        # Must be usable as a filename via the store's id validation.
        store = SessionStore(Path("unused"))
        assert store._path_for(a).name == f"{a}.json"

    def test_make_title_collapses_and_truncates(self) -> None:
        assert make_title("fix\nthe   bug") == "fix the bug"
        long = make_title("x" * 200)
        assert len(long) == 80
        assert long.endswith("…")


class TestDanglingToolCalls:
    def test_interrupted_tool_call_is_not_persisted(self) -> None:
        """A trailing assistant tool_calls with no tool reply would 400 on resume."""
        conv = Conversation()
        conv.add(Message(role=Role.USER, content="run the tests"))
        conv.add(
            Message(
                role=Role.ASSISTANT,
                content="",
                tool_calls=[{"id": "call_9", "type": "function", "function": {"name": "shell"}}],
            )
        )
        record = SessionRecord.from_conversation(
            conv,
            session_id="interrupted",
            title="run the tests",
            cwd="/p",
            model="m",
            created_at=time.time(),
        )
        assert [m.role for m in record.messages] == [Role.USER]
        assert record.message_count == 1

    def test_answered_tool_call_is_kept(self) -> None:
        record = _sample_record()
        assert any(m.tool_calls for m in record.messages)
        assert record.messages[-1].content == "Done."

    def test_partially_answered_tool_call_is_dropped(self) -> None:
        conv = Conversation()
        conv.add(Message(role=Role.USER, content="do two things"))
        conv.add(
            Message(
                role=Role.ASSISTANT,
                content="",
                tool_calls=[
                    {"id": "a", "type": "function", "function": {"name": "shell"}},
                    {"id": "b", "type": "function", "function": {"name": "shell"}},
                ],
            )
        )
        conv.add(Message(role=Role.TOOL, content="ok", tool_call_id="a", name="shell"))
        record = SessionRecord.from_conversation(
            conv,
            session_id="partial",
            title="do two things",
            cwd="/p",
            model="m",
            created_at=time.time(),
        )
        # The unanswered call would be rejected by the backend, so the whole
        # assistant turn is dropped; the orphan tool reply is harmless context.
        assert all(not m.tool_calls for m in record.messages)


class TestVersionGate:
    def test_future_version_reports_version_not_corruption(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path / "sessions")
        record = _sample_record("future")
        store.save(record)
        path = store._dir / "future.json"
        data = json.loads(path.read_text())
        data["version"] = SESSION_VERSION + 1
        data["a_new_field_from_v2"] = True
        path.write_text(json.dumps(data))

        with pytest.raises(SessionStoreError, match="unsupported version"):
            store.load("future")


class TestSessionsDirLocation:
    def test_anchored_to_project_root_not_cwd(
        self, tmp_project_with_mita: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Running from a subdirectory must not create a nested .mita/ project root."""
        from mita.sessions import get_sessions_dir

        subdir = tmp_project_with_mita / "src" / "deep"
        subdir.mkdir(parents=True)
        monkeypatch.chdir(subdir)

        assert get_sessions_dir() == tmp_project_with_mita / ".mita" / "sessions"

    def test_falls_back_to_cwd_outside_a_project(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from mita.sessions import get_sessions_dir

        loose = tmp_path / "loose"
        loose.mkdir()
        monkeypatch.chdir(loose)
        assert get_sessions_dir() == loose / ".mita" / "sessions"
