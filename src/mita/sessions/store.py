"""JSON-file store for saved chat sessions."""

from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

from mita.agent.conversation import Conversation, Message, Role

SESSION_VERSION = 1

# Resume spec meaning "the most recently active session".
LATEST = "latest"

_TITLE_MAX_LEN = 80

# Session ids become filenames; reject anything that could escape the sessions dir.
_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


class SessionStoreError(Exception):
    """Raised when a session cannot be loaded or saved."""


class SessionMeta(BaseModel):
    """Session metadata — everything a session listing shows."""

    # extra="ignore" lets list_metas() validate full session files as metadata only
    # (the file is still read and parsed; only Message validation is skipped).
    model_config = ConfigDict(extra="ignore")

    version: int = SESSION_VERSION
    id: str
    title: str
    cwd: str
    model: str
    created_at: float
    updated_at: float
    total_tokens: int = 0
    message_count: int = 0


class SessionRecord(SessionMeta):
    """A full saved session: metadata plus the non-system message history."""

    model_config = ConfigDict(extra="forbid")

    messages: list[Message] = []

    @classmethod
    def from_conversation(
        cls,
        conversation: Conversation,
        *,
        session_id: str,
        title: str,
        cwd: str,
        model: str,
        created_at: float,
    ) -> SessionRecord:
        """Build a record from a live conversation, dropping system messages.

        The system prompt (and injected RAG context) is regenerated on resume,
        so only user/assistant/tool messages are persisted.
        """
        messages = drop_unanswered_tool_calls(
            [m for m in conversation.messages if m.role != Role.SYSTEM]
        )
        return cls(
            id=session_id,
            title=make_title(title),
            cwd=cwd,
            model=model,
            created_at=created_at,
            updated_at=time.time(),
            total_tokens=conversation.total_tokens,
            message_count=len(messages),
            messages=messages,
        )


def drop_unanswered_tool_calls(messages: list[Message]) -> list[Message]:
    """Drop trailing assistant tool_calls that have no matching tool replies.

    An interrupt (Ctrl-C) can leave the history ending in an assistant message
    requesting tool calls that were never answered. Backends reject that shape,
    so persisting it would make the session permanently unresumable.
    """
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if msg.role != Role.ASSISTANT or not msg.tool_calls:
            continue
        answered = {
            m.tool_call_id for m in messages[i + 1 :] if m.role == Role.TOOL and m.tool_call_id
        }
        if all(call.get("id") in answered for call in msg.tool_calls):
            break
        # This turn was cut short: drop it and any partial replies that follow.
        return messages[:i]
    return list(messages)


def new_session_id() -> str:
    """Generate a sortable, collision-proof session id."""
    return time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]


def make_title(text: str) -> str:
    """Collapse whitespace and truncate a first prompt into a session title."""
    title = " ".join(text.split())
    if len(title) > _TITLE_MAX_LEN:
        title = title[: _TITLE_MAX_LEN - 1] + "…"
    return title


class SessionStore:
    """Reads and writes session JSON files in a directory."""

    def __init__(self, sessions_dir: Path) -> None:
        self._dir = sessions_dir

    def _path_for(self, session_id: str) -> Path:
        if not _ID_RE.match(session_id):
            raise SessionStoreError(f"Invalid session id: {session_id!r}")
        return self._dir / f"{session_id}.json"

    def save(self, record: SessionRecord) -> None:
        """Atomically write a session file."""
        from mita.config.toml_writer import atomic_write_text

        path = self._path_for(record.id)
        atomic_write_text(path, record.model_dump_json())

    def load(self, session_id: str) -> SessionRecord:
        """Load a session by id.

        Raises SessionStoreError on invalid id, missing file, corrupt JSON,
        or an unsupported schema version.
        """
        path = self._path_for(session_id)
        try:
            raw = path.read_text()
        except OSError as e:
            raise SessionStoreError(f"Session not found: {session_id}") from e
        # Check the version before full validation: a newer schema may add fields
        # that SessionRecord forbids, and "unsupported version" is the useful error.
        try:
            version = int(json.loads(raw).get("version", 0))
        except (ValueError, AttributeError, TypeError) as e:
            raise SessionStoreError(f"Session file is corrupt: {path}") from e
        if version != SESSION_VERSION:
            raise SessionStoreError(f"Session {session_id} has unsupported version {version}")
        try:
            return SessionRecord.model_validate_json(raw)
        except ValidationError as e:
            raise SessionStoreError(f"Session file is corrupt: {path}") from e

    def list_metas(self) -> list[SessionMeta]:
        """Return metadata for all readable sessions, newest activity first."""
        metas: list[SessionMeta] = []
        if not self._dir.is_dir():
            return metas
        for path in self._dir.glob("*.json"):
            try:
                meta = SessionMeta.model_validate_json(path.read_text())
            except (OSError, ValidationError):
                continue
            if meta.version != SESSION_VERSION:
                continue
            metas.append(meta)
        metas.sort(key=lambda m: m.updated_at, reverse=True)
        return metas

    def latest_id(self) -> str | None:
        """Return the id of the most recently active session, if any."""
        metas = self.list_metas()
        return metas[0].id if metas else None

    def resolve(self, spec: str) -> str:
        """Resolve a resume spec (a session id, or LATEST for the most recent)."""
        if spec != LATEST:
            return spec
        latest = self.latest_id()
        if latest is None:
            raise SessionStoreError("No saved sessions to resume.")
        return latest

    def clear(self) -> int:
        """Delete all session files. Returns the number removed."""
        count = 0
        if not self._dir.is_dir():
            return count
        for path in self._dir.glob("*.json"):
            try:
                path.unlink()
            except OSError:
                continue
            count += 1
        return count

    def prune(self, max_age_days: int) -> int:
        """Delete session files not written for max_age_days. 0 disables pruning.

        Uses file modification time: autosave rewrites the whole file each turn,
        so mtime tracks last activity without reading any file contents.
        """
        if max_age_days <= 0 or not self._dir.is_dir():
            return 0
        cutoff = time.time() - max_age_days * 86400
        count = 0
        for path in self._dir.glob("*.json"):
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
                    count += 1
            except OSError:
                continue
        return count
