"""Smoke tests for the Textual TUI (TUI enablement)."""

from __future__ import annotations

import pytest

from mita.config.schema import MitaConfig
from mita.sessions import get_sessions_dir
from mita.sessions.store import SessionStore
from mita.sessions.tracker import SessionTracker
from mita.tools.registry import create_default_registry
from mita.tools.schema import ToolCall, ToolResult
from mita.ui.sink import ConfirmDecision, ConfirmRequest, UISink
from mita.ui.tui.app import ConfirmScreen, MitaTUI, SessionPickerScreen


def _app(**kwargs) -> MitaTUI:  # type: ignore[no-untyped-def]
    config = MitaConfig()
    tracker = SessionTracker(SessionStore(get_sessions_dir()), config)
    return MitaTUI(config, create_default_registry(), tracker=tracker, **kwargs)


def test_app_is_a_uisink() -> None:
    assert isinstance(_app(), UISink)


class TestConfirmScreen:
    def test_button_ids_map_to_decisions(self) -> None:
        # The mapping the modal uses to translate a button press into a decision.
        screen = ConfirmScreen(ConfirmRequest(tool_name="shell", summary="run?"))
        assert screen._request.tool_name == "shell"


class TestTUIRuntime:
    @pytest.mark.asyncio()
    async def test_mounts_and_shows_welcome(self) -> None:
        app = _app()
        async with app.run_test() as pilot:
            await pilot.pause()
            # UISink methods render without raising once mounted.
            app.notice("a notice")
            app.error("bad [/x] token")
            app.tool_call(ToolCall(id="1", name="file_read", arguments={"path": "x"}))
            app.tool_result(ToolResult(tool_call_id="1", success=True, output="ok"))
            app.stream_token("hello ")
            app.stream_token("world")
            app.stream_end()
            app.response_stats(1, 2, 0.5, 0.1)

    @pytest.mark.asyncio()
    async def test_quit_command_exits(self) -> None:
        app = _app()
        async with app.run_test() as pilot:
            await pilot.pause()
            prompt = app.query_one("#prompt")
            prompt.value = "/quit"
            await pilot.press("enter")
            await pilot.pause()
        # run_test context exits cleanly when the app has exited.
        assert app.is_running is False

    @pytest.mark.asyncio()
    async def test_confirm_returns_decision_from_modal(self) -> None:
        app = _app()
        async with app.run_test() as pilot:
            await pilot.pause()

            async def ask() -> ConfirmDecision:
                return await app.confirm(ConfirmRequest(tool_name="shell", summary="run rm?"))

            task = app.run_worker(ask(), exclusive=False)
            await pilot.pause()
            # Press the "Always" button in the modal.
            await pilot.click("#always")
            await pilot.pause()
            result = await task.wait()
        assert result is ConfirmDecision.ALLOW_SESSION


class TestTUISessions:
    def _record(self):  # type: ignore[no-untyped-def]
        import time

        from mita.agent.conversation import Conversation, Message, Role
        from mita.sessions.store import SessionRecord

        conv = Conversation()
        conv.add(Message(role=Role.USER, content="hello"))
        conv.add(Message(role=Role.ASSISTANT, content="hi"))
        return SessionRecord.from_conversation(
            conv,
            session_id="tui-test-session",
            title="hello",
            cwd="/elsewhere",
            model="test-model",
            created_at=time.time(),
        )

    @pytest.mark.asyncio()
    async def test_resume_record_seeds_conversation(self) -> None:
        record = self._record()
        app = _app(resume_record=record)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app._mita_conversation is not None
            contents = [m.content for m in app._mita_conversation.messages]
            assert "hello" in contents
            assert "hi" in contents
            assert app._mita_tracker.session_id == "tui-test-session"

    @pytest.mark.asyncio()
    async def test_resume_command_with_no_sessions(self) -> None:
        app = _app()
        async with app.run_test() as pilot:
            await pilot.pause()
            prompt = app.query_one("#prompt")
            prompt.value = "/resume"
            await pilot.press("enter")
            await pilot.pause()
            # No saved sessions: the app stays running and no picker is shown.
            assert app.is_running is True
            assert app._mita_conversation is None

    @pytest.mark.asyncio()
    async def test_resume_command_with_id_loads_session(self) -> None:
        record = self._record()
        SessionStore(get_sessions_dir()).save(record)

        app = _app()
        async with app.run_test() as pilot:
            await pilot.pause()
            prompt = app.query_one("#prompt")
            prompt.value = "/resume tui-test-session"
            await pilot.press("enter")
            await pilot.pause()
            assert app._mita_conversation is not None
            assert app._mita_tracker.session_id == "tui-test-session"


class TestTUIResumePicker:
    @pytest.mark.asyncio()
    async def test_resume_picker_opens_without_crashing(self) -> None:
        """Regression: push_screen_wait needs a worker, else NoActiveWorker kills the app."""
        from mita.agent.conversation import Conversation, Message, Role
        from mita.sessions.store import SessionRecord

        conv = Conversation()
        conv.add(Message(role=Role.USER, content="hello"))
        SessionStore(get_sessions_dir()).save(
            SessionRecord.from_conversation(
                conv,
                session_id="picker-session",
                title="hello",
                cwd="/elsewhere",
                model="m",
                created_at=1.0,
            )
        )

        app = _app()
        async with app.run_test() as pilot:
            await pilot.pause()
            prompt = app.query_one("#prompt")
            prompt.value = "/resume"
            await pilot.press("enter")
            await pilot.pause()
            await pilot.pause()
            assert app.is_running is True
            # The picker modal is up and lists the saved session.
            assert isinstance(app.screen, SessionPickerScreen)

    @pytest.mark.asyncio()
    async def test_clear_starts_a_new_session(self) -> None:
        app = _app()
        async with app.run_test() as pilot:
            await pilot.pause()
            original = app._mita_tracker.session_id
            prompt = app.query_one("#prompt")
            prompt.value = "/clear"
            await pilot.press("enter")
            await pilot.pause()
            assert app._mita_tracker.session_id != original
