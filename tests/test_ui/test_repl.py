"""Tests for the interactive REPL."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mita.ui.repl import _try_render_skill, repl_loop


class TestReplLoop:
    @pytest.mark.asyncio()
    async def test_empty_input_ignored(self) -> None:
        """Empty input lines should be skipped."""
        console = MagicMock()
        on_input = AsyncMock()

        with patch("mita.ui.repl.PromptSession") as mock_session_cls:
            session = MagicMock()
            # Return empty, then /quit
            session.prompt_async = AsyncMock(side_effect=["", "/quit"])
            mock_session_cls.return_value = session
            with patch("mita.ui.repl.display_welcome"), patch("mita.ui.repl.display_goodbye"):
                await repl_loop(console, on_input)

        on_input.assert_not_called()

    @pytest.mark.asyncio()
    async def test_quit_command(self) -> None:
        """Various quit commands should exit the loop."""
        for cmd in ["/quit", "/exit", "/q"]:
            console = MagicMock()
            on_input = AsyncMock()

            with patch("mita.ui.repl.PromptSession") as mock_session_cls:
                session = MagicMock()
                session.prompt_async = AsyncMock(return_value=cmd)
                mock_session_cls.return_value = session
                with patch("mita.ui.repl.display_welcome"), patch("mita.ui.repl.display_goodbye"):
                    await repl_loop(console, on_input)

            on_input.assert_not_called()

    @pytest.mark.asyncio()
    async def test_clear_command(self) -> None:
        """/clear should call on_clear and continue."""
        console = MagicMock()
        on_input = AsyncMock()
        on_clear = MagicMock()

        with patch("mita.ui.repl.PromptSession") as mock_session_cls:
            session = MagicMock()
            session.prompt_async = AsyncMock(side_effect=["/clear", "/quit"])
            mock_session_cls.return_value = session
            with patch("mita.ui.repl.display_welcome"), patch("mita.ui.repl.display_goodbye"):
                await repl_loop(console, on_input, on_clear=on_clear)

        on_clear.assert_called_once()
        on_input.assert_not_called()

    @pytest.mark.asyncio()
    async def test_clear_without_callback(self) -> None:
        """/clear without on_clear should just print."""
        console = MagicMock()
        on_input = AsyncMock()

        with patch("mita.ui.repl.PromptSession") as mock_session_cls:
            session = MagicMock()
            session.prompt_async = AsyncMock(side_effect=["/clear", "/quit"])
            mock_session_cls.return_value = session
            with patch("mita.ui.repl.display_welcome"), patch("mita.ui.repl.display_goodbye"):
                await repl_loop(console, on_input, on_clear=None)

        console.print.assert_called()

    @pytest.mark.asyncio()
    async def test_normal_input(self) -> None:
        """Regular input should be passed to on_input."""
        console = MagicMock()
        on_input = AsyncMock()

        with patch("mita.ui.repl.PromptSession") as mock_session_cls:
            session = MagicMock()
            session.prompt_async = AsyncMock(side_effect=["hello world", "/quit"])
            mock_session_cls.return_value = session
            with patch("mita.ui.repl.display_welcome"), patch("mita.ui.repl.display_goodbye"):
                await repl_loop(console, on_input)

        on_input.assert_called_once_with("hello world")

    @pytest.mark.asyncio()
    async def test_eof_exits(self) -> None:
        """EOFError should exit the loop gracefully."""
        console = MagicMock()
        on_input = AsyncMock()

        with patch("mita.ui.repl.PromptSession") as mock_session_cls:
            session = MagicMock()
            session.prompt_async = AsyncMock(side_effect=EOFError)
            mock_session_cls.return_value = session
            with patch("mita.ui.repl.display_welcome"), patch("mita.ui.repl.display_goodbye"):
                await repl_loop(console, on_input)

        on_input.assert_not_called()

    @pytest.mark.asyncio()
    async def test_keyboard_interrupt_exits(self) -> None:
        """KeyboardInterrupt should exit the loop gracefully."""
        console = MagicMock()
        on_input = AsyncMock()

        with patch("mita.ui.repl.PromptSession") as mock_session_cls:
            session = MagicMock()
            session.prompt_async = AsyncMock(side_effect=KeyboardInterrupt)
            mock_session_cls.return_value = session
            with patch("mita.ui.repl.display_welcome"), patch("mita.ui.repl.display_goodbye"):
                await repl_loop(console, on_input)

        on_input.assert_not_called()

    @pytest.mark.asyncio()
    async def test_keyboard_interrupt_during_input(self) -> None:
        """KeyboardInterrupt during on_input is caught."""
        console = MagicMock()
        on_input = AsyncMock(side_effect=KeyboardInterrupt)

        with patch("mita.ui.repl.PromptSession") as mock_session_cls:
            session = MagicMock()
            session.prompt_async = AsyncMock(side_effect=["hello", "/quit"])
            mock_session_cls.return_value = session
            with patch("mita.ui.repl.display_welcome"), patch("mita.ui.repl.display_goodbye"):
                await repl_loop(console, on_input)

        on_input.assert_called_once()

    @pytest.mark.asyncio()
    async def test_skill_invocation(self) -> None:
        """Slash commands that match skills should render and run."""
        console = MagicMock()
        on_input = AsyncMock()

        with (
            patch("mita.ui.repl.PromptSession") as mock_session_cls,
            patch("mita.ui.repl._try_render_skill", return_value="rendered prompt"),
        ):
            session = MagicMock()
            session.prompt_async = AsyncMock(side_effect=["/myskill arg", "/quit"])
            mock_session_cls.return_value = session
            with patch("mita.ui.repl.display_welcome"), patch("mita.ui.repl.display_goodbye"):
                await repl_loop(console, on_input)

        on_input.assert_called_once_with("rendered prompt")

    @pytest.mark.asyncio()
    async def test_skill_not_found(self) -> None:
        """Slash commands with no matching skill fall through to on_input."""
        console = MagicMock()
        on_input = AsyncMock()

        with (
            patch("mita.ui.repl.PromptSession") as mock_session_cls,
            patch("mita.ui.repl._try_render_skill", return_value=None),
        ):
            session = MagicMock()
            session.prompt_async = AsyncMock(side_effect=["/unknown", "/quit"])
            mock_session_cls.return_value = session
            with patch("mita.ui.repl.display_welcome"), patch("mita.ui.repl.display_goodbye"):
                await repl_loop(console, on_input)

        on_input.assert_called_once_with("/unknown")

    @pytest.mark.asyncio()
    async def test_skill_interrupt(self) -> None:
        """KeyboardInterrupt during skill execution is caught."""
        console = MagicMock()
        on_input = AsyncMock(side_effect=KeyboardInterrupt)

        with (
            patch("mita.ui.repl.PromptSession") as mock_session_cls,
            patch("mita.ui.repl._try_render_skill", return_value="rendered"),
        ):
            session = MagicMock()
            session.prompt_async = AsyncMock(side_effect=["/myskill", "/quit"])
            mock_session_cls.return_value = session
            with patch("mita.ui.repl.display_welcome"), patch("mita.ui.repl.display_goodbye"):
                await repl_loop(console, on_input)


class TestTryRenderSkill:
    def test_no_matching_skill(self) -> None:
        console = MagicMock()
        with (
            patch("mita.skills.loader.discover_skills", return_value=[]),
            patch("mita.skills.loader.find_skill", return_value=None),
        ):
            result = _try_render_skill("/nonexistent", [], console)
            assert result is None

    def test_matching_skill(self) -> None:
        console = MagicMock()
        mock_skill = MagicMock()
        mock_skill.frontmatter.name = "test-skill"
        with (
            patch("mita.skills.loader.discover_skills", return_value=[mock_skill]),
            patch("mita.skills.loader.find_skill", return_value=mock_skill),
            patch("mita.skills.executor.render_skill", return_value="rendered output"),
        ):
            result = _try_render_skill("/test-skill", ["/path"], console)
            assert result == "rendered output"


class TestReloadAndHelp:
    """Audit finding S7: /reload re-reads config/memory without a restart."""

    @pytest.mark.asyncio()
    async def test_reload_command(self) -> None:
        console = MagicMock()
        on_input = AsyncMock()
        on_reload = MagicMock()

        with patch("mita.ui.repl.PromptSession") as mock_session_cls:
            session = MagicMock()
            session.prompt_async = AsyncMock(side_effect=["/reload", "/quit"])
            mock_session_cls.return_value = session
            with patch("mita.ui.repl.display_welcome"), patch("mita.ui.repl.display_goodbye"):
                await repl_loop(console, on_input, on_reload=on_reload)

        on_reload.assert_called_once()
        on_input.assert_not_called()

    @pytest.mark.asyncio()
    async def test_help_command(self) -> None:
        console = MagicMock()
        on_input = AsyncMock()

        with patch("mita.ui.repl.PromptSession") as mock_session_cls:
            session = MagicMock()
            session.prompt_async = AsyncMock(side_effect=["/help", "/quit"])
            mock_session_cls.return_value = session
            with patch("mita.ui.repl.display_welcome"), patch("mita.ui.repl.display_goodbye"):
                await repl_loop(console, on_input)

        on_input.assert_not_called()
