"""Tests for Ollama server lifecycle management."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from mita.models.server import (
    ensure_server,
    find_ollama_binary,
    is_managed,
    is_server_running,
    start_server,
    stop_server,
)


@pytest.fixture(autouse=True)
def _reset_managed_process() -> None:  # type: ignore[misc]
    """Reset the module-level _managed_process between tests."""
    import mita.models.server as mod

    mod._managed_process = None
    yield  # type: ignore[misc]
    # Cleanup: kill any leftover process
    if mod._managed_process is not None:
        try:
            mod._managed_process.kill()
        except (OSError, ProcessLookupError):
            pass
        mod._managed_process = None


class TestFindOllamaBinary:
    def test_found(self) -> None:
        with patch("mita.models.server.shutil.which", return_value="/usr/local/bin/ollama"):
            assert find_ollama_binary() == "/usr/local/bin/ollama"

    def test_not_found(self) -> None:
        with patch("mita.models.server.shutil.which", return_value=None):
            assert find_ollama_binary() is None


class TestIsServerRunning:
    def test_running(self) -> None:
        with patch("mita.models.server.OllamaClient") as mock_cls:
            mock_cls.return_value.is_running.return_value = True
            assert is_server_running() is True

    def test_not_running(self) -> None:
        with patch("mita.models.server.OllamaClient") as mock_cls:
            mock_cls.return_value.is_running.return_value = False
            assert is_server_running() is False


class TestStartServer:
    def test_already_running(self) -> None:
        with patch("mita.models.server.is_server_running", return_value=True):
            assert start_server() is True

    def test_no_binary(self) -> None:
        with (
            patch("mita.models.server.is_server_running", return_value=False),
            patch("mita.models.server.find_ollama_binary", return_value=None),
        ):
            assert start_server() is False

    def test_start_and_becomes_healthy(self) -> None:
        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.poll.return_value = None

        health_calls = iter([False, True])

        with (
            patch("mita.models.server.is_server_running", side_effect=health_calls),
            patch("mita.models.server.find_ollama_binary", return_value="/usr/bin/ollama"),
            patch("mita.models.server.subprocess.Popen", return_value=mock_proc),
            patch("mita.models.server.time.sleep"),
            patch("mita.models.server.atexit.register"),
        ):
            assert start_server(timeout=5) is True

    def test_process_exits_unexpectedly(self) -> None:
        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.poll.return_value = 1  # Already exited

        with (
            patch("mita.models.server.is_server_running", return_value=False),
            patch("mita.models.server.find_ollama_binary", return_value="/usr/bin/ollama"),
            patch("mita.models.server.subprocess.Popen", return_value=mock_proc),
            patch("mita.models.server.atexit.register"),
        ):
            assert start_server(timeout=5) is False

    def test_popen_oserror(self) -> None:
        with (
            patch("mita.models.server.is_server_running", return_value=False),
            patch("mita.models.server.find_ollama_binary", return_value="/usr/bin/ollama"),
            patch("mita.models.server.subprocess.Popen", side_effect=OSError("exec failed")),
        ):
            assert start_server() is False


class TestStopServer:
    def test_nothing_to_stop(self) -> None:
        assert stop_server() is False

    def test_stops_managed_process(self) -> None:
        import mita.models.server as mod

        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.poll.return_value = None  # Still running
        mod._managed_process = mock_proc

        assert stop_server() is True
        mock_proc.terminate.assert_called_once()
        assert mod._managed_process is None


class TestIsManaged:
    def test_not_managed(self) -> None:
        assert is_managed() is False

    def test_managed(self) -> None:
        import mita.models.server as mod

        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.poll.return_value = None
        mod._managed_process = mock_proc

        assert is_managed() is True


class TestEnsureServer:
    def test_already_running(self) -> None:
        with patch("mita.models.server.is_server_running", return_value=True):
            assert ensure_server() is True

    def test_auto_manage_disabled(self) -> None:
        with patch("mita.models.server.is_server_running", return_value=False):
            assert ensure_server(auto_manage=False) is False

    def test_auto_manage_starts(self) -> None:
        with (
            patch("mita.models.server.is_server_running", return_value=False),
            patch("mita.models.server.start_server", return_value=True) as mock_start,
        ):
            assert ensure_server(auto_manage=True) is True
            mock_start.assert_called_once()
