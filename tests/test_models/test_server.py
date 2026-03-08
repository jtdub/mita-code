"""Tests for Ollama server lifecycle management."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mita.models.ollama_client import OllamaModelInfo
from mita.models.server import (
    _read_pid,
    _remove_pid_file,
    _write_pid,
    ensure_model,
    ensure_server,
    find_ollama_binary,
    is_managed,
    is_server_running,
    start_server,
    stop_server,
)


@pytest.fixture(autouse=True)
def _isolate_pid_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the PID file to a temp dir so tests don't interfere."""
    import mita.models.server as mod

    pid_dir = tmp_path / "mita"
    pid_dir.mkdir()
    monkeypatch.setattr(mod, "_PID_DIR", pid_dir)
    monkeypatch.setattr(mod, "_PID_FILE", pid_dir / "ollama.pid")


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


class TestPidFile:
    def test_write_and_read(self) -> None:
        with patch("mita.models.server.os.kill"):
            _write_pid(12345)
            assert _read_pid() == 12345

    def test_read_missing(self) -> None:
        assert _read_pid() is None

    def test_read_stale_pid(self) -> None:
        _write_pid(99999)
        with patch("mita.models.server.os.kill", side_effect=ProcessLookupError):
            assert _read_pid() is None

    def test_remove(self) -> None:
        _write_pid(12345)
        _remove_pid_file()
        assert _read_pid() is None

    def test_remove_missing(self) -> None:
        # Should not raise
        _remove_pid_file()


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
        mock_proc.pid = 42

        health_calls = iter([False, True])

        with (
            patch("mita.models.server.is_server_running", side_effect=health_calls),
            patch("mita.models.server.find_ollama_binary", return_value="/usr/bin/ollama"),
            patch("mita.models.server.subprocess.Popen", return_value=mock_proc),
            patch("mita.models.server.time.sleep"),
        ):
            assert start_server(timeout=5) is True

    def test_writes_pid_file(self, tmp_path: Path) -> None:
        import mita.models.server as mod

        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.poll.return_value = None
        mock_proc.pid = 42

        with (
            patch("mita.models.server.is_server_running", side_effect=[False, True]),
            patch("mita.models.server.find_ollama_binary", return_value="/usr/bin/ollama"),
            patch("mita.models.server.subprocess.Popen", return_value=mock_proc),
            patch("mita.models.server.time.sleep"),
        ):
            start_server(timeout=5)
            assert mod._PID_FILE.read_text() == "42"

    def test_process_exits_unexpectedly(self) -> None:
        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.poll.return_value = 1
        mock_proc.pid = 42

        with (
            patch("mita.models.server.is_server_running", return_value=False),
            patch("mita.models.server.find_ollama_binary", return_value="/usr/bin/ollama"),
            patch("mita.models.server.subprocess.Popen", return_value=mock_proc),
        ):
            assert start_server(timeout=5) is False

    def test_popen_oserror(self) -> None:
        with (
            patch("mita.models.server.is_server_running", return_value=False),
            patch("mita.models.server.find_ollama_binary", return_value="/usr/bin/ollama"),
            patch("mita.models.server.subprocess.Popen", side_effect=OSError("exec failed")),
        ):
            assert start_server() is False

    def test_detached_session(self) -> None:
        """Verify start_new_session=True is passed."""
        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.poll.return_value = None
        mock_proc.pid = 99

        with (
            patch("mita.models.server.is_server_running", side_effect=[False, True]),
            patch("mita.models.server.find_ollama_binary", return_value="/usr/bin/ollama"),
            patch("mita.models.server.subprocess.Popen", return_value=mock_proc) as mock_popen,
            patch("mita.models.server.time.sleep"),
        ):
            start_server(timeout=5)
            _, kwargs = mock_popen.call_args
            assert kwargs.get("start_new_session") is True


class TestStopServer:
    def test_nothing_to_stop(self) -> None:
        assert stop_server() is False

    def test_stops_tracked_process(self) -> None:
        _write_pid(12345)
        with (
            patch("mita.models.server.os.kill") as mock_kill,
            patch("mita.models.server.time.sleep"),
        ):
            # First os.kill(pid, 0) in _read_pid succeeds (process alive)
            # Then SIGTERM, then os.kill(pid, 0) raises ProcessLookupError (dead)
            mock_kill.side_effect = [None, None, ProcessLookupError]
            assert stop_server() is True

    def test_removes_pid_file(self) -> None:
        import mita.models.server as mod

        _write_pid(12345)
        with (
            patch("mita.models.server.os.kill") as mock_kill,
            patch("mita.models.server.time.sleep"),
        ):
            mock_kill.side_effect = [None, None, ProcessLookupError]
            stop_server()
            assert not mod._PID_FILE.exists()


class TestIsManaged:
    def test_not_managed(self) -> None:
        assert is_managed() is False

    def test_managed_with_pid(self) -> None:
        _write_pid(12345)
        with patch("mita.models.server.os.kill"):
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


class TestEnsureModel:
    def _make_models(self, *names: str) -> list[OllamaModelInfo]:
        return [OllamaModelInfo(name=n, size_gb=1.0) for n in names]

    def test_model_already_installed(self) -> None:
        with patch("mita.models.server.OllamaClient") as mock_cls:
            mock_cls.return_value.list_models.return_value = self._make_models("qwen2.5-coder:7b")
            assert ensure_model("qwen2.5-coder:7b") is True

    def test_model_installed_without_tag(self) -> None:
        with patch("mita.models.server.OllamaClient") as mock_cls:
            mock_cls.return_value.list_models.return_value = self._make_models(
                "qwen2.5-coder:latest"
            )
            assert ensure_model("qwen2.5-coder") is True

    def test_model_not_installed_pulls(self) -> None:
        with patch("mita.models.server.OllamaClient") as mock_cls:
            mock_cls.return_value.list_models.return_value = []
            mock_cls.return_value.pull.return_value = iter([{"status": "success"}])
            assert ensure_model("qwen2.5-coder:7b") is True

    def test_pull_failure(self) -> None:
        import ollama as ollama_lib

        with patch("mita.models.server.OllamaClient") as mock_cls:
            mock_cls.return_value.list_models.return_value = []
            mock_cls.return_value.pull.side_effect = ollama_lib.ResponseError("not found")
            assert ensure_model("bad-model") is False

    def test_connection_error(self) -> None:
        with patch("mita.models.server.OllamaClient") as mock_cls:
            mock_cls.return_value.list_models.side_effect = ConnectionError
            assert ensure_model("test-model") is False
