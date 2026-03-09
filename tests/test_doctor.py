"""Tests for the mita doctor command."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from mita.cli import app

runner = CliRunner()


class TestDoctorCommand:
    def test_doctor_runs(self) -> None:
        """Doctor command exits without error."""
        with (
            patch(
                "mita.models.server.find_ollama_binary",
                return_value="/usr/local/bin/ollama",
            ),
            patch("mita.models.server.is_server_running", return_value=False),
            patch("mita.memory.discovery.discover_memory_files", return_value=[]),
            patch("mita.index.store.IndexStore.exists", return_value=False),
        ):
            result = runner.invoke(app, ["doctor"])
            assert result.exit_code == 0
            assert "Python" in result.output

    def test_doctor_shows_python_version(self) -> None:
        """Doctor reports the current Python version."""
        with (
            patch("mita.models.server.find_ollama_binary", return_value=None),
            patch("mita.models.server.is_server_running", return_value=False),
            patch("mita.memory.discovery.discover_memory_files", return_value=[]),
            patch("mita.index.store.IndexStore.exists", return_value=False),
        ):
            result = runner.invoke(app, ["doctor"])
            assert result.exit_code == 0
            assert "Python" in result.output

    def test_doctor_ollama_not_found(self) -> None:
        """Doctor shows error when Ollama binary is missing."""
        with (
            patch("mita.models.server.find_ollama_binary", return_value=None),
            patch("mita.models.server.is_server_running", return_value=False),
            patch("mita.memory.discovery.discover_memory_files", return_value=[]),
            patch("mita.index.store.IndexStore.exists", return_value=False),
        ):
            result = runner.invoke(app, ["doctor"])
            assert result.exit_code == 0
            assert "not found" in result.output or "not installed" in result.output

    def test_doctor_ollama_found(self) -> None:
        """Doctor shows success when Ollama binary is present."""
        with (
            patch(
                "mita.models.server.find_ollama_binary",
                return_value="/usr/bin/ollama",
            ),
            patch("mita.models.server.is_server_running", return_value=False),
            patch("mita.memory.discovery.discover_memory_files", return_value=[]),
            patch("mita.index.store.IndexStore.exists", return_value=False),
        ):
            result = runner.invoke(app, ["doctor"])
            assert result.exit_code == 0
            assert "Ollama binary found" in result.output

    def test_doctor_server_running(self) -> None:
        """Doctor shows Ollama server running when it is."""
        mock_client = MagicMock()
        mock_client.list_models.return_value = []

        with (
            patch(
                "mita.models.server.find_ollama_binary",
                return_value="/usr/bin/ollama",
            ),
            patch("mita.models.server.is_server_running", return_value=True),
            patch("mita.memory.discovery.discover_memory_files", return_value=[]),
            patch(
                "mita.models.ollama_client.OllamaClient",
                return_value=mock_client,
            ),
            patch("mita.index.store.IndexStore.exists", return_value=False),
        ):
            result = runner.invoke(app, ["doctor"])
            assert result.exit_code == 0
            assert "server running" in result.output

    def test_doctor_server_not_running(self) -> None:
        """Doctor shows suggestion when Ollama is not running."""
        with (
            patch(
                "mita.models.server.find_ollama_binary",
                return_value="/usr/bin/ollama",
            ),
            patch("mita.models.server.is_server_running", return_value=False),
            patch("mita.memory.discovery.discover_memory_files", return_value=[]),
            patch("mita.index.store.IndexStore.exists", return_value=False),
        ):
            result = runner.invoke(app, ["doctor"])
            assert result.exit_code == 0
            assert "not running" in result.output

    def test_doctor_memory_found(self) -> None:
        """Doctor reports when memory files are found."""
        from pathlib import Path

        with (
            patch("mita.models.server.find_ollama_binary", return_value=None),
            patch("mita.models.server.is_server_running", return_value=False),
            patch(
                "mita.memory.discovery.discover_memory_files",
                return_value=[Path("/tmp/MITA.md")],
            ),
            patch("mita.index.store.IndexStore.exists", return_value=False),
        ):
            result = runner.invoke(app, ["doctor"])
            assert result.exit_code == 0
            assert "Memory files found" in result.output
