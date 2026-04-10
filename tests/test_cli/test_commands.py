"""Tests for CLI commands."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from typer.testing import CliRunner

from mita.cli import app

runner = CliRunner()


# ── Version / Verbose ────────────────────────────────────────────


class TestVersionFlag:
    def test_version(self) -> None:
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "mita" in result.output

    def test_verbose(self) -> None:
        result = runner.invoke(app, ["--verbose", "--help"])
        assert result.exit_code == 0


class TestChatPermissionFlag:
    def test_invalid_permission_mode(self) -> None:
        result = runner.invoke(app, ["chat", "--permission", "yolo"])
        assert result.exit_code != 0
        assert "Invalid permission mode" in result.output

    def test_chat_help_shows_permission_flag(self) -> None:
        result = runner.invoke(app, ["chat", "--help"])
        assert result.exit_code == 0
        assert "--permission" in result.output


# ── Config commands ──────────────────────────────────────────────


class TestConfigCommands:
    def test_config_show(self) -> None:
        result = runner.invoke(app, ["config", "show"])
        assert result.exit_code == 0

    def test_config_path(self) -> None:
        result = runner.invoke(app, ["config", "path"])
        assert result.exit_code == 0
        assert "Global" in result.output


# ── Memory commands ──────────────────────────────────────────────


class TestMemoryCommands:
    def test_memory_show(self) -> None:
        result = runner.invoke(app, ["memory", "show"])
        assert result.exit_code == 0

    def test_memory_path(self) -> None:
        result = runner.invoke(app, ["memory", "path"])
        assert result.exit_code == 0

    def test_memory_edit_global(self) -> None:
        with patch("mita.memory.manager.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = runner.invoke(app, ["memory", "edit", "--global"])
            assert result.exit_code == 0

    def test_memory_edit_project(self, tmp_project: Path) -> None:
        with (
            patch("mita.memory.manager.Path.cwd", return_value=tmp_project),
            patch("mita.memory.manager.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0)
            result = runner.invoke(app, ["memory", "edit", "--project"])
            assert result.exit_code == 0

    def test_memory_edit_default(self, tmp_project: Path) -> None:
        with (
            patch("mita.memory.manager.Path.cwd", return_value=tmp_project),
            patch("mita.memory.manager.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0)
            result = runner.invoke(app, ["memory", "edit"])
            assert result.exit_code == 0

    def test_memory_add_project(self, tmp_project: Path) -> None:
        with patch("mita.memory.manager.Path.cwd", return_value=tmp_project):
            result = runner.invoke(app, ["memory", "add", "test note", "--project"])
            assert result.exit_code == 0
            assert "Added" in result.output

    def test_memory_add_global(self) -> None:
        result = runner.invoke(app, ["memory", "add", "global note", "--global"])
        assert result.exit_code == 0
        assert "Added" in result.output


# ── Models commands ──────────────────────────────────────────────


class TestModelsCommands:
    def test_models_list(self) -> None:
        with patch("mita.models.manager.OllamaClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.list_models.return_value = []
            mock_cls.return_value = mock_client
            result = runner.invoke(app, ["models", "list"])
            assert result.exit_code == 0

    def test_models_pull(self) -> None:
        with patch("mita.models.manager.OllamaClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.pull.return_value = [{"status": "success"}]
            mock_cls.return_value = mock_client
            result = runner.invoke(app, ["models", "pull", "test-model"])
            assert result.exit_code == 0

    def test_models_remove(self) -> None:
        with patch("mita.models.manager.OllamaClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.delete.return_value = None
            mock_cls.return_value = mock_client
            result = runner.invoke(app, ["models", "remove", "test-model"])
            assert result.exit_code == 0

    def test_models_recommend(self) -> None:
        with patch("mita.models.manager.detect_hardware") as mock_hw:
            mock_hw.return_value = MagicMock(
                ram_gb=16,
                cpu_cores=8,
                cpu_name="Test CPU",
                gpus=[],
                os="darwin",
                apple_silicon=False,
                unified_memory=False,
                available_vram_gb=12.0,
            )
            with patch("mita.models.manager.recommend_models", return_value=[]):
                result = runner.invoke(app, ["models", "recommend"])
                assert result.exit_code == 0

    def test_models_info(self) -> None:
        with patch("mita.models.manager.OllamaClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.show.return_value = {
                "modelinfo": {"general.parameter_count": 7000000000},
                "details": {"family": "test", "parameter_size": "7B", "quantization_level": "Q4_0"},
            }
            mock_cls.return_value = mock_client
            result = runner.invoke(app, ["models", "info", "test-model"])
            assert result.exit_code == 0

    def test_models_default(self) -> None:
        with patch("mita.models.manager.find_model", return_value=None):
            result = runner.invoke(app, ["models", "default", "test-model:7b"])
            assert result.exit_code == 0

    def test_models_hardware(self) -> None:
        with patch("mita.models.manager.detect_hardware") as mock_hw:
            mock_hw.return_value = MagicMock(
                ram_gb=16,
                cpu_cores=8,
                cpu_name="Test CPU",
                gpus=[],
                os="darwin",
                apple_silicon=False,
                unified_memory=False,
                available_vram_gb=12.0,
            )
            result = runner.invoke(app, ["models", "hardware"])
            assert result.exit_code == 0


# ── Ollama commands ──────────────────────────────────────────────


class TestOllamaCommands:
    def test_ollama_start_already_running(self) -> None:
        with patch("mita.models.server.is_server_running", return_value=True):
            result = runner.invoke(app, ["ollama", "start"])
            assert result.exit_code == 0
            assert "already running" in result.output

    def test_ollama_start_not_running(self) -> None:
        with (
            patch("mita.models.server.is_server_running", return_value=False),
            patch("mita.models.server.start_server", return_value=True),
        ):
            result = runner.invoke(app, ["ollama", "start"])
            assert result.exit_code == 0

    def test_ollama_start_failed(self) -> None:
        with (
            patch("mita.models.server.is_server_running", return_value=False),
            patch("mita.models.server.start_server", return_value=False),
        ):
            result = runner.invoke(app, ["ollama", "start"])
            assert result.exit_code != 0

    def test_ollama_stop_not_managed(self) -> None:
        with patch("mita.models.server.is_managed", return_value=False):
            result = runner.invoke(app, ["ollama", "stop"])
            assert result.exit_code == 0
            assert "not started by Mita" in result.output

    def test_ollama_stop_managed(self) -> None:
        with (
            patch("mita.models.server.is_managed", return_value=True),
            patch("mita.models.server.stop_server"),
        ):
            result = runner.invoke(app, ["ollama", "stop"])
            assert result.exit_code == 0

    def test_ollama_status_running(self) -> None:
        with (
            patch("mita.models.server.find_ollama_binary", return_value="/usr/local/bin/ollama"),
            patch("mita.models.server.is_server_running", return_value=True),
            patch("mita.models.server.is_managed", return_value=False),
        ):
            result = runner.invoke(app, ["ollama", "status"])
            assert result.exit_code == 0
            assert "running" in result.output

    def test_ollama_status_not_running(self) -> None:
        with (
            patch("mita.models.server.find_ollama_binary", return_value=None),
            patch("mita.models.server.is_server_running", return_value=False),
        ):
            result = runner.invoke(app, ["ollama", "status"])
            assert result.exit_code == 0
            assert "not running" in result.output


# ── Index commands ───────────────────────────────────────────────


class TestIndexCommands:
    def test_index_status(self) -> None:
        with patch("mita.index.manager.show_index_status", new_callable=AsyncMock):
            result = runner.invoke(app, ["index", "status"])
            assert result.exit_code == 0

    def test_index_clear(self) -> None:
        with patch("mita.index.manager.clear_index", new_callable=AsyncMock):
            result = runner.invoke(app, ["index", "clear"])
            assert result.exit_code == 0


# ── Skills commands ──────────────────────────────────────────────


class TestSkillsCommands:
    def test_skills_list(self) -> None:
        with patch("mita.skills.manager.discover_skills", return_value=[]):
            result = runner.invoke(app, ["skills", "list"])
            assert result.exit_code == 0

    def test_skills_path(self) -> None:
        result = runner.invoke(app, ["skills", "path"])
        assert result.exit_code == 0


# ── Hooks commands ───────────────────────────────────────────────


class TestHooksCommands:
    def test_hooks_list(self) -> None:
        result = runner.invoke(app, ["hooks", "list"])
        assert result.exit_code == 0


# ── Plugins commands ─────────────────────────────────────────────


class TestPluginsCommands:
    def test_plugins_list_no_plugins(self) -> None:
        from mita.config.schema import MitaConfig

        config = MitaConfig(plugins=[])
        with patch("mita.cli.load_config", return_value=config):
            result = runner.invoke(app, ["plugins", "list"])
            assert result.exit_code == 0
            assert "No plugins" in result.output

    def test_plugins_add_no_args(self) -> None:
        result = runner.invoke(app, ["plugins", "add", "test-plugin"])
        assert result.exit_code != 0

    def test_plugins_add_command(self, tmp_project_with_mita: Path) -> None:
        settings_file = tmp_project_with_mita / ".mita" / "settings.toml"
        settings_file.write_text("")
        with patch(
            "mita.config.defaults.ensure_project_config_path",
            return_value=settings_file,
        ):
            result = runner.invoke(
                app, ["plugins", "add", "test-plugin", "--command", "python -m server"]
            )
            assert result.exit_code == 0
            content = settings_file.read_text()
            assert "test-plugin" in content

    def test_plugins_add_url(self, tmp_project_with_mita: Path) -> None:
        settings_file = tmp_project_with_mita / ".mita" / "settings.toml"
        settings_file.write_text("")
        with patch(
            "mita.config.defaults.ensure_project_config_path",
            return_value=settings_file,
        ):
            result = runner.invoke(
                app, ["plugins", "add", "sse-plugin", "--url", "http://localhost:8080"]
            )
            assert result.exit_code == 0

    def test_plugins_add_no_project(self) -> None:
        with patch("mita.config.defaults.ensure_project_config_path", return_value=None):
            result = runner.invoke(
                app, ["plugins", "add", "test-plugin", "--command", "python -m server"]
            )
            assert result.exit_code != 0

    def test_plugins_remove_not_found(self, tmp_project_with_mita: Path) -> None:
        settings_file = tmp_project_with_mita / ".mita" / "settings.toml"
        settings_file.write_text("")
        with patch(
            "mita.config.defaults.get_project_config_path",
            return_value=settings_file,
        ):
            result = runner.invoke(app, ["plugins", "remove", "nonexistent"])
            assert result.exit_code == 0
            assert "not found" in result.output

    def test_plugins_remove_no_project(self) -> None:
        with patch("mita.config.defaults.get_project_config_path", return_value=None):
            result = runner.invoke(app, ["plugins", "remove", "test"])
            assert result.exit_code != 0

    def test_plugins_test_not_configured(self) -> None:
        result = runner.invoke(app, ["plugins", "test", "nonexistent"])
        assert result.exit_code != 0


# ── Doctor command ───────────────────────────────────────────────


class TestDoctorCommand:
    def test_doctor_runs(self) -> None:
        with (
            patch("mita.models.server.find_ollama_binary", return_value="/usr/local/bin/ollama"),
            patch("mita.models.server.is_server_running", return_value=False),
            patch("mita.memory.discovery.discover_memory_files", return_value=[]),
            patch("mita.index.store.IndexStore.exists", return_value=False),
        ):
            result = runner.invoke(app, ["doctor"])
            assert result.exit_code == 0
            assert "Python" in result.output

    def test_doctor_all_healthy(self, tmp_path: Path) -> None:
        mem_file = tmp_path / "MITA.md"
        mem_file.write_text("# test")
        with (
            patch("mita.models.server.find_ollama_binary", return_value="/usr/local/bin/ollama"),
            patch("mita.models.server.is_server_running", return_value=True),
            patch("mita.models.server.is_managed", return_value=False),
            patch("mita.memory.discovery.discover_memory_files", return_value=[mem_file]),
            patch("mita.index.store.IndexStore.exists", return_value=True),
            patch("mita.models.ollama_client.OllamaClient.is_model_installed", return_value=True),
        ):
            result = runner.invoke(app, ["doctor"])
            assert result.exit_code == 0
