"""Tests for model manager CLI command handlers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import ollama as ollama_lib
import pytest

from mita.models.manager import (
    list_models,
    pull_model,
    remove_model,
    set_default_model,
    show_hardware,
    show_model_info,
    show_recommendations,
)
from mita.models.ollama_client import OllamaModelInfo

CaptureFixture = pytest.CaptureFixture[str]


@pytest.fixture()
def mock_config() -> MagicMock:
    """Shared mock MitaConfig."""
    cfg = MagicMock()
    cfg.ollama.host = "http://localhost:11434"
    cfg.ollama.timeout = 120
    cfg.ollama.auto_manage = True
    return cfg


@pytest.fixture()
def mock_client() -> MagicMock:
    """Shared mock OllamaClient."""
    client = MagicMock()
    client.is_running.return_value = True
    return client


@pytest.fixture()
def _patch_client(mock_client: MagicMock, mock_config: MagicMock) -> MagicMock:  # type: ignore[misc]
    """Patch _get_client to return our mock client and config."""
    with patch("mita.models.manager._get_client", return_value=(mock_client, mock_config)):
        yield mock_client


class TestListModels:
    def test_lists_installed_models(self, _patch_client: MagicMock, capsys: CaptureFixture) -> None:
        _patch_client.list_models.return_value = [
            OllamaModelInfo(
                name="qwen2.5-coder:7b",
                size_gb=4.0,
                parameter_size="7B",
                quantization="Q4_0",
                family="qwen2",
            )
        ]
        list_models()
        output = capsys.readouterr().out
        assert "qwen2.5-coder:7b" in output

    def test_no_models_installed(self, _patch_client: MagicMock, capsys: CaptureFixture) -> None:
        _patch_client.list_models.return_value = []
        list_models()
        output = capsys.readouterr().out
        assert "No models installed" in output

    def test_ollama_not_running(
        self, _patch_client: MagicMock, mock_config: MagicMock, capsys: CaptureFixture
    ) -> None:
        mock_config.ollama.auto_manage = False
        _patch_client.is_running.return_value = False
        list_models()
        output = capsys.readouterr().out
        assert "Cannot connect to Ollama" in output


class TestPullModel:
    def test_successful_pull(self, _patch_client: MagicMock, capsys: CaptureFixture) -> None:
        _patch_client.pull.return_value = iter(
            [{"status": "success", "completed": 100, "total": 100}]
        )
        pull_model("test-model")
        output = capsys.readouterr().out
        assert "Successfully pulled" in output

    def test_pull_failure(self, _patch_client: MagicMock, capsys: CaptureFixture) -> None:
        _patch_client.pull.side_effect = ollama_lib.ResponseError("not found")
        pull_model("bad-model")
        output = capsys.readouterr().out
        assert "Failed to pull" in output

    def test_pull_ollama_not_running(
        self, _patch_client: MagicMock, mock_config: MagicMock, capsys: CaptureFixture
    ) -> None:
        mock_config.ollama.auto_manage = False
        _patch_client.is_running.return_value = False
        pull_model("test-model")
        output = capsys.readouterr().out
        assert "Cannot connect to Ollama" in output


class TestRemoveModel:
    def test_successful_remove(self, _patch_client: MagicMock, capsys: CaptureFixture) -> None:
        remove_model("test-model")
        _patch_client.remove.assert_called_once_with("test-model")
        output = capsys.readouterr().out
        assert "Removed test-model" in output

    def test_remove_failure(self, _patch_client: MagicMock, capsys: CaptureFixture) -> None:
        _patch_client.remove.side_effect = ollama_lib.ResponseError("not found")
        remove_model("bad-model")
        output = capsys.readouterr().out
        assert "Failed to remove" in output


class TestShowModelInfo:
    def test_registry_model(self, _patch_client: MagicMock, capsys: CaptureFixture) -> None:
        _patch_client.show.return_value = {"details": {"family": "qwen2"}}
        show_model_info("qwen2.5-coder:7b")
        output = capsys.readouterr().out
        assert "qwen2.5-coder:7b" in output

    def test_unknown_model_not_found(
        self, _patch_client: MagicMock, capsys: CaptureFixture
    ) -> None:
        _patch_client.show.side_effect = ollama_lib.ResponseError("not found")
        show_model_info("nonexistent-model")
        output = capsys.readouterr().out
        assert "not found" in output.lower()


class TestShowRecommendations:
    @patch("mita.models.manager.detect_hardware")
    def test_shows_recommendations(self, mock_detect: MagicMock, capsys: CaptureFixture) -> None:
        from mita.models.hardware import GPUInfo, HardwareInfo

        mock_detect.return_value = HardwareInfo(
            ram_gb=32,
            cpu_cores=8,
            cpu_name="Test CPU",
            gpus=[GPUInfo(name="Test GPU", vram_gb=24, vendor="nvidia")],
            os="linux",
        )
        show_recommendations()
        output = capsys.readouterr().out
        assert "Hardware Detected" in output
        assert "Recommended Models" in output

    @patch("mita.models.manager.detect_hardware")
    def test_no_models_fit(self, mock_detect: MagicMock, capsys: CaptureFixture) -> None:
        from mita.models.hardware import HardwareInfo

        mock_detect.return_value = HardwareInfo(
            ram_gb=1,
            cpu_cores=1,
            cpu_name="Test CPU",
            gpus=[],
            os="linux",
        )
        show_recommendations()
        output = capsys.readouterr().out
        assert "No models" in output


class TestSetDefaultModel:
    def test_known_model(self, capsys: CaptureFixture) -> None:
        set_default_model("qwen2.5-coder:7b")
        output = capsys.readouterr().out
        assert "in the registry" in output
        assert "config" in output.lower()

    def test_unknown_model(self, capsys: CaptureFixture) -> None:
        set_default_model("custom-model:latest")
        output = capsys.readouterr().out
        assert "not in the curated registry" in output


class TestShowHardware:
    @patch("mita.models.manager.detect_hardware")
    def test_shows_hardware_table(self, mock_detect: MagicMock, capsys: CaptureFixture) -> None:
        from mita.models.hardware import GPUInfo, HardwareInfo

        mock_detect.return_value = HardwareInfo(
            ram_gb=16,
            cpu_cores=4,
            cpu_name="Test CPU",
            gpus=[GPUInfo(name="NVIDIA RTX 4090", vram_gb=24, vendor="nvidia")],
            os="linux",
        )
        show_hardware()
        output = capsys.readouterr().out
        assert "Hardware Information" in output
        assert "Test CPU" in output
        assert "NVIDIA RTX 4090" in output
