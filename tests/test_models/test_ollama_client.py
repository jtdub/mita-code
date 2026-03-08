"""Tests for OllamaClient wrapper."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import ollama
import pytest

from mita.models.ollama_client import OllamaClient


class TestOllamaClientInit:
    def test_default_host_and_timeout(self) -> None:
        with patch("mita.models.ollama_client.ollama.Client") as mock_client_cls:
            OllamaClient()
            mock_client_cls.assert_called_once_with(host="http://localhost:11434", timeout=120)

    def test_custom_host_and_timeout(self) -> None:
        with patch("mita.models.ollama_client.ollama.Client") as mock_client_cls:
            OllamaClient(host="http://myhost:1234", timeout=60)
            mock_client_cls.assert_called_once_with(host="http://myhost:1234", timeout=60)


class TestIsRunning:
    def test_returns_true_when_reachable(self) -> None:
        with patch("mita.models.ollama_client.ollama.Client") as mock_cls:
            mock_inner = MagicMock()
            mock_cls.return_value = mock_inner
            client = OllamaClient()
            assert client.is_running() is True
            mock_inner.list.assert_called_once()

    @pytest.mark.parametrize("exc", [ConnectionError, OSError, ollama.ResponseError, TimeoutError])
    def test_returns_false_on_expected_errors(self, exc: type) -> None:
        with patch("mita.models.ollama_client.ollama.Client") as mock_cls:
            mock_inner = MagicMock()
            if exc == ollama.ResponseError:
                mock_inner.list.side_effect = exc("err")
            else:
                mock_inner.list.side_effect = exc()
            mock_cls.return_value = mock_inner
            client = OllamaClient()
            assert client.is_running() is False


class TestListModels:
    def _make_model_obj(
        self,
        name: str = "qwen2.5-coder:7b",
        size: int = 4_000_000_000,
        param_size: str = "7B",
        quant: str = "Q4_0",
        family: str = "qwen2",
    ) -> MagicMock:
        details = MagicMock()
        details.parameter_size = param_size
        details.quantization_level = quant
        details.family = family
        m = MagicMock()
        m.model = name
        m.size = size
        m.details = details
        m.modified_at = "2025-01-01"
        return m

    def test_list_models_returns_info(self) -> None:
        with patch("mita.models.ollama_client.ollama.Client") as mock_cls:
            mock_inner = MagicMock()
            response = MagicMock()
            response.models = [self._make_model_obj()]
            mock_inner.list.return_value = response
            mock_cls.return_value = mock_inner

            client = OllamaClient()
            models = client.list_models()
            assert len(models) == 1
            assert models[0].name == "qwen2.5-coder:7b"
            assert models[0].parameter_size == "7B"
            assert models[0].family == "qwen2"

    def test_list_models_empty(self) -> None:
        with patch("mita.models.ollama_client.ollama.Client") as mock_cls:
            mock_inner = MagicMock()
            response = MagicMock()
            response.models = []
            mock_inner.list.return_value = response
            mock_cls.return_value = mock_inner

            client = OllamaClient()
            assert client.list_models() == []

    def test_list_models_dict_details(self) -> None:
        """When details is a dict instead of an object."""
        with patch("mita.models.ollama_client.ollama.Client") as mock_cls:
            mock_inner = MagicMock()
            m = MagicMock()
            m.model = "test:latest"
            m.size = 2_000_000_000
            m.details = {"parameter_size": "3B", "quantization_level": "Q8_0", "family": "llama"}
            m.modified_at = ""
            response = MagicMock()
            response.models = [m]
            mock_inner.list.return_value = response
            mock_cls.return_value = mock_inner

            client = OllamaClient()
            models = client.list_models()
            assert models[0].parameter_size == "3B"
            assert models[0].quantization == "Q8_0"
            assert models[0].family == "llama"


class TestPull:
    def test_pull_stream(self) -> None:
        with patch("mita.models.ollama_client.ollama.Client") as mock_cls:
            mock_inner = MagicMock()
            mock_inner.pull.return_value = [
                {"status": "downloading", "completed": 50, "total": 100},
                {"status": "success", "completed": 100, "total": 100},
            ]
            mock_cls.return_value = mock_inner

            client = OllamaClient()
            updates = list(client.pull("test-model", stream=True))
            assert len(updates) == 2
            assert updates[0]["status"] == "downloading"
            assert updates[1]["status"] == "success"

    def test_pull_stream_object_chunks(self) -> None:
        """When streaming returns objects instead of dicts."""
        with patch("mita.models.ollama_client.ollama.Client") as mock_cls:
            mock_inner = MagicMock()
            chunk = MagicMock()
            chunk.status = "done"
            mock_inner.pull.return_value = [chunk]
            mock_cls.return_value = mock_inner

            client = OllamaClient()
            updates = list(client.pull("test-model", stream=True))
            assert len(updates) == 1
            assert updates[0]["status"] == "done"

    def test_pull_no_stream(self) -> None:
        with patch("mita.models.ollama_client.ollama.Client") as mock_cls:
            mock_inner = MagicMock()
            mock_cls.return_value = mock_inner

            client = OllamaClient()
            updates = list(client.pull("test-model", stream=False))
            assert len(updates) == 1
            assert updates[0]["status"] == "success"
            mock_inner.pull.assert_called_once_with("test-model", stream=False)


class TestRemove:
    def test_remove_calls_delete(self) -> None:
        with patch("mita.models.ollama_client.ollama.Client") as mock_cls:
            mock_inner = MagicMock()
            mock_cls.return_value = mock_inner

            client = OllamaClient()
            client.remove("test-model")
            mock_inner.delete.assert_called_once_with("test-model")


class TestShow:
    def test_show_dict_response(self) -> None:
        with patch("mita.models.ollama_client.ollama.Client") as mock_cls:
            mock_inner = MagicMock()
            mock_inner.show.return_value = {"details": {"family": "llama"}}
            mock_cls.return_value = mock_inner

            client = OllamaClient()
            result = client.show("test-model")
            assert result == {"details": {"family": "llama"}}

    def test_show_object_response(self) -> None:
        with patch("mita.models.ollama_client.ollama.Client") as mock_cls:
            mock_inner = MagicMock()
            resp = MagicMock(spec=[])  # no dict interface
            resp.modelfile = "FROM llama"
            resp.parameters = ""
            resp.template = ""
            resp.details = {"family": "llama"}
            resp.model_info = {}
            mock_inner.show.return_value = resp
            mock_cls.return_value = mock_inner

            client = OllamaClient()
            result = client.show("test-model")
            assert result["modelfile"] == "FROM llama"
            assert result["details"] == {"family": "llama"}
