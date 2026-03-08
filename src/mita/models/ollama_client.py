"""Wrapper around the Ollama Python client."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import ollama
from pydantic import BaseModel


class OllamaModelInfo(BaseModel):
    """Info about an installed Ollama model."""

    name: str
    size_gb: float
    parameter_size: str = ""
    quantization: str = ""
    family: str = ""
    modified_at: str = ""


class OllamaClient:
    """Thin wrapper around the ollama Python client."""

    def __init__(self, host: str = "http://localhost:11434") -> None:
        self._client = ollama.Client(host=host)
        self._host = host

    def is_running(self) -> bool:
        """Check if the Ollama server is reachable."""
        try:
            self._client.list()
            return True
        except Exception:  # noqa: BLE001
            return False

    def list_models(self) -> list[OllamaModelInfo]:
        """List all installed models."""
        response = self._client.list()
        models: list[OllamaModelInfo] = []
        for m in response.models:
            details: Any = m.details or {}
            size_gb = round((m.size or 0) / (1024**3), 2)

            # Handle details as either dict or object
            if isinstance(details, dict):
                param_size = details.get("parameter_size", "")
                quant = details.get("quantization_level", "")
                family = details.get("family", "")
            else:
                param_size = getattr(details, "parameter_size", "") or ""
                quant = getattr(details, "quantization_level", "") or ""
                family = getattr(details, "family", "") or ""

            models.append(
                OllamaModelInfo(
                    name=m.model or "",
                    size_gb=size_gb,
                    parameter_size=param_size,
                    quantization=quant,
                    family=family,
                    modified_at=str(m.modified_at or ""),
                )
            )
        return models

    def pull(self, model_name: str, stream: bool = True) -> Iterator[dict[str, Any]]:
        """Pull a model from the Ollama registry.

        Yields progress dicts with keys like 'status', 'completed', 'total'.
        """
        if stream:
            response = self._client.pull(model_name, stream=True)
            for chunk in response:
                if isinstance(chunk, dict):
                    yield chunk
                else:
                    yield {"status": getattr(chunk, "status", str(chunk))}
        else:
            self._client.pull(model_name, stream=False)
            yield {"status": "success"}

    def remove(self, model_name: str) -> None:
        """Remove an installed model."""
        self._client.delete(model_name)

    def show(self, model_name: str) -> dict[str, Any]:
        """Show details about a model."""
        response = self._client.show(model_name)
        if isinstance(response, dict):
            return response
        # Convert response object to dict
        return {
            "modelfile": getattr(response, "modelfile", ""),
            "parameters": getattr(response, "parameters", ""),
            "template": getattr(response, "template", ""),
            "details": getattr(response, "details", {}),
            "model_info": getattr(response, "model_info", {}),
        }


class ModelPullProgress(BaseModel):
    """Progress update during a model pull."""

    status: str
    completed: int = 0
    total: int = 0
    percent: float = 0.0
