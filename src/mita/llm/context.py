"""Best-effort context-window probing per backend (finding C5).

The static default (32768) is wrong for a llama.cpp GGUF loaded at 4096 or a vLLM
`--max-model-len`. Probe the backend when possible; fall back to the configured value.
Results are cached per (provider, base_url, model) so a chat session probes once.
"""

from __future__ import annotations

import logging

import httpx

from mita.config.schema import MitaConfig
from mita.llm.providers import resolve_backend

_logger = logging.getLogger(__name__)
_cache: dict[tuple[str, str, str], int] = {}


async def resolve_context_window(config: MitaConfig) -> int:
    """Return the effective context window: explicit int, probed value, or the fallback."""
    setting = config.llm.context_probe
    fallback = config.model.context_window

    if setting == "off":
        return fallback
    if setting != "auto":
        try:
            return int(setting)
        except ValueError:
            return fallback

    backend = resolve_backend(config)
    key = (config.llm.provider.value, backend.api_base, config.model.default)
    if key in _cache:
        return _cache[key]

    probed = await _probe(backend.spec.context_probe, backend.api_base, config.model.default)
    result = probed if probed and probed > 0 else fallback
    _cache[key] = result
    return result


async def _probe(strategy: str, api_base: str, model: str) -> int | None:
    """Query the backend for its context length. Returns None on any failure."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            if strategy == "ollama_show":
                resp = await client.post(f"{api_base.rstrip('/')}/api/show", json={"model": model})
                info = resp.json().get("model_info", {})
                for k, v in info.items():
                    if k.endswith(".context_length") and isinstance(v, int):
                        return v
            elif strategy == "openai_models":
                resp = await client.get(f"{api_base.rstrip('/')}/models")
                data = resp.json().get("data", [])
                if data and isinstance(data[0], dict):
                    val = data[0].get("max_model_len")
                    if isinstance(val, int):
                        return val
            elif strategy == "llamacpp_props":
                resp = await client.get(f"{api_base.rstrip('/')}/props")
                body = resp.json()
                gen = body.get("default_generation_settings", {})
                for candidate in (gen.get("n_ctx"), body.get("n_ctx")):
                    if isinstance(candidate, int):
                        return candidate
    except (httpx.HTTPError, ValueError, KeyError) as e:
        _logger.debug("Context probe (%s) failed: %s", strategy, e)
    return None


def clear_cache() -> None:
    """Clear the probe cache (used by tests)."""
    _cache.clear()
