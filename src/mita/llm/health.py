"""Reachability check for OpenAI-compatible backends (finding C5)."""

from __future__ import annotations

import httpx


def backend_reachable(api_base: str, api_key: str | None, timeout: float = 5.0) -> bool:
    """Return True if the backend answers at all at ``<api_base>/models``.

    Any HTTP response (including 404/401) means the server is up — only a connection
    failure or timeout counts as unreachable, so a backend whose endpoint shape differs
    is not falsely reported as down.
    """
    url = api_base.rstrip("/") + "/models"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        httpx.get(url, headers=headers, timeout=timeout)
        return True
    except httpx.HTTPError:
        return False
