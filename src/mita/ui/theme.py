"""Color scheme and style constants for terminal UI."""

from __future__ import annotations

from rich.style import Style
from rich.theme import Theme

# Style constants
STYLES = {
    "mita.prompt": Style(color="cyan", bold=True),
    "mita.assistant": Style(color="white"),
    "mita.tool_name": Style(color="yellow", bold=True),
    "mita.tool_result": Style(color="green"),
    "mita.tool_error": Style(color="red"),
    "mita.info": Style(color="blue"),
    "mita.warning": Style(color="yellow"),
    "mita.error": Style(color="red", bold=True),
    "mita.dim": Style(dim=True),
    "mita.success": Style(color="green", bold=True),
    "mita.token_count": Style(color="cyan", dim=True),
}

MITA_THEME = Theme(STYLES)
