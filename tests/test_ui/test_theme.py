"""Tests for UI theme."""

from __future__ import annotations

from mita.ui.theme import MITA_THEME, STYLES


class TestTheme:
    def test_styles_not_empty(self) -> None:
        assert len(STYLES) > 0

    def test_theme_has_styles(self) -> None:
        assert MITA_THEME is not None

    def test_expected_styles_present(self) -> None:
        expected = [
            "mita.prompt",
            "mita.error",
            "mita.tool_name",
            "mita.tool_result",
            "mita.dim",
        ]
        for name in expected:
            assert name in STYLES
