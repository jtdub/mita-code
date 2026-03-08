"""Tests for the streaming module."""

from __future__ import annotations

from mita.llm.streaming import _extract_delta_content


class TestExtractDeltaContent:
    def test_object_style_chunk(self) -> None:
        """Test extracting from an object-style chunk."""

        class Delta:
            content = "hello"

        class Choice:
            delta = Delta()

        class Chunk:
            choices = [Choice()]

        assert _extract_delta_content(Chunk()) == "hello"

    def test_dict_style_chunk(self) -> None:
        """Test extracting from a dict-style chunk."""
        chunk = {"choices": [{"delta": {"content": "world"}}]}
        assert _extract_delta_content(chunk) == "world"

    def test_empty_choices(self) -> None:
        chunk = {"choices": []}
        assert _extract_delta_content(chunk) == ""

    def test_no_content(self) -> None:

        class Delta:
            content = None

        class Choice:
            delta = Delta()

        class Chunk:
            choices = [Choice()]

        assert _extract_delta_content(Chunk()) == ""

    def test_malformed_chunk(self) -> None:
        assert _extract_delta_content({}) == ""
        assert _extract_delta_content("garbage") == ""
