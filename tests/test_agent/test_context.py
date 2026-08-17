"""Tests for context assembly."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from mita.agent.context import assemble_context
from mita.agent.conversation import Conversation, Role
from mita.config.schema import MitaConfig
from mita.tools.registry import create_default_registry


class TestAssembleContext:
    def test_adds_system_message(self, tmp_path: Path) -> None:
        conv = Conversation()
        config = MitaConfig()
        registry = create_default_registry()

        with patch("mita.agent.context.load_memory", return_value="test memory"):
            assemble_context(conv, config, registry, cwd=tmp_path)

        assert len(conv.messages) == 1
        assert conv.messages[0].role == Role.SYSTEM
        assert "Mita" in conv.messages[0].content

    def test_includes_memory_in_prompt(self, tmp_path: Path) -> None:
        conv = Conversation()
        config = MitaConfig()
        registry = create_default_registry()

        with patch("mita.agent.context.load_memory", return_value="Use ruff for linting."):
            assemble_context(conv, config, registry, cwd=tmp_path)

        assert "ruff" in conv.messages[0].content

    def test_includes_tool_names(self, tmp_path: Path) -> None:
        conv = Conversation()
        config = MitaConfig()
        registry = create_default_registry()

        with patch("mita.agent.context.load_memory", return_value=""):
            assemble_context(conv, config, registry, cwd=tmp_path)

        system_content = conv.messages[0].content
        assert "file_read" in system_content
        assert "file_write" in system_content
        assert "shell" in system_content


class TestMemorySettingsApplied:
    """Audit finding S6: assemble_context must pass config.memory to load_memory."""

    def test_passes_config_memory_settings(self, tmp_path: Path) -> None:
        conv = Conversation()
        config = MitaConfig()
        config.memory.max_lines_per_file = 7
        registry = create_default_registry()

        with patch("mita.agent.context.load_memory", return_value="") as mock_load:
            assemble_context(conv, config, registry, cwd=tmp_path)

        _, kwargs = mock_load.call_args
        assert kwargs.get("settings") is config.memory
