"""Tests for system prompt construction."""

from __future__ import annotations

from mita.agent.system_prompt import build_system_prompt
from mita.config.schema import MitaConfig
from mita.tools.schema import ToolDefinition, ToolParameter


class TestBuildSystemPrompt:
    def test_includes_tools(self) -> None:
        config = MitaConfig()
        tools = [
            ToolDefinition(
                name="file_read",
                description="Read a file.",
                parameters=[ToolParameter(name="path", type="string", description="File path.")],
            ),
        ]
        prompt = build_system_prompt(config, "", tools)
        assert "file_read" in prompt
        assert "Read a file" in prompt
        assert "path: string" in prompt

    def test_includes_memory(self) -> None:
        config = MitaConfig()
        prompt = build_system_prompt(config, "Always use pytest.", [])
        assert "Always use pytest" in prompt
        assert "Project Memory" in prompt

    def test_no_memory_section_when_empty(self) -> None:
        config = MitaConfig()
        prompt = build_system_prompt(config, "", [])
        assert "Project Memory" not in prompt

    def test_marks_destructive_tools(self) -> None:
        config = MitaConfig()
        tools = [
            ToolDefinition(
                name="file_write",
                description="Write a file.",
                parameters=[],
                destructive=True,
            ),
        ]
        prompt = build_system_prompt(config, "", tools)
        assert "DESTRUCTIVE" in prompt

    def test_includes_base_instructions(self) -> None:
        config = MitaConfig()
        prompt = build_system_prompt(config, "", [])
        assert "Mita" in prompt
        assert "coding assistant" in prompt
