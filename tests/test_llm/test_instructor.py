"""Tests for Instructor wrapper for structured tool call parsing."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from mita.config.schema import MitaConfig
from mita.llm.instructor import (
    InstructorClient,
    ParsedToolCall,
    ToolCallResponse,
    get_instructor_client,
)


class TestParsedToolCall:
    def test_creation(self) -> None:
        tc = ParsedToolCall(id="tc1", name="file_read", arguments={"path": "/tmp/f.py"})
        assert tc.id == "tc1"
        assert tc.name == "file_read"
        assert tc.arguments == {"path": "/tmp/f.py"}

    def test_empty_arguments(self) -> None:
        tc = ParsedToolCall(id="tc2", name="shell", arguments={})
        assert tc.arguments == {}


class TestToolCallResponse:
    def test_defaults(self) -> None:
        resp = ToolCallResponse()
        assert resp.reasoning == ""
        assert resp.tool_calls == []
        assert resp.text_response == ""

    def test_with_tool_calls(self) -> None:
        tc = ParsedToolCall(id="1", name="shell", arguments={"command": "ls"})
        resp = ToolCallResponse(
            reasoning="Need to list files",
            tool_calls=[tc],
            text_response="Let me check.",
        )
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "shell"
        assert resp.reasoning == "Need to list files"


class TestInstructorClient:
    def test_init(self) -> None:
        config = MitaConfig()
        client = InstructorClient(config)
        assert client._model == "ollama/qwen2.5-coder:7b"
        assert client._api_base == "http://localhost:11434"
        assert client._temperature == 0.1
        assert client._max_tokens == 4096

    def test_init_custom_config(self) -> None:
        from mita.config.schema import ModelSettings, OllamaSettings

        config = MitaConfig(
            ollama=OllamaSettings(host="http://custom:9999"),
            model=ModelSettings(default="codellama:7b", temperature=0.5, max_tokens=2048),
        )
        client = InstructorClient(config)
        assert client._model == "ollama/codellama:7b"
        assert client._api_base == "http://custom:9999"
        assert client._temperature == 0.5
        assert client._max_tokens == 2048

    @pytest.mark.asyncio
    async def test_parse_tool_calls(self) -> None:
        config = MitaConfig()
        client = InstructorClient(config)
        expected = ToolCallResponse(
            reasoning="test",
            tool_calls=[ParsedToolCall(id="1", name="shell", arguments={"command": "ls"})],
        )
        client._client = AsyncMock()
        client._client.create = AsyncMock(return_value=expected)
        result = await client.parse_tool_calls([{"role": "user", "content": "list files"}])
        assert result.tool_calls[0].name == "shell"

    @pytest.mark.asyncio
    async def test_parse_structured(self) -> None:
        from pydantic import BaseModel

        class TestModel(BaseModel):
            answer: str = ""

        config = MitaConfig()
        client = InstructorClient(config)
        expected = TestModel(answer="42")
        client._client = AsyncMock()
        client._client.create = AsyncMock(return_value=expected)
        result = await client.parse_structured(
            [{"role": "user", "content": "what is 6*7?"}],
            response_model=TestModel,
        )
        assert result.answer == "42"


class TestGetInstructorClient:
    def test_returns_client(self) -> None:
        config = MitaConfig()
        client = get_instructor_client(config)
        assert isinstance(client, InstructorClient)
