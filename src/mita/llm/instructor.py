"""Instructor wrapper for structured tool call parsing via JSON mode."""

from __future__ import annotations

from typing import Any, TypeVar

import instructor
import litellm
from pydantic import BaseModel

from mita.config.schema import MitaConfig
from mita.tools.schema import ToolCall

T = TypeVar("T", bound=BaseModel)


class ToolCallResponse(BaseModel):
    """Structured response containing tool calls parsed from LLM output."""

    reasoning: str = ""
    tool_calls: list[ToolCall] = []
    text_response: str = ""


class InstructorClient:
    """Instructor-wrapped LiteLLM client for structured output parsing."""

    def __init__(self, config: MitaConfig) -> None:
        self._model = f"ollama/{config.model.default}"
        self._api_base = config.ollama.host
        self._temperature = config.model.temperature
        self._max_tokens = config.model.max_tokens

        self._client = instructor.from_litellm(
            litellm.acompletion,
            mode=instructor.Mode.JSON,
        )

    async def parse_tool_calls(
        self,
        messages: list[dict[str, Any]],
    ) -> ToolCallResponse:
        """Parse structured tool calls from LLM output using Instructor JSON mode."""
        response: ToolCallResponse = await self._client.create(
            model=self._model,
            messages=messages,
            response_model=ToolCallResponse,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            api_base=self._api_base,
        )
        return response

    async def parse_structured(
        self,
        messages: list[dict[str, Any]],
        response_model: type[T],
    ) -> T:
        """Parse any structured Pydantic model from LLM output."""
        response: T = await self._client.create(
            model=self._model,
            messages=messages,
            response_model=response_model,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            api_base=self._api_base,
        )
        return response


def get_instructor_client(config: MitaConfig) -> InstructorClient:
    """Create an Instructor client from configuration."""
    return InstructorClient(config)
