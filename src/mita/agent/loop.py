"""Core async agent loop: prompt → LLM → parse tool calls → execute → loop."""

from __future__ import annotations

import time
import uuid
from typing import Any

from rich.console import Console

from mita.agent.context import assemble_context
from mita.agent.conversation import Conversation, Message, Role
from mita.config.schema import MitaConfig
from mita.llm.client import LLMClient
from mita.tools.executor import execute_tool
from mita.tools.registry import ToolRegistry, create_default_registry
from mita.tools.schema import ToolCall
from mita.ui.display import (
    display_error,
    display_markdown,
    display_response_stats,
    display_streaming_end,
    display_streaming_token,
    display_tool_call,
    display_tool_result,
    prompt_user_confirm,
)
from mita.ui.spinner import thinking_spinner

MAX_ITERATIONS = 25
_RAG_CONTEXT_PREFIX = "Relevant code from the project index:"


async def run_agent(
    user_prompt: str,
    config: MitaConfig,
    console: Console,
    conversation: Conversation | None = None,
    registry: ToolRegistry | None = None,
    llm_client: LLMClient | None = None,
) -> Conversation:
    """Run the agent loop for a single user prompt.

    Args:
        user_prompt: The user's input.
        config: Application configuration.
        console: Rich console for output.
        conversation: Existing conversation to continue, or None to start fresh.
        registry: Tool registry, or None to create default.
        llm_client: LLM client, or None to create from config.

    Returns:
        The updated conversation.
    """
    # Initialize
    if registry is None:
        registry = create_default_registry()

    # Load MCP plugin tools (if configured and not already loaded)
    if config.plugins and not any(d.source.startswith("mcp:") for d in registry.get_definitions()):
        from mita.plugins.manager import PluginManager

        plugin_mgr = PluginManager(config.plugins)
        await plugin_mgr.start_all(console=console)
        await plugin_mgr.register_tools_async(registry)

    if llm_client is None:
        llm_client = LLMClient(config)
    if conversation is None:
        conversation = Conversation()
        assemble_context(conversation, config, registry)

    # Add user message
    conversation.add(Message(role=Role.USER, content=user_prompt))

    # Inject RAG context if index is available (replace previous RAG message)
    if config.index.enabled:
        try:
            from mita.index.retriever import Retriever

            retriever = Retriever(config)
            if retriever.is_available():
                rag_context = await retriever.retrieve_formatted(user_prompt)
                if rag_context:
                    # Remove any previous RAG context message
                    conversation.messages = [
                        m
                        for m in conversation.messages
                        if not (m.role == Role.SYSTEM and m.content.startswith(_RAG_CONTEXT_PREFIX))
                    ]
                    conversation.add(
                        Message(
                            role=Role.SYSTEM,
                            content=f"{_RAG_CONTEXT_PREFIX}\n{rag_context}",
                        )
                    )
        except (ConnectionError, FileNotFoundError, ImportError, OSError):
            pass  # Index unavailable; proceed without RAG

    # Agent loop
    for iteration in range(MAX_ITERATIONS):
        # Truncate to fit context window
        conversation.truncate_to_fit(config.model.context_window)

        # Call LLM
        # Tools are described in the system prompt — don't also pass JSON schemas
        # via the `tools` parameter, as native function calling is much slower
        # on local models. (See CLAUDE.md: "Instructor JSON mode is the primary
        # tool-call path.")
        messages = conversation.get_messages_for_api()

        if config.ui.stream:
            assistant_text, tool_calls_raw, stats = await _stream_response(
                llm_client, messages, [], console
            )
            if config.ui.show_token_count:
                display_response_stats(
                    console,
                    prompt_tokens=stats.prompt_tokens,
                    completion_tokens=stats.completion_tokens,
                    total_time=stats.total_time,
                    ttft=stats.ttft,
                )
        else:
            t0 = time.monotonic()
            with thinking_spinner(console):
                response = await llm_client.chat(messages, tools=[])
            elapsed = time.monotonic() - t0
            assistant_text, tool_calls_raw = _parse_response(response)
            if assistant_text:
                display_markdown(console, assistant_text)
            if config.ui.show_token_count:
                usage = _extract_usage(response) or {}
                display_response_stats(
                    console,
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    total_time=elapsed,
                )

        # Handle tool calls
        if tool_calls_raw:
            conversation.add(
                Message(
                    role=Role.ASSISTANT,
                    content=assistant_text,
                    tool_calls=tool_calls_raw,
                )
            )
            await _process_tool_calls(tool_calls_raw, conversation, registry, config, console)
            continue

        # No tool calls — add assistant message and stop
        conversation.add(Message(role=Role.ASSISTANT, content=assistant_text))
        break
    else:
        display_error(
            console,
            f"Reached maximum iterations ({MAX_ITERATIONS}). Stopping.",
        )

    return conversation


class _ResponseStats:
    """Token usage and timing stats from an LLM response."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_time: float = 0.0
    ttft: float | None = None


async def _stream_response(
    client: LLMClient,
    messages: list[dict[str, Any]],
    tool_schemas: list[dict[str, Any]],
    console: Console,
) -> tuple[str, list[dict[str, Any]], _ResponseStats]:
    """Stream the LLM response, displaying tokens as they arrive.

    Returns:
        Tuple of (text_content, tool_calls_raw, stats).
    """
    full_text = ""
    tool_calls_by_index: dict[int, dict[str, Any]] = {}
    first_token = True
    stats = _ResponseStats()
    start_time = time.monotonic()

    # Show spinner while waiting for first token
    spinner_ctx = thinking_spinner(console)
    spinner_ctx.__enter__()

    async for chunk in client.stream_chat(messages, tools=tool_schemas):
        if first_token:
            spinner_ctx.__exit__(None, None, None)
            stats.ttft = time.monotonic() - start_time
            first_token = False

        delta = _extract_delta(chunk)
        if delta:
            full_text += delta
            display_streaming_token(console, delta)

        # Accumulate tool call deltas
        _accumulate_tool_call_deltas(chunk, tool_calls_by_index)

        # Extract usage from final chunk (LiteLLM includes it on the last chunk)
        usage = _extract_usage(chunk)
        if usage:
            stats.prompt_tokens = usage.get("prompt_tokens", 0)
            stats.completion_tokens = usage.get("completion_tokens", 0)

    # Clean up spinner if no chunks arrived at all
    if first_token:
        spinner_ctx.__exit__(None, None, None)

    stats.total_time = time.monotonic() - start_time

    if full_text:
        display_streaming_end(console)

    # Convert accumulated tool calls to list
    tool_calls_raw = [tool_calls_by_index[i] for i in sorted(tool_calls_by_index)]
    return full_text, tool_calls_raw, stats


def _extract_usage(chunk: Any) -> dict[str, int] | None:
    """Extract token usage from a streaming chunk (typically the last one)."""
    try:
        usage = getattr(chunk, "usage", None) or (
            chunk.get("usage") if isinstance(chunk, dict) else None
        )
        if usage is None:
            return None
        if hasattr(usage, "prompt_tokens"):
            return {
                "prompt_tokens": usage.prompt_tokens or 0,
                "completion_tokens": usage.completion_tokens or 0,
            }
        if isinstance(usage, dict) and "prompt_tokens" in usage:
            return usage
    except (AttributeError, KeyError):
        pass
    return None


def _extract_delta(chunk: Any) -> str:
    """Extract text content from a streaming chunk."""
    try:
        choices = chunk.choices if hasattr(chunk, "choices") else chunk.get("choices", [])
        if not choices:
            return ""
        delta = choices[0].delta if hasattr(choices[0], "delta") else choices[0].get("delta", {})
        if hasattr(delta, "content"):
            return delta.content or ""
        if isinstance(delta, dict):
            return delta.get("content", "") or ""
    except (IndexError, AttributeError, KeyError):
        pass
    return ""


def _accumulate_tool_call_deltas(
    chunk: Any, tool_calls_by_index: dict[int, dict[str, Any]]
) -> None:
    """Accumulate tool call deltas from a streaming chunk.

    Streaming tool calls arrive as incremental deltas indexed by position.
    This function merges them into complete tool call dicts.
    """
    try:
        choices = chunk.choices if hasattr(chunk, "choices") else chunk.get("choices", [])
        if not choices:
            return
        delta = choices[0].delta if hasattr(choices[0], "delta") else choices[0].get("delta", {})

        raw_tcs = (
            delta.tool_calls
            if hasattr(delta, "tool_calls")
            else delta.get("tool_calls")
            if isinstance(delta, dict)
            else None
        )
        if not raw_tcs:
            return

        for tc in raw_tcs:
            idx = getattr(tc, "index", None) if hasattr(tc, "index") else tc.get("index", 0)
            if idx is None:
                idx = 0

            if idx not in tool_calls_by_index:
                tc_id = (getattr(tc, "id", "") if hasattr(tc, "id") else tc.get("id", "")) or str(
                    uuid.uuid4()
                )
                tool_calls_by_index[idx] = {
                    "id": tc_id,
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                }

            entry = tool_calls_by_index[idx]
            func = getattr(tc, "function", None) if hasattr(tc, "function") else tc.get("function")

            if func is not None:
                fname = getattr(func, "name", None) if hasattr(func, "name") else func.get("name")
                if fname:
                    entry["function"]["name"] += fname

                fargs = (
                    getattr(func, "arguments", None)
                    if hasattr(func, "arguments")
                    else func.get("arguments")
                )
                if fargs:
                    entry["function"]["arguments"] += fargs
    except (IndexError, AttributeError, KeyError):
        pass


def _parse_response(response: Any) -> tuple[str, list[dict[str, Any]]]:
    """Parse a non-streaming LLM response into text and tool calls."""
    try:
        choices = response.choices if hasattr(response, "choices") else response.get("choices", [])
        if not choices:
            return "", []

        message = (
            choices[0].message if hasattr(choices[0], "message") else choices[0].get("message", {})
        )

        content = ""
        if hasattr(message, "content"):
            content = message.content or ""
        elif isinstance(message, dict):
            content = message.get("content", "") or ""

        tool_calls: list[dict[str, Any]] = []
        raw_calls = (
            message.tool_calls
            if hasattr(message, "tool_calls")
            else message.get("tool_calls")
            if isinstance(message, dict)
            else None
        )
        if raw_calls:
            for tc in raw_calls:
                if hasattr(tc, "function"):
                    tool_calls.append(
                        {
                            "id": getattr(tc, "id", str(uuid.uuid4())),
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                    )
                elif isinstance(tc, dict):
                    tool_calls.append(tc)

        return content, tool_calls
    except (IndexError, AttributeError, KeyError):
        return "", []


async def _process_tool_calls(
    tool_calls_raw: list[dict[str, Any]],
    conversation: Conversation,
    registry: ToolRegistry,
    config: MitaConfig,
    console: Console,
) -> None:
    """Process tool calls from an LLM response."""
    import json

    for tc_raw in tool_calls_raw:
        func = tc_raw.get("function", {})
        tc_id = tc_raw.get("id", str(uuid.uuid4()))
        name = func.get("name", "") if isinstance(func, dict) else ""
        args_raw = func.get("arguments", "{}") if isinstance(func, dict) else "{}"

        # Parse arguments
        if isinstance(args_raw, str):
            try:
                arguments = json.loads(args_raw)
            except json.JSONDecodeError:
                arguments = {}
        else:
            arguments = args_raw if isinstance(args_raw, dict) else {}

        tool_call = ToolCall(id=tc_id, name=name, arguments=arguments)

        # Display the tool call
        display_tool_call(console, tool_call)

        # Create confirm function bound to console
        async def confirm_fn(prompt: str) -> bool:
            return await prompt_user_confirm(console, prompt)

        # Execute with safety checks
        result = await execute_tool(tool_call, registry, config.tools, confirm_fn=confirm_fn)

        # Display result
        display_tool_result(console, result)

        # Add tool result to conversation
        conversation.add(
            Message(
                role=Role.TOOL,
                content=result.output if result.success else (result.error or "Error"),
                tool_call_id=tc_id,
                name=name,
            )
        )
