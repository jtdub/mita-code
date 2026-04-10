"""Core async agent loop: prompt → LLM → parse tool calls → execute → loop."""

from __future__ import annotations

import contextlib
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console

from mita.agent.context import assemble_context
from mita.agent.conversation import Conversation, Message, Role
from mita.config.schema import MitaConfig
from mita.llm.client import LLMClient
from mita.llm.streaming import extract_delta_content
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

_logger = logging.getLogger(__name__)

_RAG_CONTEXT_PREFIX = "Relevant code from the project index:"


async def run_agent(
    user_prompt: str,
    config: MitaConfig,
    console: Console,
    conversation: Conversation | None = None,
    registry: ToolRegistry | None = None,
    llm_client: LLMClient | None = None,
    session_approved: set[str] | None = None,
) -> Conversation:
    """Run the agent loop for a single user prompt.

    Args:
        user_prompt: The user's input.
        config: Application configuration.
        console: Rich console for output.
        conversation: Existing conversation to continue, or None to start fresh.
        registry: Tool registry, or None to create default.
        llm_client: LLM client, or None to create from config.
        session_approved: Tool names approved for the session (skip confirmation).
            Not mutated by the agent loop — callers manage the set.

    Returns:
        The updated conversation.
    """
    # Initialize
    if registry is None:
        registry = create_default_registry()

    if llm_client is None:
        llm_client = LLMClient(config)
    if conversation is None:
        conversation = Conversation()
        assemble_context(conversation, config, registry)

    # Add user message
    conversation.add(Message(role=Role.USER, content=user_prompt))

    # Inject RAG context if index is available (replace previous RAG message)
    retriever = None
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
            _logger.warning("RAG index unavailable, proceeding without it", exc_info=True)

    # Fire session_start hooks
    if config.hooks:
        from mita.hooks.runner import run_hooks

        await run_hooks(
            "session_start",
            config.hooks,
            console=console,
            timeout=config.hook_settings.timeout,
        )

    # Agent loop
    last_tool_signature: str | None = None
    repeat_count = 0
    max_iterations = config.max_iterations
    assistant_text = ""
    tool_schemas = registry.get_openai_schemas()
    for _iteration in range(max_iterations):
        try:
            # Truncate to fit context window
            conversation.truncate_to_fit(config.model.context_window)

            # Call LLM with tool schemas so the model can produce structured tool calls
            messages = conversation.get_messages_for_api()

            if config.ui.stream:
                assistant_text, tool_calls_raw, stats = await _stream_response(
                    llm_client, messages, tool_schemas, console
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
                    response = await llm_client.chat(messages, tools=tool_schemas)
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

            # Fallback: parse tool calls from text if model didn't use native calling
            if not tool_calls_raw and assistant_text:
                parsed, remaining_text = _extract_tool_calls_from_text(assistant_text, registry)
                if parsed:
                    tool_calls_raw = parsed
                    assistant_text = remaining_text

            # Handle tool calls
            if tool_calls_raw:
                # Detect repeated identical tool calls (model stuck in a loop)
                sig = json.dumps(
                    [
                        (
                            tc.get("function", {}).get("name"),
                            tc.get("function", {}).get("arguments"),
                        )
                        for tc in tool_calls_raw
                    ],
                    sort_keys=True,
                )
                if sig == last_tool_signature:
                    repeat_count += 1
                    if repeat_count >= 1:
                        display_error(
                            console,
                            "Detected repeated tool call — stopping to avoid infinite loop.",
                        )
                        conversation.add(Message(role=Role.ASSISTANT, content=assistant_text))
                        break
                else:
                    repeat_count = 0
                last_tool_signature = sig

                conversation.add(
                    Message(
                        role=Role.ASSISTANT,
                        content=assistant_text,
                        tool_calls=tool_calls_raw,
                    )
                )
                await _process_tool_calls(
                    tool_calls_raw, conversation, registry, config, console, session_approved
                )
                continue

            # No tool calls — add assistant message and stop
            conversation.add(Message(role=Role.ASSISTANT, content=assistant_text))
            break

        except KeyboardInterrupt:
            display_error(console, "[Interrupted]")
            # Add any partial response as assistant message
            if assistant_text:
                conversation.add(Message(role=Role.ASSISTANT, content=assistant_text))
            break
        except (ConnectionError, TimeoutError, OSError) as e:
            _logger.warning("LLM call failed: %s", e, exc_info=True)
            display_error(console, f"LLM error: {e}")
            if assistant_text:
                conversation.add(Message(role=Role.ASSISTANT, content=assistant_text))
            break
        except json.JSONDecodeError as e:
            _logger.warning("Failed to parse LLM response: %s", e, exc_info=True)
            display_error(console, f"Response parse error: {e}")
            break
        except Exception as e:  # noqa: BLE001
            _logger.error("Unexpected error in agent loop: %s", e, exc_info=True)
            display_error(console, f"Unexpected error: {e}")
            if assistant_text:
                conversation.add(Message(role=Role.ASSISTANT, content=assistant_text))
            break
    else:
        display_error(
            console,
            f"Reached maximum iterations ({max_iterations}). Stopping.",
        )

    # Fire session_end hooks
    if config.hooks:
        from mita.hooks.runner import run_hooks

        await run_hooks(
            "session_end",
            config.hooks,
            console=console,
            timeout=config.hook_settings.timeout,
        )

    return conversation


@dataclass
class _ResponseStats:
    """Token usage and timing stats from an LLM response."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_time: float = 0.0
    ttft: float | None = field(default=None)


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
    text_parts: list[str] = []
    tool_calls_by_index: dict[int, dict[str, Any]] = {}
    first_token = True
    stats = _ResponseStats()
    start_time = time.monotonic()

    spinner_stack = contextlib.ExitStack()
    spinner_stack.enter_context(thinking_spinner(console))

    try:
        async for chunk in client.stream_chat(messages, tools=tool_schemas):
            if first_token:
                spinner_stack.close()
                stats.ttft = time.monotonic() - start_time
                first_token = False

            delta = extract_delta_content(chunk)
            if delta:
                text_parts.append(delta)
                display_streaming_token(console, delta)

            _accumulate_tool_call_deltas(chunk, tool_calls_by_index)

            usage = _extract_usage(chunk)
            if usage:
                stats.prompt_tokens = usage.get("prompt_tokens", 0)
                stats.completion_tokens = usage.get("completion_tokens", 0)
    finally:
        spinner_stack.close()

    stats.total_time = time.monotonic() - start_time
    full_text = "".join(text_parts)

    if full_text:
        display_streaming_end(console)

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


def _extract_tool_calls_from_text(
    text: str, registry: ToolRegistry
) -> tuple[list[dict[str, Any]], str]:
    """Extract tool calls from text when the model outputs JSON instead of native calls.

    Some local models output tool call JSON in text content rather than using
    the structured tool_calls field. This parses those and returns them as
    proper tool call dicts along with any remaining non-tool text.

    Returns:
        Tuple of (tool_calls_raw, remaining_text).
    """
    tool_calls: list[dict[str, Any]] = []
    remaining_parts: list[str] = []

    # First strip markdown fences, then find bare JSON objects
    # Replace fenced JSON blocks with their contents for uniform parsing
    fenced = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
    normalized = fenced.sub(r"\1", text)

    # Find JSON objects by scanning for top-level braces
    json_spans: list[tuple[int, int, dict[str, Any]]] = []
    i = 0
    while i < len(normalized):
        if normalized[i] == "{":
            obj, end = _try_parse_json_object(normalized, i)
            if obj is not None:
                json_spans.append((i, end, obj))
                i = end
                continue
        i += 1

    last_end = 0
    for start, end, obj in json_spans:
        before = normalized[last_end:start].strip()
        if before:
            remaining_parts.append(before)
        last_end = end

        name = obj.get("name", "")
        arguments = obj.get("arguments", obj.get("params", {}))
        if name and registry.has_tool(name) and isinstance(arguments, dict):
            tool_calls.append(
                {
                    "id": str(uuid.uuid4()),
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments),
                    },
                }
            )
        else:
            remaining_parts.append(normalized[start:end])

    trailing = normalized[last_end:].strip()
    if trailing:
        remaining_parts.append(trailing)

    remaining_text = "\n".join(remaining_parts).strip()
    return tool_calls, remaining_text


def _try_parse_json_object(text: str, start: int) -> tuple[dict[str, Any] | None, int]:
    """Try to parse a JSON object starting at position start in text.

    Returns (parsed_dict, end_position) or (None, start) if parsing fails.
    """
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    obj = json.loads(candidate)
                    if isinstance(obj, dict):
                        return obj, i + 1
                except json.JSONDecodeError:
                    return None, start
    return None, start


async def _process_tool_calls(
    tool_calls_raw: list[dict[str, Any]],
    conversation: Conversation,
    registry: ToolRegistry,
    config: MitaConfig,
    console: Console,
    session_approved: set[str] | None = None,
) -> None:
    """Process tool calls from an LLM response."""
    # Import hooks runner once if hooks are configured
    _run_hooks = None
    if config.hooks:
        from mita.hooks.runner import run_hooks

        _run_hooks = run_hooks

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

        # Fire pre_tool_call hooks
        if _run_hooks is not None:
            await _run_hooks(
                "pre_tool_call",
                config.hooks,
                context={"tool": name, "args": arguments},
                console=console,
                timeout=config.hook_settings.timeout,
            )

        # Create confirm function bound to console
        async def confirm_fn(prompt: str) -> bool:
            return await prompt_user_confirm(console, prompt)

        # Execute with safety checks
        result = await execute_tool(
            tool_call,
            registry,
            config.tools,
            confirm_fn=confirm_fn,
            session_approved=session_approved,
        )

        # Display result
        display_tool_result(console, result)

        # Fire post_tool_call hooks
        if _run_hooks is not None:
            await _run_hooks(
                "post_tool_call",
                config.hooks,
                context={"tool": name, "result": str(result.output or result.error)},
                console=console,
                timeout=config.hook_settings.timeout,
            )

        # Fire on_file_write hooks for file_write/file_edit tools
        if _run_hooks is not None and result.success and name in ("file_write", "file_edit"):
            file_path = arguments.get("path", "")
            if file_path:
                await _run_hooks(
                    "on_file_write",
                    config.hooks,
                    context={"file_path": file_path},
                    console=console,
                    timeout=config.hook_settings.timeout,
                )

        # Add tool result to conversation
        conversation.add(
            Message(
                role=Role.TOOL,
                content=result.output if result.success else (result.error or "Error"),
                tool_call_id=tc_id,
                name=name,
            )
        )
