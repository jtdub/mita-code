"""Core async agent loop: a LangGraph state graph over prompt → model → tools."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import time
import uuid
from collections.abc import Mapping, Sequence
from typing import Any, TypedDict, cast

from langchain_core.messages import AIMessage, AIMessageChunk
from langgraph.graph import END, START, StateGraph
from rich.console import Console

from mita.agent.context import assemble_context
from mita.agent.conversation import Conversation, Message, Role
from mita.config.schema import MitaConfig
from mita.llm.context import resolve_context_window
from mita.llm.factory import build_chat_model
from mita.tools.executor import execute_tool
from mita.tools.registry import ToolRegistry, create_default_registry
from mita.tools.schema import ToolCall
from mita.ui.sink import ConfirmDecision, ConfirmRequest, RichConsoleSink, UISink

_logger = logging.getLogger(__name__)

_RAG_CONTEXT_PREFIX = "Relevant code from the project index:"


class _AgentState(TypedDict):
    """State shared by the graph nodes. The conversation is mutated in place."""

    conversation: Conversation
    config: MitaConfig
    registry: ToolRegistry
    sink: UISink
    session_approved: set[str]
    auto_confirm: bool
    console: Console | None
    context_window: int
    bound_model: Any
    last_tool_sig: str | None
    iteration: int
    stop: bool
    stop_reason: str | None


async def run_agent(
    user_prompt: str,
    config: MitaConfig,
    console: Console,
    conversation: Conversation | None = None,
    registry: ToolRegistry | None = None,
    session_approved: set[str] | None = None,
    auto_confirm: bool = False,
    sink: UISink | None = None,
) -> Conversation:
    """Run the agent loop for a single user prompt.

    Args:
        user_prompt: The user's input.
        config: Application configuration.
        console: Rich console for output (used to build the default sink).
        conversation: Existing conversation to continue, or None to start fresh.
        registry: Tool registry, or None to create default.
        session_approved: Tool names approved for the session (skip confirmation).
        auto_confirm: When True, approve destructive actions without prompting.
        sink: UI event sink; defaults to a RichConsoleSink over ``console``.

    Returns:
        The updated conversation.
    """
    if sink is None:
        sink = RichConsoleSink(console)
    if registry is None:
        registry = create_default_registry()
    if conversation is None:
        conversation = Conversation()
        assemble_context(conversation, config, registry)

    conversation.add(Message(role=Role.USER, content=user_prompt))

    # Fire session_start hooks
    if config.hooks:
        from mita.hooks.runner import run_hooks

        await run_hooks(
            "session_start",
            config.hooks,
            console=console,
            timeout=config.hook_settings.timeout,
        )

    context_window = await resolve_context_window(config)

    model = build_chat_model(config)
    tool_schemas = registry.get_openai_schemas()
    bound_model: Any = model.bind_tools(tool_schemas) if tool_schemas else model

    state: _AgentState = {
        "conversation": conversation,
        "config": config,
        "registry": registry,
        "sink": sink,
        "session_approved": session_approved if session_approved is not None else set(),
        "auto_confirm": auto_confirm,
        "console": console,
        "context_window": context_window,
        "bound_model": bound_model,
        "last_tool_sig": None,
        "iteration": 0,
        "stop": False,
        "stop_reason": None,
    }

    final_state: Any = state
    try:
        compiled = _build_graph().compile()
        final_state = await cast(Any, compiled).ainvoke(state)
    except (KeyboardInterrupt, asyncio.CancelledError):
        sink.error("[Interrupted]")
    except Exception as e:  # noqa: BLE001 - a failing tool must not kill the turn
        _logger.error("Unexpected error in agent loop: %s", e, exc_info=True)
        sink.error(f"Unexpected error: {e}")

    if final_state.get("stop_reason") == "max_iterations":
        sink.error(f"Reached maximum iterations ({config.max_iterations}). Stopping.")

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


def _build_graph() -> StateGraph[_AgentState]:
    """Build the agent graph: retrieve → model → (tools → model)* → end."""
    graph = StateGraph(_AgentState)
    graph.add_node("retrieve", _retrieve_node)
    graph.add_node("model", _model_node)
    graph.add_node("tools", _tools_node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "model")
    graph.add_conditional_edges("model", _route_after_model, {"tools": "tools", "end": END})
    graph.add_conditional_edges("tools", _route_after_tools, {"model": "model", "end": END})
    return graph


async def _retrieve_node(state: _AgentState) -> dict[str, Any]:
    """Inject RAG context for the latest user message, replacing the previous one."""
    config = state["config"]
    conversation = state["conversation"]
    if not config.index.enabled:
        return {}

    try:
        from mita.index.retriever import Retriever

        retriever = Retriever(config)
        if retriever.is_available():
            user_texts = [m.content for m in conversation.messages if m.role == Role.USER]
            if user_texts:
                rag_context = await retriever.retrieve_formatted(user_texts[-1])
                if rag_context:
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
    except Exception:  # noqa: BLE001 - RAG is optional; never let it abort the turn
        # Broad by design: embedding backends raise provider-specific errors
        # (ollama.ResponseError, httpx.HTTPError, ...) that must degrade, not crash.
        _logger.warning("RAG index unavailable, proceeding without it", exc_info=True)

    return {}


async def _model_node(state: _AgentState) -> dict[str, Any]:
    """Call the model, stream output, parse tool calls, and record the assistant turn."""
    config = state["config"]
    conversation = state["conversation"]
    registry = state["registry"]
    sink = state["sink"]
    iteration = state["iteration"] + 1

    conversation.truncate_to_fit(state["context_window"])
    messages = conversation.get_messages_for_api()
    bound_model = state["bound_model"]

    assistant_text = ""
    tool_calls_raw: list[dict[str, Any]] = []
    usage: dict[str, int] | None = None
    elapsed = 0.0
    ttft: float | None = None

    try:
        if config.ui.stream:
            assistant_text, tool_calls_raw, usage, elapsed, ttft = await _stream_model(
                bound_model, messages, sink
            )
        else:
            t0 = time.monotonic()
            with sink.busy("Thinking..."):
                response = await bound_model.ainvoke(messages)
            elapsed = time.monotonic() - t0
            if isinstance(response, AIMessage):
                assistant_text = _content_text(response.content)
                tool_calls_raw = _tool_calls_to_openai(response.tool_calls)
                usage = _usage_metadata(response)
            if assistant_text:
                sink.assistant_message(assistant_text)

        if config.ui.show_token_count and usage:
            sink.response_stats(
                usage.get("prompt_tokens", 0),
                usage.get("completion_tokens", 0),
                elapsed,
                ttft,
            )
    except (KeyboardInterrupt, asyncio.CancelledError):
        sink.error("[Interrupted]")
        if assistant_text:
            conversation.add(Message(role=Role.ASSISTANT, content=assistant_text))
        return _stop(state, conversation, iteration, "interrupted")
    except (ConnectionError, TimeoutError, OSError) as e:
        _logger.warning("LLM call failed: %s", e, exc_info=True)
        sink.error(f"LLM error: {e}")
        if assistant_text:
            conversation.add(Message(role=Role.ASSISTANT, content=assistant_text))
        return _stop(state, conversation, iteration, "error")
    except Exception as e:  # noqa: BLE001
        _logger.error("Unexpected error in agent loop: %s", e, exc_info=True)
        sink.error(f"Unexpected error: {e}")
        if assistant_text:
            conversation.add(Message(role=Role.ASSISTANT, content=assistant_text))
        return _stop(state, conversation, iteration, "error")

    # Fallback: parse tool calls from text if the model didn't use native calling.
    if not tool_calls_raw and assistant_text:
        parsed, remaining = _extract_tool_calls_from_text(assistant_text, registry)
        if parsed:
            tool_calls_raw = parsed
            assistant_text = remaining

    # Detect a repeated identical tool-call batch and stop before it runs again.
    last_sig = state["last_tool_sig"]
    if tool_calls_raw:
        sig = json.dumps(
            [
                (tc.get("function", {}).get("name"), tc.get("function", {}).get("arguments"))
                for tc in tool_calls_raw
            ],
            sort_keys=True,
        )
        if sig == state["last_tool_sig"]:
            sink.error("Detected repeated tool call — stopping to avoid infinite loop.")
            conversation.add(Message(role=Role.ASSISTANT, content=assistant_text))
            return _stop(state, conversation, iteration, "repeat")
        last_sig = sig

    conversation.add(
        Message(
            role=Role.ASSISTANT,
            content=assistant_text,
            tool_calls=tool_calls_raw,
        )
    )

    stop = iteration >= config.max_iterations
    return {
        "conversation": conversation,
        "last_tool_sig": last_sig,
        "iteration": iteration,
        "stop": stop,
        "stop_reason": "max_iterations" if stop else None,
    }


def _stop(
    state: _AgentState, conversation: Conversation, iteration: int, reason: str
) -> dict[str, Any]:
    """Build a state update that halts the graph."""
    return {
        "conversation": conversation,
        "iteration": iteration,
        "stop": True,
        "stop_reason": reason,
    }


async def _tools_node(state: _AgentState) -> dict[str, Any]:
    """Execute the tool calls on the latest assistant message."""
    conversation = state["conversation"]
    last_message = conversation.messages[-1]
    await _process_tool_calls(
        last_message.tool_calls,
        conversation,
        state["registry"],
        state["config"],
        state["sink"],
        session_approved=state["session_approved"],
        auto_confirm=state["auto_confirm"],
        console=state["console"],
    )
    return {"conversation": conversation}


def _route_after_model(state: _AgentState) -> str:
    """Run the tools node when the assistant requested tools, else end."""
    last = state["conversation"].messages[-1]
    if last.role == Role.ASSISTANT and last.tool_calls:
        return "tools"
    return "end"


def _route_after_tools(state: _AgentState) -> str:
    """Return to the model unless a stop condition was set."""
    return "end" if state["stop"] else "model"


async def _stream_model(
    model: Any,
    messages: list[dict[str, Any]],
    sink: UISink,
) -> tuple[str, list[dict[str, Any]], dict[str, int] | None, float, float | None]:
    """Stream the model response, displaying tokens as they arrive.

    Tool calls are read from the merged ``AIMessageChunk`` so that a batch of
    calls arriving in one chunk (with ``index`` unset) stays separate — keying
    on the chunk index collapses them into one (finding: streamed tool-call
    merge).

    Returns:
        Tuple of (text_content, tool_calls_raw, usage, elapsed, ttft).
    """
    text_parts: list[str] = []
    usage: dict[str, int] | None = None
    first_token = True
    start_time = time.monotonic()
    ttft: float | None = None
    accumulated: AIMessageChunk | None = None

    spinner_stack = contextlib.ExitStack()
    spinner_stack.enter_context(sink.busy("Thinking..."))

    try:
        async for chunk in model.astream(messages):
            if first_token:
                spinner_stack.close()
                ttft = time.monotonic() - start_time
                first_token = False

            accumulated = chunk if accumulated is None else accumulated + chunk

            text = _content_text(chunk.content)
            if text:
                text_parts.append(text)
                sink.stream_token(text)

            usage = _usage_metadata(chunk) or usage
    finally:
        spinner_stack.close()

    elapsed = time.monotonic() - start_time
    full_text = "".join(text_parts)
    if full_text:
        sink.stream_end()

    tool_calls_raw = _tool_calls_to_openai(accumulated.tool_calls) if accumulated else []
    return full_text, tool_calls_raw, usage, elapsed, ttft


def _tool_calls_to_openai(tool_calls: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Convert LangChain tool calls to the OpenAI function-call shape."""
    result: list[dict[str, Any]] = []
    for call in tool_calls:
        result.append(
            {
                "id": call.get("id") or str(uuid.uuid4()),
                "type": "function",
                "function": {
                    "name": call.get("name", ""),
                    # An empty argument set must become "{}", not "", or the next
                    # request fails to parse it (finding: empty tool arguments).
                    "arguments": json.dumps(call.get("args") or {}),
                },
            }
        )
    return result


def _usage_metadata(message: Any) -> dict[str, int] | None:
    """Extract token usage from a LangChain message, if reported."""
    meta = getattr(message, "usage_metadata", None)
    if not meta:
        return None
    usage: dict[str, int] = {}
    if meta.get("input_tokens") is not None:
        usage["prompt_tokens"] = meta["input_tokens"]
    if meta.get("output_tokens") is not None:
        usage["completion_tokens"] = meta["output_tokens"]
    return usage or None


def _content_text(content: Any) -> str:
    """Extract plain text from message content (str, None, or a content block list)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and "text" in block
        ]
        return "\n".join(parts)
    return ""


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
    sink: UISink,
    session_approved: set[str] | None = None,
    auto_confirm: bool = False,
    console: Console | None = None,
) -> None:
    """Process tool calls from an LLM response.

    Every tool call in ``tool_calls_raw`` is guaranteed a matching TOOL result
    message, even if the batch is interrupted mid-way. Without that, the assistant
    message carries tool_calls with no answers and the next request 400s on
    OpenAI-compatible endpoints.
    """
    # Import hooks runner once if hooks are configured
    _run_hooks = None
    if config.hooks:
        from mita.hooks.runner import run_hooks

        _run_hooks = run_hooks

    answered: set[str] = set()
    try:
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
            sink.tool_call(tool_call)

            # Fire pre_tool_call hooks
            if _run_hooks is not None:
                await _run_hooks(
                    "pre_tool_call",
                    config.hooks,
                    context={"tool": name, "args": arguments},
                    console=console,
                    timeout=config.hook_settings.timeout,
                )

            # Confirmation: auto-approve when requested, else ask the sink with an
            # allow-for-session option that adds the tool to session_approved.
            async def confirm_fn(prompt: str, _name: str = name) -> bool:
                if auto_confirm:
                    return True
                decision = await sink.confirm(ConfirmRequest(tool_name=_name, summary=prompt))
                if decision == ConfirmDecision.ALLOW_SESSION:
                    if session_approved is not None:
                        session_approved.add(_name)
                    return True
                return decision == ConfirmDecision.ALLOW_ONCE

            # Execute with safety checks
            result = await execute_tool(
                tool_call,
                registry,
                config.tools,
                confirm_fn=confirm_fn,
                session_approved=session_approved,
            )

            # Display result
            sink.tool_result(result)

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
            answered.add(tc_id)
    finally:
        # Backfill results for any tool calls left unanswered (e.g. interrupt).
        for tc_raw in tool_calls_raw:
            tc_id = tc_raw.get("id")
            if not tc_id or tc_id in answered:
                continue
            func = tc_raw.get("function", {})
            name = func.get("name", "") if isinstance(func, dict) else ""
            conversation.add(
                Message(
                    role=Role.TOOL,
                    content="Not executed (interrupted).",
                    tool_call_id=tc_id,
                    name=name,
                )
            )
