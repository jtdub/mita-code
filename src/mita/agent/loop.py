"""Core async agent loop: a LangGraph state graph over prompt → model → tools."""

from __future__ import annotations

import asyncio
import contextlib
import functools
import json
import logging
import re
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypedDict, cast

from langchain_core.messages import AIMessage
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
        "stop_reason": None,
    }

    final_state: Any = state
    try:
        final_state = await cast(Any, _compiled_graph()).ainvoke(state)
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


@functools.cache
def _compiled_graph() -> Any:
    """Return the compiled agent graph.

    The graph is static: every per-turn value travels in the state, so one
    compiled graph serves every turn.
    """
    return _build_graph().compile()


async def _retrieve_node(state: _AgentState) -> dict[str, Any]:
    """Inject RAG context for the latest user message, replacing the previous one.

    Retrieval is optional. Embedding backends raise provider-specific errors, so
    every failure degrades to a turn without context.
    """
    config = state["config"]
    conversation = state["conversation"]
    if not config.index.enabled:
        return {}

    try:
        from mita.index.retriever import Retriever

        retriever = Retriever(config)
        latest_user_text = _latest_user_text(conversation)
        if retriever.is_available() and latest_user_text is not None:
            rag_context = await retriever.retrieve_formatted(latest_user_text)
            if rag_context:
                _replace_rag_context(conversation, rag_context)
    except Exception:  # noqa: BLE001
        _logger.warning("RAG index unavailable, proceeding without it", exc_info=True)

    return {}


def _latest_user_text(conversation: Conversation) -> str | None:
    """Return the content of the most recent user message, or None."""
    return next(
        (m.content for m in reversed(conversation.messages) if m.role == Role.USER),
        None,
    )


def _replace_rag_context(conversation: Conversation, rag_context: str) -> None:
    """Drop the previous retrieved-code system message and append the new one."""
    conversation.messages = [
        m
        for m in conversation.messages
        if not (m.role == Role.SYSTEM and m.content.startswith(_RAG_CONTEXT_PREFIX))
    ]
    conversation.add(Message(role=Role.SYSTEM, content=f"{_RAG_CONTEXT_PREFIX}\n{rag_context}"))


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
                assistant_text = str(response.text)
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
        return _stop(iteration, "interrupted")
    except Exception as e:  # noqa: BLE001
        _report_model_error(e, sink)
        if assistant_text:
            conversation.add(Message(role=Role.ASSISTANT, content=assistant_text))
        return _stop(iteration, "error")

    if not tool_calls_raw and assistant_text:
        parsed, remaining = _extract_tool_calls_from_text(assistant_text, registry)
        if parsed:
            tool_calls_raw = parsed
            assistant_text = remaining

    signature = _tool_call_signature(tool_calls_raw)
    if signature is not None and signature == state["last_tool_sig"]:
        sink.error("Detected repeated tool call — stopping to avoid infinite loop.")
        conversation.add(Message(role=Role.ASSISTANT, content=assistant_text))
        return _stop(iteration, "repeat")

    conversation.add(
        Message(
            role=Role.ASSISTANT,
            content=assistant_text,
            tool_calls=tool_calls_raw,
        )
    )

    # Only report exhaustion when the turn still wanted to continue. A plain
    # answer on the last allowed iteration is a normal end, not a stop.
    at_limit = iteration >= config.max_iterations
    stop_reason = "max_iterations" if at_limit and tool_calls_raw else None
    return {
        "last_tool_sig": signature or state["last_tool_sig"],
        "iteration": iteration,
        "stop_reason": stop_reason,
    }


def _report_model_error(error: Exception, sink: UISink) -> None:
    """Log a failed model call and show it, naming transport faults separately."""
    if isinstance(error, ConnectionError | TimeoutError | OSError):
        _logger.warning("LLM call failed: %s", error, exc_info=True)
        sink.error(f"LLM error: {error}")
    else:
        _logger.error("Unexpected error in agent loop: %s", error, exc_info=True)
        sink.error(f"Unexpected error: {error}")


def _tool_call_signature(tool_calls_raw: list[dict[str, Any]]) -> str | None:
    """Return a stable key for a tool-call batch, or None when the batch is empty.

    Two turns that produce the same key mean the model is repeating itself.
    """
    if not tool_calls_raw:
        return None
    return json.dumps(
        [
            (tc.get("function", {}).get("name"), tc.get("function", {}).get("arguments"))
            for tc in tool_calls_raw
        ],
        sort_keys=True,
    )


def _stop(iteration: int, reason: str) -> dict[str, Any]:
    """Build a state update that halts the graph."""
    return {"iteration": iteration, "stop_reason": reason}


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
    return {}


def _route_after_model(state: _AgentState) -> str:
    """Run the tools node when the assistant requested tools, else end."""
    last = state["conversation"].messages[-1]
    if last.role == Role.ASSISTANT and last.tool_calls:
        return "tools"
    return "end"


def _route_after_tools(state: _AgentState) -> str:
    """Return to the model unless a stop condition was set."""
    return "end" if state["stop_reason"] else "model"


async def _stream_model(
    model: Any,
    messages: list[dict[str, Any]],
    sink: UISink,
) -> tuple[str, list[dict[str, Any]], dict[str, int] | None, float, float | None]:
    """Stream the model response, displaying tokens as they arrive.

    Returns:
        Tuple of (text_content, tool_calls_raw, usage, elapsed, ttft).
    """
    text_parts: list[str] = []
    usage: dict[str, int] | None = None
    first_token = True
    start_time = time.monotonic()
    ttft: float | None = None
    pending_calls: dict[Any, _PendingToolCall] = {}

    spinner_stack = contextlib.ExitStack()
    spinner_stack.enter_context(sink.busy("Thinking..."))

    try:
        async for chunk in model.astream(messages):
            if first_token:
                spinner_stack.close()
                ttft = time.monotonic() - start_time
                first_token = False

            _merge_tool_call_chunks(pending_calls, chunk)

            text = str(chunk.text)
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

    tool_calls_raw = [call.to_openai() for call in pending_calls.values()]
    return full_text, tool_calls_raw, usage, elapsed, ttft


@dataclass
class _PendingToolCall:
    """One tool call under construction from a stream of fragments."""

    call_id: str | None = None
    name_parts: list[str] = field(default_factory=list)
    argument_parts: list[str] = field(default_factory=list)

    def to_openai(self) -> dict[str, Any]:
        """Return the finished call in the OpenAI function-call shape."""
        return _openai_tool_call(
            "".join(self.name_parts), "".join(self.argument_parts), self.call_id
        )


def _merge_tool_call_chunks(pending: dict[Any, _PendingToolCall], chunk: Any) -> None:
    """Add one stream chunk's tool-call fragments to the calls under construction.

    A backend that sends a whole call in one chunk leaves ``index`` unset, so the
    fragment id keys the call and a batch stays separate. A backend that splits a
    call across chunks numbers them, so ``index`` keys the call.
    """
    for fragment in getattr(chunk, "tool_call_chunks", None) or []:
        index = fragment.get("index")
        key = index if index is not None else fragment.get("id") or 0
        call = pending.setdefault(key, _PendingToolCall())
        if fragment.get("id"):
            call.call_id = fragment["id"]
        if fragment.get("name"):
            call.name_parts.append(fragment["name"])
        if fragment.get("args"):
            call.argument_parts.append(fragment["args"])


def _openai_tool_call(name: str, arguments: str, call_id: str | None = None) -> dict[str, Any]:
    """Build one tool call in the OpenAI function-call shape.

    An empty argument set becomes "{}". An empty string fails to parse on the
    next request.
    """
    return {
        "id": call_id or str(uuid.uuid4()),
        "type": "function",
        "function": {"name": name, "arguments": arguments or "{}"},
    }


def _tool_calls_to_openai(tool_calls: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Convert LangChain tool calls to the OpenAI function-call shape."""
    return [
        _openai_tool_call(call.get("name", ""), json.dumps(call.get("args") or {}), call.get("id"))
        for call in tool_calls
    ]


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
            tool_calls.append(_openai_tool_call(name, json.dumps(arguments)))
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
