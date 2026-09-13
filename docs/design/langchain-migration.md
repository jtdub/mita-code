# Design: Replace LiteLLM + MCP SDK with the LangChain suite

Status: **implemented (2026-09)**. All six phases landed. `litellm`, `instructor`, and the
direct `mcp` dependency are gone; `mcp` remains transitive via `langchain-mcp-adapters`.

> **Implementation notes (deviations from the proposal):**
>
> - `run_agent()` dropped the `llm_client` parameter (was only used by tests; `cli.py`,
>   the TUI, and `sessions` are unchanged).
> - `MultiServerMCPClient` is constructed with `handle_tool_errors=False`, so an MCP
>   `isError` result raises and becomes a failed `ToolResult` — identical to the old
>   behavior.
> - `ChatOpenAI` in the pinned `langchain-openai` takes `max_completion_tokens` (not
>   `max_tokens`), `api_key` must be a `SecretStr`, and `stream_usage=True` is set so
>   token counts stream on local OpenAI-compatible backends.
> - `OpenAIEmbeddings` is built with `check_embedding_ctx_length=False` (tiktoken checks
>   are OpenAI-specific and break local servers).
> - `ProviderSpec.prefix` (the LiteLLM model string) was removed; `is_ollama` is now a
>   plain spec field and `ResolvedBackend.model` is gone.
> - The stdio env allowlist (C1) is retained by passing
>   `mcp.client.stdio.get_default_environment()` overlaid with the plugin's `env`.
> - One MCP session is held open per plugin for its lifetime (via an `AsyncExitStack`),
>   so stdio servers keep their process and state between calls; tool calls carry a
>   120-second timeout.
> - Streamed tool calls are read from the merged `AIMessageChunk`, not keyed by chunk
>   index, so a batch of calls in one chunk stays separate.
> - LangSmith telemetry is disabled at startup unless the user opted in explicitly.

Replaces two direct dependencies
(`litellm`, `mcp`) and the hand-rolled agent loop with the LangChain ecosystem
(`langchain-core`, `langchain-ollama`, `langchain-openai`, `langchain-mcp-adapters`,
`langgraph`).

> **Review note (2026-09):** Verified against current ecosystem docs before writing:
>
> - `langchain-mcp-adapters` `MultiServerMCPClient` supports `stdio`, `streamable_http`,
>   `sse`, and `websocket` connections, plus a `tool_name_prefix` option for name
>   collision handling and a `handle_tool_errors` option that converts MCP `isError=True`
>   results into error ToolMessages (matches our current failed-result semantics).
> - The adapters are a thin wrapper over the **official MCP SDK** — `mcp` remains a
>   *transitive* dependency. "Replacing the MCP SDK" here means removing our direct dep and
>   the hand-rolled client (`plugins/client.py`), not removing the protocol library.
> - `ChatOllama` (langchain-ollama) exposes `num_ctx`, `num_gpu`, `num_thread`,
>   `num_predict`, `temperature`, `seed`, `top_k`, `top_p`, `stop`, `format`,
>   `keep_alive`, `repeat_penalty`, `repeat_last_n`, `tfs_z`, `mirostat*`. It does **not**
>   expose `num_batch`, `use_mmap`, `use_mlock`, `num_keep`, `main_gpu`, `low_vram`,
>   `flash_attention` — see the Ollama options risk below.

## Problem

The runtime sits on two independent third-party client stacks that overlap with what the
LangChain suite already provides, and the agent loop re-implements graph machinery that
LangGraph gives us for free:

1. `llm/client.py` wraps LiteLLM (`acompletion`) for chat + streaming; `llm/providers.py`
   encodes LiteLLM model prefixes (`ollama_chat/`, `hosted_vllm/`, `lm_studio/`…).
2. `index/embeddings.py` uses `litellm.aembedding` for non-Ollama backends and the native
   `ollama` client for Ollama.
3. `plugins/client.py` hand-rolls an MCP client over `mcp.ClientSession` /
   `stdio_client` / `sse_client` / `streamablehttp_client`.
4. `agent/loop.py` implements its own loop: streaming, tool-call parsing, a text-JSON
   fallback, repeat-detection, truncation, iteration limits — all machinery LangGraph
   provides (events, recursion limits, trimming).
5. `instructor` is still a declared dependency but is imported nowhere (dead, per
   CLAUDE.md).

Goal of the change: one abstraction family for models, tools, and agent execution, and a
smaller list of *direct* dependencies.

## Goals / Non-goals

**Goals**
- Chat + streaming via per-provider LangChain chat models (`ChatOllama`, `ChatOpenAI`).
- Embeddings via `OllamaEmbeddings` / `OpenAIEmbeddings`.
- MCP plugins via `MultiServerMCPClient`.
- The agent loop becomes a LangGraph `StateGraph` (custom nodes, not `create_react_agent`
  — see the graph design below).
- **Feature parity with today's safety/UX behavior** (confirmation flow, permission modes,
  banned commands, hooks, repeat-guard, text-JSON fallback, streaming, token display).
- Keep the `run_agent()` entry-point signature stable so `cli.py`, `ui/tui`, and
  `sessions` do not change.
- Drop the dead `instructor` dep.

**Non-goals**
- No LangChain memory/vectorstore abstractions — the LanceDB + Tree-sitter index and
  `assemble_context()` stay as-is.
- No LangGraph checkpointer for cross-turn persistence — mita's JSON session store remains
  the source of truth; one graph invoke per REPL turn (current semantics).
- No LangChain prompt templates — `assemble_context()` keeps producing the system prompt.

## Target architecture

```
cli.py / ui/tui          (unchanged — call run_agent())
   │
   ▼
agent/loop.py ── run_agent(): builds a LangGraph StateGraph, invokes it per turn,
                 fires session_start/end hooks around the invoke.
   │
   ├── model node:  bind_tools(registry.get_openai_schemas()) → chat model
   │                streaming via astream_events → UISink
   │                text-JSON fallback (reuses _extract_tool_calls_from_text)
   ├── tools node:  for each AIMessage.tool_calls →
   │                mita.tools.executor.execute_tool()  (ALL safety logic unchanged)
   │                pre/post_tool_call + on_file_write hooks
   │                sink.tool_call / sink.tool_result
   │                append ToolMessages (backfill on interruption)
   └── retrieve node: RAG injection (replaces previous RAG system message)

tools/ToolRegistry + executor     (unchanged — single source of tool schemas + safety)
   └── plugins/PluginManager      (rewritten internals over MultiServerMCPClient)

llm/providers.py → llm/factory.py (provider enum → LangChain model class + kwargs)
index/embeddings.py               (OllamaEmbeddings / OpenAIEmbeddings)
```

`ToolRegistry` remains the single source of tool schemas and handlers. The graph's model
node binds the raw OpenAI schemas (`bind_tools` accepts plain schema dicts), and the tools
node dispatches through the existing `execute_tool()` so every safety gate, confirmation
prompt, and result cap is preserved byte-for-byte. MCP tools still register into the
registry via `PluginManager.register_tools()`, so built-in and plugin tools share one
execution path.

## Dependency changes

Remove from `pyproject.toml` (direct deps):
- `litellm`
- `instructor` (already unused)
- `mcp` (becomes transitive via `langchain-mcp-adapters`)

Add:
- `langchain-core`
- `langchain` (for `create_react_agent`/graph prebuilt utilities if used)
- `langchain-ollama`
- `langchain-openai`
- `langchain-mcp-adapters`
- `langgraph`

Keep: `ollama` (model management; also the underlying client langchain-ollama uses),
`httpx` (health + context probes), `lancedb`, `tree-sitter`, `prompt-toolkit`, `textual`,
`rich`, `typer`.

Pin versions at implementation time and confirm Python 3.11 compatibility and the
`astream_events(version="v2")` API against the pinned `langgraph`.

## Module changes

### `llm/`

- **`providers.py`** — `ProviderSpec.prefix` (LiteLLM model string) is replaced by a
  `model_class` selector plus constructor kwargs. Keep `ResolvedBackend` (provider,
  `api_base`, `api_key`, `is_ollama`, default base URL, api-key placeholder rule) — the
  fields feed both chat and embedding model construction.
- **`client.py` → `factory.py` (new)** — `build_chat_model(config) -> BaseChatModel` and
  `build_embedding_model(config) -> Embeddings`.
  - Ollama → `ChatOllama(model, base_url, temperature, num_predict=max_tokens, num_ctx,
    num_gpu, num_thread, …)` from the mapped runtime options.
  - Everything else → `ChatOpenAI(model, base_url, api_key, temperature, max_tokens)`.
  - Delete `client.py` (its `LLMClient`/`get_client` call sites are all inside the loop,
    which is being rewritten).
- **`streaming.py`** — delete. Token streaming is driven by LangGraph
  `astream_events` (`on_chat_model_stream` → `sink.stream_token`); the LiteLLM-shape
  `extract_delta_content` is no longer needed.
- **`context.py`, `health.py`** — unchanged (raw `httpx`, provider-agnostic).

### `index/embeddings.py`

Keep the `EmbeddingClient` interface (`embed_texts`, `embed_single`,
`is_model_available`, `model`). Swap internals:
- Ollama → `OllamaEmbeddings(model=..., base_url=...)`.
- Non-Ollama → `OpenAIEmbeddings(model=..., api_key=..., base_url=...)`.
`is_model_available()` keeps the Ollama `list()` check for the Ollama provider and the
"assume available" behavior otherwise.

### `plugins/`

- **`client.py`** — delete. Connection handling moves to
  `langchain_mcp_adapters.client.MultiServerMCPClient`.
- **`manager.py`** — keep the public API (`plugin_names`, `start_all`, `stop_all`,
  `start_plugin`, `stop_plugin`, `register_tools`, `test_plugin`) so `cli.py` is
  untouched. Rewrite internals:
  - Build the `connections` dict from `PluginDefinition`s (`stdio` → `command`/`args`,
    `sse` → `url`, `streamable_http` → `url`), mapping transport strings.
  - Load tools via `client.get_tools()` (set `tool_name_prefix=True` to preserve
    per-plugin namespacing).
  - **Preserve the credential-leak mitigation (C1):** the env allowlist for stdio
    subprocesses must be carried into the `StdioConnection` env. Verify
    `langchain-mcp-adapters` exposes env pass-through; if not, wrap the connection.
  - **Preserve destructive-by-default for plugin tools (C2):** `readOnlyHint` → attach
    `destructive` metadata to each LangChain tool so the tools node's confirmation gate
    behaves identically.
  - **Preserve the tool-name sanitizer** (`mcp_tool_name`, 64-char OpenAI charset) —
    LangChain tool names are not charset-limited, but strict OpenAI-compatible providers
    still reject bad names.
  - `test_plugin`: keep ping semantics via `get_server_info()` or a tool-list call
    (verify what the adapters expose; do not regress `mita plugins test`).
  - `handle_tool_errors=True` (default) is desired: MCP `isError` becomes an error
    ToolMessage the model can self-correct against.

### `tools/`

- **Unchanged core:** `ToolRegistry`, `executor.execute_tool()`, `safety.py`, all builtin
  handlers. The graph tools node reuses `execute_tool()` so banned-command checks,
  permission modes, `auto_approve`, `session_approved`, confirmation prompts, glob/grep
  caps, and shell timeout all survive unchanged.
- Optional helper `tools/langchain.py`: `to_langchain_tool(defn, handler) -> BaseTool`
  (StructuredTool with `args_schema` from `ToolDefinition.parameters`), used only if we
  later want to pass `BaseTool`s to `bind_tools`/`create_react_agent` instead of raw
  schema dicts. Not required for the custom-graph design.

### `agent/`

- **`loop.py`** — rewrite as a graph builder. Keep `run_agent()`'s signature and return
  type (`Conversation`).
  - **State:** `messages` (with `add_messages` reducer), plus `config`, `registry`,
    `sink`, `session_approved`, `last_tool_sig`, `iteration`, `conversation`.
  - **Nodes:**
    1. `retrieve` — RAG injection; replace the previous `_RAG_CONTEXT_PREFIX` system
       message (only when `index.enabled`).
    2. `model` — `truncate_to_fit(context_window)`; invoke the bound model; stream tokens
       to `sink`; text-JSON fallback when the model returns text but no native tool calls
       (reuse `_extract_tool_calls_from_text`).
    3. `tools` — for each `AIMessage.tool_calls`: run `execute_tool()` (safety +
       confirmation via `sink.confirm`), fire `pre/post_tool_call` + `on_file_write`
       hooks, emit `sink.tool_call`/`sink.tool_result`, append a `ToolMessage` per call.
       Backfill `"Not executed (interrupted)."` ToolMessages for any call left unanswered
       (the current guard against 400s on OpenAI-compatible endpoints).
  - **Edges:** model → tools (if tool calls and no repeat) → model; model → END when no
    tool calls; repeat-detection (identical consecutive tool-call signature) stops the
    turn with the same error message as today.
  - **Iteration guard:** a counter in state (incremented in the `model` node) with a
    conditional edge to END — keeps `max_iterations` exact. `recursion_limit` stays as a
    backstop, not the primary guard.
  - **Streaming:** `graph.astream_events(state, version="v2")`; `on_chat_model_stream`
    chunks feed `sink.stream_token`; node code emits `sink.assistant_message`,
    `sink.tool_call`, `sink.tool_result`, `sink.response_stats` (from
    `AIMessage.usage_metadata`). This single path serves both the Rich REPL and the
    Textual TUI via the existing `UISink` protocol.
  - **Error handling:** wrap the invoke exactly as today (KeyboardInterrupt/
    CancelledError, ConnectionError/TimeoutError/OSError, unexpected → `sink.error`,
    conversation stays valid, session_end hooks still fire).
  - `session_start`/`session_end` hooks fire in the `run_agent()` wrapper (unchanged).
- **`conversation.py`, `context.py`, `system_prompt.py`** — unchanged. The graph state's
  `messages` are read from / written back to the `Conversation` so sessions, resume, and
  the TUI keep working.

### `cli.py`, `ui/`, `sessions/`

No changes required if `run_agent()`'s signature is stable. Verify `mita chat`,
`mita ask --output json|text|rich`, `--tui`, `--resume`/`--continue`, and `/reload`,
`/resume`, `/clear` REPL commands end-to-end after the swap.

## Feature-parity matrix

| Current behavior | New location |
|---|---|
| LiteLLM chat + streaming | `llm/factory.py` → `ChatOllama`/`ChatOpenAI`; graph model node |
| `litellm.aembedding` | `OllamaEmbeddings`/`OpenAIEmbeddings` in `EmbeddingClient` |
| Provider registry (prefixes) | `providers.py` → `model_class` + kwargs |
| MCP stdio/SSE/streamable_http client | `MultiServerMCPClient` (stdio, sse, streamable_http) |
| MCP env allowlist (C1) | StdioConnection env pass-through (verify/preserve) |
| MCP destructive-by-default (C2) | `destructive` tool metadata from `readOnlyHint` |
| MCP tool-name sanitize (64-char) | kept, applied to LangChain tool names |
| Native tool calls + text-JSON fallback | model node (reuse `_extract_tool_calls_from_text`) |
| Repeat-detection / loop guard | tools node via `last_tool_sig` state |
| `max_iterations` | `iteration` counter in state (+ `recursion_limit` backstop) |
| `truncate_to_fit(context_window)` | model node |
| RAG injection per turn | `retrieve` node |
| Confirmation / permission modes / banned cmds | unchanged `executor.execute_tool()` |
| `session_approved` (allow-for-session) | graph state |
| `auto_confirm` (`ask --yes`) | graph state → confirm gate |
| Hooks (`session_start/end`, pre/post tool, on_file_write) | wrapper + tools node |
| Streaming tokens to Rich/TUI | `astream_events` → `UISink` |
| Token count display | `AIMessage.usage_metadata` |
| Tool-result backfill on interrupt | tools node `finally` path |
| Max-iteration / error messages | model/tools node sink.error |

## Migration phases

Each phase ends with the suite green (coverage ≥ 80%).

1. **Dependencies.** Add `langchain*` + `langgraph`; remove `litellm`, `instructor`, `mcp`
   (direct). Pin versions. Suite passes — nothing imports the new libs yet.
2. **LLM client swap.** `providers.py` + new `llm/factory.py`; `EmbeddingClient` internals;
   delete `llm/client.py`, `llm/streaming.py`; port `tests/test_llm/*`,
   `tests/test_index/test_embeddings.py`. The loop is temporarily wired to the factory so
   `mita chat` still works before the graph rewrite.
3. **Agent graph.** Build the `StateGraph` in `loop.py` with the same `run_agent()`
   signature; port streaming, fallback, repeat-guard, truncation, RAG, hooks, token stats;
   port `tests/test_agent/*` (994 lines — the bulk of the test work).
4. **MCP swap.** Rewrite `plugins/manager.py` internals over `MultiServerMCPClient`;
   delete `plugins/client.py`; port `tests/test_plugins/*`; verify env allowlist,
   destructive metadata, tool-name sanitizer, `mita plugins list/test`.
5. **Integration.** Exercise `chat`, `ask` (all output modes), `--tui`, resume/continue,
   REPL commands. Update `docs/reference/cli.md` if any help text changed.
6. **Cleanup.** Remove dead code, update `README.md` tech-stack table (drops Instructor,
   adds LangChain/LangGraph), add a CHANGELOG fragment, re-run `invoke check`.

## Backward compatibility

- **No config schema changes.** `[llm]`, `[ollama]`, `[model]`, `[tools]`, `[plugins]`,
  `[hooks]` sections and all enums are unchanged.
- `run_agent()` signature and return type are stable → `cli.py`, TUI, and sessions code
  are untouched.
- `.mita/sessions/*.json` format unchanged (system prompt regenerated on resume as today).
- Tool behavior unchanged — `executor.execute_tool()` and all builtin handlers are reused.
- `mcp` moves from direct to transitive dependency; no user-visible change.

## Risks & mitigations

1. **Ollama advanced runtime options** — `OllamaRuntimeOptions` has 10 fields; `ChatOllama`
   maps only `num_gpu`, `num_thread`, `num_ctx` (plus temperature/seed/top_p etc.).
   `num_batch`, `use_mmap`, `use_mlock`, `num_keep`, `main_gpu`, `low_vram`,
   `flash_attention` have no direct mapping. **Recommended:** map what exists, drop the
   rest with a one-line warning at startup (these are rarely used). Alternative: a small
   custom Ollama wrapper to forward the full `options` blob — rejected for now (complexity).
   → Open question 1.
2. **Streaming integration** — `astream_events` is more moving parts than the current
   `async for`. Mitigate: emit display from within nodes and only use events for token
   streams; pin the version and verify `version="v2"` chunk shapes in phase 3 with a
   fake model test.
3. **`mcp` stays transitive** — the "MCP SDK replacement" cannot be total without
   hand-rolling the protocol. Accepted; the goal is removing our direct dep and the
   hand-rolled client. → Open question 3.
4. **Plugin env allowlist (C1)** — if `MultiServerMCPClient` can't pass a safe env to
   stdio subprocesses, credential leakage regresses. Verify in phase 4; wrap the
   connection if needed. Non-negotiable.
5. **SSE transport** — `SSEConnection` exists, but SSE is deprecated by the MCP spec in
   favor of `streamable_http`. Keep `sse` working (current users have `transport = "sse"`)
   and document `streamable_http` as the forward path. → Open question 4.
6. **Tool-call reliability via `bind_tools` with raw dict schemas** on local models —
   same failure modes as today. Keep the text-JSON fallback in the model node; port the
   repeat-guard so a model stuck in a loop still stops.
7. **Dependency weight** — LangChain + LangGraph is a substantially larger install than
   LiteLLM. This is the explicit cost of consolidation; confirm it's acceptable for a
   "local-first" tool. → Open question 6.
8. **Version pinning / API drift** — `langgraph` moves fast (`astream_events` versions,
   `create_react_agent` signatures). Pin exact versions in `poetry.lock` and add a
   fake-model test that pins the event/chunk shape.

## Test plan

- **Factory:** provider → model class + kwargs mapping for all six providers; Ollama
  options mapping table; api-key placeholder rule.
- **Embeddings:** `EmbeddingClient` unit tests for Ollama and OpenAI-compatible paths.
- **Graph (`test_agent/*` rewrite):**
  - safety: banned command denied; confirmation allow / deny / allow-for-session;
    permission modes; `auto_confirm`; session approval persistence across turns.
  - hooks fire in order (session_start, pre/post tool, on_file_write, session_end).
  - repeat-guard stops identical consecutive tool calls.
  - text-JSON fallback parses a fake text response into tool calls.
  - RAG message injected and replaced per turn; index disabled → skipped.
  - truncation fits the window; `max_iterations` stops the turn.
  - token stats emitted when `show_token_count`.
  - error paths: connection error, interrupt — conversation stays valid, ToolMessages
    backfilled.
- **Plugins:** tool registration metadata (destructive from readOnlyHint), env allowlist,
  tool-name sanitization, start/stop lifecycle, `mita plugins list/test`.
- **Regression:** `mita chat` / `mita ask` / `--tui` smoke tests against a fake model.

## Cleanup

- Delete `src/mita/llm/client.py`, `src/mita/llm/streaming.py`, `src/mita/plugins/client.py`.
- Remove `instructor` from `pyproject.toml` and README's tech-stack table.
- Update `README.md` tech stack: drop LiteLLM + Instructor, add LangChain/LangGraph.
- Remove `tests/test_llm/test_instructor.py` if it still exists (already dead per CLAUDE.md).
- Add a CHANGELOG fragment per repo convention.

## Open questions

1. Advanced Ollama options: map the supported subset and drop the rest with a warning
   (recommended), or implement a custom Ollama wrapper to preserve all ten fields?
2. Confirm custom `StateGraph` over `create_react_agent` — the latter can't cleanly host
   the confirmation gate, hooks, or the text-JSON fallback without middleware; the custom
   graph keeps `executor.py` unchanged.
3. Accept `mcp` as a transitive dependency (recommended) vs. hand-rolling the MCP protocol
   to eliminate it (not recommended).
4. Keep `sse` transport support (deprecated by the spec) or treat `streamable_http` as the
   only HTTP transport and warn on `sse` configs?
5. Confirm per-turn graph invocation (one invoke per REPL input) stays — i.e., no
   persistent graph/checkpointer across turns.
6. Is the larger install footprint acceptable for a local-first tool?