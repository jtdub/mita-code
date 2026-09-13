# Design: opencode as a Mita backend via LangChain

Status: **proposal**. No code written yet. Companion to
`docs/design/langchain-migration.md` — assumes the LangChain migration landed.

## Problem

Mita runs models entirely on the user's machine. The user also runs opencode, which has
access to cloud providers (opencode Zen and any configured provider). The request: let
Mita "use opencode via langchain", in two senses:

1. **opencode as a plugin (short term).** opencode is driven through its HTTP server
   (`opencode serve`) or via community MCP bridges (`mcp-server-opencode`). After the
   MCP→`langchain-mcp-adapters` migration, an opencode MCP server is just another plugin:
   `mita plugins add opencode --command "npx -y mcp-server-opencode"`. Its tools flow into
   the LangGraph agent as LangChain tools, and Mita delegates coding subtasks to opencode
   as a subagent. No custom adapter code.
2. **opencode as the `[llm]` backend (long term).** Add `opencode` to `LLMProvider` and
   point the LangChain model factory at opencode's server. This is the hard part and the
   subject of this document.

The rest of this document covers (2).

## Why opencode cannot be a plain OpenAI-compatible backend

`opencode serve` publishes an OpenAPI 3.1 spec (default `http://127.0.0.1:4096/doc`)
with a custom JSON-RPC-style API: `/global/health`, `/project`, `/instance`,
`/session`, `/event` (SSE), `/tui`. It has **no `/v1/chat/completions`**. `ChatOpenAI`
cannot point at it, so the factory must return a custom `BaseChatModel` that speaks the
opencode protocol.

## The core tension: opencode is an agent, not a model

opencode does not expose raw inference. Its server runs a full agent loop: it owns the
model call, its built-in tools, MCP servers, permissions, and sessions. That changes what
"opencode as a model" means:

- **Mita would not execute tools.** opencode runs its own tools against the user's
  workspace. Mita's safety layer — `executor.py` (banned commands, permission modes,
  confirmation, `session_approved`) and its hooks — would NOT gate opencode's tool calls.
  The LangGraph tools node becomes a pass-through that watches opencode's output.
- **Tool schemas.** `bind_tools` arrives in `_generate` as a `tools` kwarg, but opencode's
  API has no way to accept a caller-supplied tool schema; opencode decides its own tools.
  Mita's seven built-in tools and MCP plugin tools would be invisible to opencode.

So the bridge is not "LangChain model → opencode model". It is "Mita agent → opencode
agent", wrapped so it satisfies the `BaseChatModel` contract. Treat it as a subagent
adapter, not an inference client.

## Option A: model-graph passthrough (recommended)

`OpencodeChatModel(BaseChatModel)` maps one Mita turn to one opencode session prompt.

For each Mita `chat`/`ask` turn:

1. Ensure an opencode server is reachable (`opencode serve`, `--port`/`--hostname` from
   `[llm] base_url`; basic auth via `OPENCODE_SERVER_USERNAME`/`PASSWORD` or the
   `[llm] api_key`).
2. Open an opencode instance for the Mita working directory (reuse one instance per Mita
   session so opencode holds workspace state; `x-opencode-directory` selects the project).
3. Send the Mita conversation tail (the last user message plus, optionally, a condensed
   history) as a session prompt.
4. Stream opencode's `/event` stream: `session.request.created` /
   `message.part.updated` text parts → `sink.stream_token`; terminal `session.updated`
   → the final assistant message.
5. Mita records opencode's final message as the assistant reply. Mita does not call its
   tools; opencode did. The LangGraph loop is bypassed for tool execution in this mode.

Mapping to `BaseChatModel`:
- `_generate` → non-streaming prompt + final message.
- `_stream` → the event stream above (yield `ChatGenerationChunk`s, fire
  `on_llm_new_token`).
- `bind_tools`/`tools` kwarg → ignored with a logged warning; opencode owns tools.
- `usage_metadata` → best-effort from `session.usage` if the server reports it, else None
  (token count display degrades gracefully).

This is honest about the trade: in `opencode` provider mode Mita becomes a thin shell over
opencode's agent. Recommended because it is small and predictable.

## Option B: opencode model behind Mita's tools (not recommended now)

Expose opencode's configured model to Mita's own tool loop. Requires opencode to run a
model with a caller-supplied tool set, which its server API does not offer today. Would
need an opencode server feature (model passthrough with external tools) to exist first.
Defer until opencode exposes it; revisit if opencode adds an OpenAI-compatible inference
endpoint.

## Config

```toml
[llm]
provider = "opencode"
base_url = "http://127.0.0.1:4096"   # opencode serve default
api_key = ""                          # basic-auth password (OPENCODE_SERVER_PASSWORD)
```

- No model registry, no context probe → `context_probe = "none"`, fall back to
  `model.context_window`.
- `model.default` is advisory (opencode picks the agent/model from its own config).
- Health check: `GET /global/health` (reuse `llm/health.py` with a new strategy).
- `mita doctor`: report the opencode server version when reachable.

## Module changes

- `config/schema.py`: add `OPencode = "opencode"` to `LLMProvider` (exact token
  `"opencode"`).
- `llm/providers.py`: `ProviderSpec(model_class="opencode", default_base_url=
  "http://127.0.0.1:4096", has_model_registry=False, needs_api_key_placeholder=False,
  context_probe="none")`.
- `llm/opencode.py` (new): `OpencodeChatModel` implementing the protocol above against
  `httpx` (client for the JSON-RPC endpoints) + SSE streaming. No new third-party deps.
- `llm/factory.py`: return `OpencodeChatModel` when `provider == "opencode"`.
- `cli.py` `_preflight_backend` / `doctor`: generic reachability already exists for
  non-Ollama providers; add the `/global/health` strategy.

## Test plan

- Unit: `OpencodeChatModel` maps a fake opencode event stream to text chunks and a final
  message; `tools` kwarg is tolerated and logged.
- Unit: provider registry entry, health strategy, base_url default, doctor output.
- Integration (marked, requires a live `opencode serve`): one `mita ask --no-tools`
  turn returns opencode's answer.
- Regression: every other provider path unchanged; token-count display degrades when no
  usage is reported.

## Open questions

1. Instance reuse: one opencode instance per Mita session (holds workspace state) vs. one
   instance per turn (isolated, slower). Lean: per-session reuse.
2. History: send only the latest user message (opencode keeps its own session) vs. replay
   condensed Mita history. Lean: latest message only.
3. Confirm the accepted trade: in `opencode` provider mode, Mita's tools, safety gates,
   and hooks do not run — opencode owns tool execution. If that is unacceptable, use the
   MCP-plugin route (sense 1) instead, and keep this backend experimental.

## Cleanup

Nothing to remove. Land this after `docs/design/langchain-migration.md` phases 1–4 so the
factory and health modules exist.