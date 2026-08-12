# Design: Multi-backend LLM support (audit finding C5)

Status: **proposal — awaiting review**. No code written yet.

## Problem

The runtime is coupled to Ollama at four independent layers, which contradicts the
"backend-agnostic via LiteLLM" design in CLAUDE.md/PLANNING.md. Pointed at any other
OpenAI-compatible server (llama.cpp `llama-server`, vLLM, LM Studio, TGI) `mita chat`
and `mita ask` fail before reaching the model.

Coupling points (from the audit):
1. Model string is a hardcoded `f"ollama/{model}"` (`llm/client.py:20`, dup in `llm/instructor.py`).
2. No `provider`/`base_url`/`api_key` anywhere in the config schema; only `[ollama]`.
3. `chat`/`ask` run an Ollama REST preflight (`ensure_server`/`ensure_model`) that hard-exits.
4. Embeddings call Ollama's native `/api/embed` (`index/embeddings.py`).
5. Context length is a static `32768` with no probe.
6. Several exception funnels don't catch `ollama.ResponseError`, so a non-Ollama
   endpoint crashes `run_agent` rather than degrading.

## Proposed config schema

Add a `[llm]` section that is the single source of truth for the backend. Keep
`[ollama]` for Ollama-specific daemon management only. Backward compatible: with no
`[llm]` section, provider defaults to `ollama` and behavior is unchanged.

```toml
[llm]
provider = "ollama"        # ollama | llamacpp | vllm | lmstudio | tgi | openai_compatible
base_url = ""              # empty = provider default (see table)
api_key = ""               # or ${ENV_VAR}; sent as Authorization: Bearer
context_probe = "auto"     # auto | off | <int>  (auto: query the backend, else static)

[model]
default = "qwen2.5-coder:7b"
context_window = 32768     # fallback when probe is off/unavailable
embedding = "nomic-embed-text"
```

Provider defaults:

| provider          | base_url default            | LiteLLM prefix          | health check        | model list        | embeddings           |
|-------------------|-----------------------------|-------------------------|---------------------|-------------------|----------------------|
| ollama            | http://localhost:11434      | `ollama_chat/`          | `/api/tags`         | `/api/tags`       | `/api/embed`         |
| llamacpp          | http://localhost:8080/v1    | `openai/` + base_url    | `/v1/models`        | `/v1/models` (1)  | `/v1/embeddings`     |
| vllm              | http://localhost:8000/v1    | `hosted_vllm/`          | `/v1/models`        | `/v1/models`      | `/v1/embeddings`     |
| lmstudio          | http://localhost:1234/v1    | `openai/` + base_url    | `/v1/models`        | `/v1/models`      | `/v1/embeddings`     |
| tgi               | http://localhost:8080/v1    | `openai/` + base_url    | `/v1/models`        | `/v1/models` (1)  | `/v1/embeddings` (2) |
| openai_compatible | (required)                  | `openai/` + base_url    | `/v1/models`        | `/v1/models`      | `/v1/embeddings`     |

(1) returns the single loaded model. (2) only if an embedding model is served.

Note `ollama/` → `ollama_chat/`: the current prefix routes through prompt-templating
instead of a native tools array even on Ollama; switching improves tool calling.

## Module changes

- **New `llm/providers.py`**: a `Provider` dataclass/registry mapping the enum to the
  table above — LiteLLM prefix builder, default base_url, health-check path, whether a
  model registry exists. One place, no logic scattered.
- **`llm/client.py`**: build the model string and `api_base`/`api_key` from the provider,
  not a literal. Set `litellm.drop_params = True` so provider-unsupported params
  (`stream_options`, Ollama `options`) are dropped rather than erroring. Only attach the
  Ollama `options` blob when provider == ollama.
- **`llm/context.py` (new small helper)**: `probe_context_window(provider, model)` →
  queries `/v1/models` (OpenAI `max_model_len`/`context_length`) or Ollama `/api/show`;
  falls back to `model.context_window`. Cache per process.
- **`models/server.py` + `cli.py` preflight**: `ensure_server`/`ensure_model` become
  no-ops (or a generic `GET base_url/models` health check) for non-ollama providers, and
  never spawn `ollama serve` unless provider == ollama.
- **`cli.py doctor`**: report health for the configured provider generically; only show
  the Ollama binary/daemon checks when provider == ollama.
- **`models` subcommands**: `list` reports the loaded model(s) via `/v1/models` when there
  is no registry; `pull`/`rm` print "not supported for provider X" instead of crashing;
  `recommend`/`hardware` still run (note their tags are Ollama-oriented).
- **`index/embeddings.py`**: route through LiteLLM `aembedding(provider_prefix+model, api_base, api_key)`
  for OpenAI-compatible providers; keep the native `ollama` path for provider == ollama.
- **Exception funnels**: `agent/loop.py` RAG block and `index/manager.py` catch
  `ollama.ResponseError` and `httpx.HTTPError` (or a normalized `LLMBackendError`) so a
  missing/incompatible embedding endpoint degrades instead of crashing the turn.
- **Capability detection + ReAct fallback** (per original prompt #2): a one-time probe
  attempts a tiny native tool-call; if the backend doesn't support function calling, fall
  back to the existing text-JSON tool-call path (already present in `loop.py`), and record
  the capability so we don't reprobe each turn.

## Backward compatibility & migration

- No `[llm]` section → provider `ollama`, existing `[ollama] host` used as base_url,
  identical behavior. No migration needed for existing users.
- `[ollama] host` remains honored as the ollama provider's base_url for one release, with
  a deprecation note pointing to `[llm] base_url`.

## Test plan

- Unit: provider registry (prefix/base_url/paths) for all six providers.
- Unit: model-string + api_base construction per provider; `drop_params` set.
- Integration: a mocked OpenAI-compatible server (aiohttp/pytest) exercising chat,
  streaming, `/v1/models`, and `/v1/embeddings` for at least vllm and llamacpp providers.
- Regression: provider defaulting to ollama with no `[llm]` reproduces current requests.
- doctor: asserts generic health path for a non-ollama provider and no `ollama serve` spawn.

## Open questions for you

1. One `[llm]` section (proposed) vs. extending `[model]` with `provider`? I prefer a
   dedicated `[llm]` section so daemon-management stays in `[ollama]`.
2. Should `pull` for non-registry backends be a hard error or a no-op with a note? I lean
   no-op-with-note so scripts don't break.
3. Keep the dead `llm/instructor.py` (delete, per audit) or repurpose it for the ReAct
   fallback? I lean delete + build the fallback into `loop.py` where the text parser lives.
