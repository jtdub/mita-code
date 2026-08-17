# Design: Multi-backend LLM support (audit finding C5)

Status: **proposal — revised after technical review**. No code written yet.

> **Review note (2026-08):** Reviewed against LiteLLM's documented provider behavior.
> Two claims in the first draft were wrong and are corrected below (the `drop_params`
> rationale and the empty-`api_key` default). The verified-correct claims: vLLM →
> `hosted_vllm/` with an `api_base` ending in `/v1`; generic OpenAI-compatible →
> `openai/<model>` + `api_base`.

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
`[ollama]` for Ollama-specific daemon management only.

```toml
[llm]
provider = "ollama"        # ollama | llamacpp | vllm | lmstudio | tgi | openai_compatible
base_url = ""              # empty = provider default (see table)
api_key = ""               # empty is fine for local servers (see api_key handling below)
context_probe = "auto"     # auto | off | <int>  (auto: query the backend, else static)

[model]
default = "qwen2.5-coder:7b"
context_window = 32768     # fallback when probe is off/unavailable
embedding = "nomic-embed-text"
```

Provider defaults:

| provider          | base_url default            | LiteLLM prefix          | health check   | context probe          | embeddings           |
|-------------------|-----------------------------|-------------------------|----------------|------------------------|----------------------|
| ollama            | http://localhost:11434      | `ollama_chat/` (see S1) | `/api/tags`    | `/api/show` model_info | native `/api/embed`  |
| llamacpp          | http://localhost:8080/v1    | `openai/` + base_url    | `/v1/models`   | `GET /props` (`n_ctx`) | `/v1/embeddings`     |
| vllm              | http://localhost:8000/v1    | `hosted_vllm/`          | `/v1/models`   | `/v1/models.max_model_len` | `/v1/embeddings` |
| lmstudio          | http://localhost:1234/v1    | `lm_studio/`            | `/v1/models`   | none — use fallback    | `/v1/embeddings`     |
| tgi               | http://localhost:8080/v1    | `openai/` + base_url    | `/v1/models`   | none — use fallback    | `/v1/embeddings` (1) |
| openai_compatible | (required)                  | `openai/` + base_url    | `/v1/models`   | none — use fallback    | `/v1/embeddings`     |

(1) only if an embedding model is served.

**api_key handling (review B2).** LiteLLM's `openai/`-routed path is built on the OpenAI
client, which **requires a non-empty api_key**. For the `openai/`-routed providers
(llamacpp, tgi, openai_compatible) an empty config `api_key` must be substituted with a
placeholder (`"sk-no-key-required"`) before the call, or LiteLLM raises an auth error even
though local servers need no auth. `lm_studio/` and `ollama_chat/` tolerate an empty key.

## Module changes

- **New `llm/providers.py`**: a `Provider` registry mapping the enum to the table above —
  LiteLLM prefix, default base_url, health-check path, context-probe strategy, whether a
  model registry exists, and the api_key placeholder rule.
- **`llm/client.py`**: build the model string and `api_base`/`api_key` from the provider,
  not a literal.
  - `litellm.drop_params = True` — this drops **only recognized OpenAI params** a backend
    doesn't support (e.g. `stream_options`). **It does NOT drop the Ollama `options` blob**
    (review B1): that is passed via `extra_body`, which LiteLLM forwards untouched. The
    correct fix is to **attach the `options` blob only when `provider == ollama`** at
    call-construction time (`client.py:54,80`), not to rely on `drop_params`.
- **`llm/context.py` (new)**: `probe_context_window(provider, model)` uses the per-provider
  strategy in the table (vLLM `/v1/models.max_model_len`; llama.cpp `GET /props` `n_ctx`;
  Ollama `/api/show` model_info — already surfaced by `OllamaClient.show()`,
  `ollama_client.py:103-115`). LM Studio / TGI / generic have no reliable probe → fall back
  to `model.context_window`. Cache per process.
- **`models/server.py` + `cli.py` preflight**: `ensure_server`/`ensure_model` become a
  generic `GET base_url/models` health check for non-ollama providers, and never spawn
  `ollama serve` unless provider == ollama.
- **`cli.py doctor`**: report health for the configured provider generically; only show the
  Ollama binary/daemon checks when provider == ollama.
- **`models` subcommands**: `list` reports the loaded model(s) via `/v1/models` when there
  is no registry; `pull`/`rm` print "not supported for provider X"; `recommend`/`hardware`
  still run **but their output is Ollama-only** — the registry *model identifiers themselves*
  are Ollama registry tags (`registry.py:27-134`), un-pullable on other backends, so
  `recommend` is advisory-only for non-Ollama providers (review M2).
- **`index/embeddings.py`**: route through LiteLLM `aembedding(...)` for OpenAI-compatible
  providers; keep the native `ollama` path for provider == ollama. **Normalize the response
  shape (review S5):** the current code reads `response.embeddings` (Ollama shape,
  `embeddings.py:37,46`); LiteLLM returns `response.data[i]["embedding"]`. The caller must
  branch on provider.
- **Exception funnels**: `agent/loop.py` RAG block (currently
  `except (ConnectionError, FileNotFoundError, ImportError, OSError)`, `loop.py:104`) and
  `index/manager.py` catch `ollama.ResponseError` and `httpx.HTTPError` (or a normalized
  `LLMBackendError`) so a missing/incompatible embedding endpoint degrades instead of
  crashing the turn.
- **Capability detection + ReAct fallback** (per original prompt #2): a one-time probe
  attempts a tiny native tool-call; if the backend doesn't support function calling, fall
  back to the existing text-JSON tool-call path (already in `loop.py`), and cache the
  capability.
- **Env-var expansion (review S4, also audit N5):** the config loader does no `${VAR}`
  substitution today. Reading `api_key`/`base_url` from the environment requires adding that
  expansion (a small, separately-tested change) OR relying on LiteLLM's own `*_API_KEY`
  env convention. This is a prerequisite, not a free feature.

## Backward compatibility & the `ollama_chat/` switch (review S1)

- No `[llm]` section → provider `ollama`, existing `[ollama] host` used as base_url.
- **This is NOT "identical behavior."** Switching Ollama's prefix from `ollama/`
  (`/api/generate`, prompt-templated) to `ollama_chat/` (`/api/chat`, native `tools` array)
  changes the request path and tool-calling for every existing user. The switch is worth it
  (native tool calls), but must be treated as a behavioral change:
  - Add a regression test asserting native tool calls **and** that `model.ollama_options`
    (num_ctx/num_gpu/etc., `schema.py:54-74`) are still forwarded under `ollama_chat/` — the
    `extra_body` mapping differs between the two prefixes and could silently drop them.
  - **Risk:** known LiteLLM issues where `ollama_chat/` tool calls via the completions API
    return no `tool_calls` (BerriAI/litellm #11104, #11433). Validate against the pinned
    litellm `^1.82.0` during implementation; if it bites, keep `ollama/` as default and make
    `ollama_chat/` opt-in.

## Test plan

- Unit: provider registry (prefix/base_url/probe/api_key-placeholder) for all six providers.
- Unit: model-string + api_base construction; `options` blob attached only for ollama;
  `drop_params` set.
- Integration: a mocked OpenAI-compatible server exercising chat, streaming, `/v1/models`,
  and `/v1/embeddings` (with shape normalization) for at least vLLM and llamacpp.
- Regression: no `[llm]` section reproduces current Ollama requests; tool calls + options
  survive the `ollama_chat/` switch.
- doctor: generic health path for a non-ollama provider; no `ollama serve` spawn.

## Open questions for you

1. One `[llm]` section (proposed) vs. extending `[model]` with `provider`? I prefer a
   dedicated `[llm]` section so daemon-management stays in `[ollama]`.
2. Should `pull` for non-registry backends be a hard error or a no-op with a note? I lean
   no-op-with-note so scripts don't break.
3. `ollama_chat/` as the new default (my recommendation, with the regression tests above),
   or keep `ollama/` default and make `ollama_chat/` opt-in until the LiteLLM tool-call
   bugs are confirmed fixed?

## Cleanup (verified safe)

- Delete `llm/instructor.py` — confirmed unused in `src/` (nothing imports
  `InstructorClient`/`get_instructor_client`). **Also delete `tests/test_llm/test_instructor.py`**,
  which imports it and would otherwise break the suite (review M1).
