# Design: Index chunking + freshness rewrite (audit finding C6)

Status: **proposal — revised after technical review**. No code written yet. Changes the
on-disk index format, so this includes a migration path.

> **Review note (2026-08):** The design was reviewed against `lancedb==0.25.3` run
> locally. All proposed LanceDB features exist in that version: explicit pyarrow schema,
> `merge_insert` upsert, `delete(where=...)`, native FTS (`create_fts_index`,
> `use_tantivy=False` — **no `tantivy` dependency**), `query_type="hybrid"` with a
> caller-supplied vector, and a built-in `RRFReranker`. The corrections below are folded
> into this revision.

## Problems (from the audit)

1. **~31% content loss.** The chunker keeps only direct children of the syntax root, and
   the line-window fallback runs only when tree-sitter yields *zero* chunks
   (`index/parser.py:107-113`). Module-level constants, imports, and config tables are
   never indexed. Measured 69% line coverage on the repo's own `parser.py`.
2. **Broken symbol extraction.** `_extract_symbol` returns `None` for C/C++/TS/Ruby,
   decorated Python defs, and every ESM export (only checks direct `identifier`/`name`/
   `property_identifier` children, `parser.py:236-240`).
3. **Full-rebuild only, no freshness.** No incremental path; nothing detects a stale index;
   `run_agent` injects stale/deleted-file content as current.
4. **Missing metadata.** LanceDB schema is inferred; no content hash, mtime, git SHA,
   symbol path, or embedding-model stamp → can't do incremental or detect model drift.
5. **Pure vector kNN.** No lexical/BM25 signal, no relevance threshold, re-embeds the raw
   user string every turn (so "yes"/"continue" swap in garbage context).
6. **Bad defaults.** Excludes omit `.venv` etc., so indexing embeds the whole virtualenv;
   secrets (`.env`, `*.pem`) are line-chunked and embedded.

## Proposed explicit LanceDB schema

Declared with a pyarrow schema / `LanceModel` (not inferred). One row per chunk:

| column          | type      | purpose                                             |
|-----------------|-----------|-----------------------------------------------------|
| chunk_id        | string    | STABLE id: `sha1(file_path + symbol_path + occurrence_ordinal)` — see below |
| file_path       | string    | repo-relative                                       |
| symbol_path     | string    | e.g. `module.ClassName.method` (empty for gap chunks) |
| occurrence_ordinal | int32  | disambiguates duplicate/empty symbol_paths within a file |
| language        | string    |                                                     |
| chunk_type      | string    | function/class/method/module/docstring/comment/config |
| start_line      | int32     | MUTABLE — updated in place on line shift             |
| end_line        | int32     | MUTABLE                                              |
| content         | string    |                                                     |
| content_hash    | string    | sha1 of content — the re-embed decision key         |
| git_blob_sha    | string    | when available                                       |
| mtime           | double    | source file mtime at index time                     |
| embed_model     | string    | model + version stamp                               |
| vector          | fixed_size_list<float>[dim] |                                   |

**Chunk-id stability (review B1).** The id must NOT include `start_line`: a function that
only shifts down (because an import was added above it) would otherwise get a new id and be
needlessly re-embedded, defeating the incremental goal. Identity is
`sha1(file_path + symbol_path + occurrence_ordinal)`; `start_line`/`end_line` are mutable
columns updated in place. The **upsert key is `chunk_id`**; the **re-embed decision is gated
on `content_hash`**. A pure line shift updates line numbers via `merge_insert` without
re-embedding. `occurrence_ordinal` disambiguates gap chunks (whose `symbol_path` is empty)
and repeated symbol paths within one file.

Field names track the cross-repo "shared ingestion schema" (prompt #0) for Hearth AI
interop. If that shared doc is run first, adopt its names verbatim.

## Chunking

- Walk the syntax tree **recursively** for definition nodes (functions, classes, methods,
  impl blocks), not just root children.
- Content NOT covered by any symbol chunk (imports, module constants, top-level config
  tables) is captured as `module`/`config` chunks via line windows over the gaps — so
  nothing is silently dropped. Guard this with a test asserting imports/constants ARE
  indexed.
- Rewrite `_extract_symbol` to use `node.child_by_field_name("name")` as the **primary**
  strategy (verified to resolve C++ class, Ruby class, Go method, TS function, Python def
  in one call), with per-language fallbacks. **C/C++ still need declarator recursion** —
  the name lives under the `declarator` field, not `name`. Prefer the field-name API over
  an exhaustive node-type table, which rots as grammars change. Track enclosing scope to
  build `symbol_path`.
- Extend `CHUNK_NODE_TYPES` to the languages already in `EXTENSION_MAP` (php, swift,
  kotlin, scala, lua, bash — all confirmed available in `tree-sitter-language-pack>=0.6.0`)
  or explicitly line-chunk them (documented, not silent).

## Incremental reindex + freshness

- `mita index build` diffs by `content_hash`: only new/changed chunks are re-embedded;
  rows for chunks no longer present are deleted via `delete(where=...)`. Full rebuild
  becomes `--rebuild`, the exception.
- **Operation ordering (review M6):** upsert changed rows via `merge_insert` on `chunk_id`
  FIRST, then delete removed rows — safer for query availability. These are separate
  LanceDB transactions (single-writer optimistic concurrency), so a crash between them
  leaves a partially-updated but still-queryable index; the next build reconciles it.
- **FTS index (review B2):** after any content change, rebuild the FTS index with
  `create_fts_index("content", use_tantivy=False, replace=True)` (also index `symbol_path`).
  Hybrid/FTS queries error if no FTS index exists, so this is mandatory, not optional.
  Newly `merge_insert`ed rows are searchable immediately (Lance brute-forces the unindexed
  delta), but periodically `optimize()`/rebuild the index or query latency degrades.
- **Embedding-model change forces a full rebuild (review S5):** `vector` is a fixed-width
  column, and nomic-embed-text (768-dim) vs bge-small (384-dim) differ, so a `merge_insert`
  of a differently-sized vector fails at the Arrow layer. A changed `embed_model` routes
  through the `index_version` migration (full rebuild), never an incremental update.
- Change detection source order: git blob SHA when in a repo, else `content_hash` + mtime.
- **Staleness check (review S4):** detect from the **working tree**, not HEAD — comparing
  against HEAD misses uncommitted edits, which are exactly the stale-context problem.
  Use `git status --porcelain` (changed + untracked source files) in a repo, falling back
  to an mtime scan (including new files absent from the index) outside a repo. If stale,
  warn before `mita ask`/`chat` use the index.
- `mita index status` reports build time, chunk count, index size, embed model, and #stale
  chunks (current status misreports these).

## Retrieval

- **Hybrid (review S1):** `query_type="hybrid"` with the Ollama-embedded query vector
  supplied directly (`.vector([...]).text(query)`), fused by the built-in `RRFReranker`
  (LanceDB's default; do NOT hand-roll fusion). The vector/FTS weight is configurable to
  weight function/class chunks above raw windows.
- **Relevance threshold (review S2):** the hybrid result column is `_relevance_score` (RRF,
  a small `1/(k+rank)`-scale value) which is NOT comparable to the old `1/(1+distance)`.
  Pin the floor to ONE documented score space — normalize scores (LanceDB `normalize`) or
  apply the floor to the pre-fusion cosine score — and ship a concrete default. Drop
  results below it so irrelevant turns inject nothing.
- **Token budget (review S3):** budget retrieval against a DEDICATED fraction of the window
  (new `[index] context_token_budget`), NOT the full `model.context_window` (which is shared
  with the system prompt, history, tool output, and generation). Reuse `CHARS_PER_TOKEN = 4`.
- Don't re-embed trivial follow-ups (skip when the user message is very short / lacks code
  tokens).
- **Note (review M9):** `nomic-embed-text` expects `search_document:`/`search_query:`
  prefixes; the current client adds none, which degrades recall. Add the prefixes if we
  stay on nomic.

## Ignore rules

- Respect `.gitignore` via real gitignore semantics — shell out to `git check-ignore` or
  use the `pathspec` library, NOT `fnmatch` (which can't express negation/anchoring/`**`).
  Add a `.mitaignore` for index-only excludes. Default excludes add `.venv`, `venv`,
  `.tox`, `site-packages`, `target`, `vendor`, `.next`, caches, `dist`.
- Add a `max_file_size` field to `IndexSettings` (`schema.py` has none today) and skip
  files above it.
- Skip likely-secret files — broaden the list to `.env*`, `*.pem`, `*.key`, `id_rsa`,
  `credentials*`, `.netrc`, `.npmrc`, `.pypirc`, `*.p12`, `*.pfx`, `*.keystore`.

## Migration

- Add an `index_version` marker file in `.mita/index/`. On version mismatch, missing
  metadata columns, or a changed `embed_model`, `mita index build` does a one-time full
  rebuild automatically and prints an ETA. **No data-loss risk** — the index is a gitignored
  derived cache; source code is the source of truth. Only cost is the re-embed on first run.

## Test plan

- Chunk-boundary correctness fixtures for Python, JS/TS, Go, Rust, C/C++, Ruby — asserting
  real symbol paths, including decorated/exported/receiver cases.
- Assert non-symbol top-level content (imports, constants) IS indexed (guards the 31% loss).
- Incremental: edit one file, assert only its changed chunks re-embed (count embed calls);
  a pure line-shift edit re-embeds NOTHING (chunk-id stability); delete a file, assert its
  rows are gone; assert the FTS index is rebuilt so a new row is findable by hybrid search.
- Ignore rules: `.venv` and a `.env` are not indexed; `git check-ignore` path respected.
- Retrieval: hybrid returns an exact-identifier match that pure-vector misses; threshold
  suppresses an irrelevant query.

## Open questions for you

1. Run the cross-repo "shared ingestion schema" design (prompt #0) **before** this, so the
   field names/chunk-id scheme are fixed for Hearth AI interop? I recommend yes.
2. Embedding model: keep `nomic-embed-text` (add task prefixes), or standardize on a
   `bge`-family model for cross-repo vector compatibility? **Caveat:** `bge-small` is NOT a
   standard Ollama tag (Ollama offers `bge-m3`, `nomic-embed-text`, `mxbai-embed-large`,
   `snowflake-arctic-embed`, `all-minilm`) and Mita embeds via Ollama today — verify
   pullability before committing. Depends on Q1.
3. Auto-update-on-stale by default, or warn-only? I lean warn-only with an opt-in
   `[index] auto_update = true`, to avoid surprise latency before a chat turn.
