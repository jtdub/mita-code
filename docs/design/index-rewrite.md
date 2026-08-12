# Design: Index chunking + freshness rewrite (audit finding C6)

Status: **proposal — awaiting review**. No code written yet. Changes the on-disk index
format, so this includes a migration path.

## Problems (from the audit)

1. **~31% content loss.** The chunker keeps only direct children of the syntax root, and
   the line-window fallback runs only when tree-sitter yields *zero* chunks
   (`index/parser.py:107-113`). Module-level constants, imports, and config tables are
   never indexed. Measured 69% line coverage on the repo's own `parser.py`.
2. **Broken symbol extraction.** `_extract_symbol` returns `None` for C/C++/TS/Ruby,
   decorated Python defs, and every ESM export.
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
| chunk_id        | string    | stable id: `sha1(file_path + symbol_path + start)`  |
| file_path       | string    | repo-relative                                       |
| symbol_path     | string    | e.g. `module.ClassName.method` (empty if none)      |
| language        | string    |                                                     |
| chunk_type      | string    | function/class/method/module/docstring/comment/config |
| start_line      | int32     |                                                     |
| end_line        | int32     |                                                     |
| content         | string    |                                                     |
| content_hash    | string    | sha1 of content — change detection                  |
| git_blob_sha    | string    | when available                                       |
| mtime           | double    | source file mtime at index time                     |
| embed_model     | string    | model + version stamp                               |
| vector          | fixed_size_list<float>[dim] |                                   |

Field names chosen to match the cross-repo "shared ingestion schema" (prompt #0) so a
Hearth AI import stays possible. If that shared doc is run first, adopt its names verbatim.

## Chunking

- Walk the syntax tree **recursively** for definition nodes (functions, classes, methods,
  impl blocks), not just root children.
- Content NOT covered by any symbol chunk (imports, module constants, top-level config
  tables) is captured as `module`/`config` chunks via line windows over the gaps — so
  nothing is silently dropped.
- Rewrite `_extract_symbol` per-language to handle `decorated_definition`,
  `export_statement`, `type_identifier`, `impl`, Go method receivers, and Ruby
  `constant` names. Track the enclosing scope to build `symbol_path`.
- Extend `CHUNK_NODE_TYPES` to the languages already in `EXTENSION_MAP` (php, swift,
  kotlin, scala, lua, bash) or explicitly line-chunk them (documented, not silent).

## Incremental reindex + freshness

- `mita index build` diffs by `content_hash`: only new/changed chunks are re-embedded;
  rows for chunks no longer present are deleted (handles renames/removals). Full rebuild
  becomes `--rebuild`, the exception.
- Change detection source order: git (`git diff`/blob SHA) when in a repo, else mtime.
- Staleness check: before `mita ask`/`chat` consult the index, compare tracked mtimes/HEAD
  against current; if stale, warn (and optionally auto-update behind a config flag).
- `mita index status` reports build time, chunk count, index size, embed model, and #stale
  chunks (finding: current status lies about these).

## Retrieval

- Hybrid: vector similarity + a LanceDB FTS (BM25) pass over `content`/`symbol_path`;
  combine with reciprocal-rank fusion. Weight function/class chunks above raw windows.
- Relevance threshold: drop results below a score floor so irrelevant turns inject nothing.
- Token budget: cap injected context against `model.context_window`; don't re-embed
  trivial follow-ups (skip when the user message is very short / lacks code tokens).

## Ignore rules

- Respect `.gitignore` plus a new `.mitaignore`; add default excludes
  (`.venv`, `venv`, `.tox`, `site-packages`, `target`, `vendor`, `.next`, caches, `dist`).
- Skip files over a configurable `max_file_size`.
- Skip likely-secret files (`.env*`, `*.pem`, `id_rsa`, `credentials*`) from indexing.

## Migration

- Add an `index_version` marker file in `.mita/index/`. On version mismatch or missing
  metadata columns, `mita index build` does a one-time full rebuild automatically and
  prints why. No manual step for users; the index is a local cache (gitignored).

## Test plan

- Chunk-boundary correctness fixtures for Python, JS/TS, Go, Rust, C/C++, Ruby — asserting
  real symbol paths, including decorated/exported/receiver cases.
- Assert non-symbol top-level content (imports, constants) IS indexed (guards the 31% loss).
- Incremental: edit one file, assert only its changed chunks re-embed (mock/count embed
  calls before/after); delete a file, assert its rows are gone.
- Ignore rules: `.venv` and a `.env` are not indexed.
- Retrieval: hybrid returns an exact-identifier match that pure-vector misses; threshold
  suppresses an irrelevant query.

## Open questions for you

1. Run the cross-repo "shared ingestion schema" design (prompt #0) **before** this, so the
   field names/chunk-id scheme are fixed for Hearth AI interop? I recommend yes.
2. Embedding model: keep `nomic-embed-text`, or standardize on `bge-small` for cross-repo
   vector compatibility? Depends on Q1.
3. Auto-update-on-stale by default, or warn-only? I lean warn-only with an opt-in
   `[index] auto_update = true`, to avoid surprise latency before a chat turn.
