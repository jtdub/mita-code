"""Tests for LanceDB vector store operations."""

from __future__ import annotations

from pathlib import Path

import pytest

from mita.index.store import CodeChunk, IndexStore, SearchResult


def _make_chunk(
    file_path: str = "test.py",
    start_line: int = 1,
    end_line: int = 5,
    content: str = "def hello(): pass",
    language: str = "python",
    symbol: str | None = "hello",
    embedding: list[float] | None = None,
) -> CodeChunk:
    return CodeChunk(
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        content=content,
        language=language,
        symbol=symbol,
        embedding=embedding,
    )


def _make_embedding(dim: int = 768, value: float = 0.1) -> list[float]:
    return [value] * dim


class TestCodeChunk:
    def test_basic_creation(self) -> None:
        chunk = _make_chunk()
        assert chunk.file_path == "test.py"
        assert chunk.language == "python"
        assert chunk.symbol == "hello"

    def test_embedding_excluded_from_dict(self) -> None:
        chunk = _make_chunk(embedding=[0.1, 0.2])
        d = chunk.model_dump()
        assert "embedding" not in d


class TestSearchResult:
    def test_creation(self) -> None:
        chunk = _make_chunk()
        result = SearchResult(chunk=chunk, score=0.95)
        assert result.score == 0.95
        assert result.chunk.file_path == "test.py"


class TestIndexStore:
    @pytest.mark.asyncio()
    async def test_create_and_search(self, tmp_path: Path) -> None:
        store = IndexStore(tmp_path / "index")
        emb = _make_embedding()
        chunks = [
            _make_chunk(file_path="a.py", symbol="func_a", embedding=emb),
            _make_chunk(file_path="b.py", symbol="func_b", embedding=_make_embedding(value=0.9)),
        ]
        await store.create_or_replace(chunks)
        assert store.exists()

        # Search with a vector close to the first chunk
        results = await store.search(emb, top_k=2)
        assert len(results) == 2
        assert isinstance(results[0], SearchResult)
        assert results[0].score > 0

    @pytest.mark.asyncio()
    async def test_exists_false_initially(self, tmp_path: Path) -> None:
        store = IndexStore(tmp_path / "index")
        assert not store.exists()

    @pytest.mark.asyncio()
    async def test_clear(self, tmp_path: Path) -> None:
        store = IndexStore(tmp_path / "index")
        chunks = [_make_chunk(embedding=_make_embedding())]
        await store.create_or_replace(chunks)
        assert store.exists()

        await store.clear()
        assert not store.exists()

    @pytest.mark.asyncio()
    async def test_status_empty(self, tmp_path: Path) -> None:
        store = IndexStore(tmp_path / "index")
        stats = await store.status()
        assert stats["exists"] is False
        assert stats["chunks"] == 0

    @pytest.mark.asyncio()
    async def test_status_with_data(self, tmp_path: Path) -> None:
        store = IndexStore(tmp_path / "index")
        chunks = [
            _make_chunk(file_path="a.py", embedding=_make_embedding()),
            _make_chunk(file_path="b.py", embedding=_make_embedding()),
        ]
        await store.create_or_replace(chunks)
        stats = await store.status()
        assert stats["exists"] is True
        assert stats["chunks"] == 2
        assert stats["files"] == 2

    @pytest.mark.asyncio()
    async def test_create_empty_noop(self, tmp_path: Path) -> None:
        store = IndexStore(tmp_path / "index")
        await store.create_or_replace([])
        assert not store.exists()

    @pytest.mark.asyncio()
    async def test_search_empty_store(self, tmp_path: Path) -> None:
        store = IndexStore(tmp_path / "index")
        results = await store.search(_make_embedding(), top_k=5)
        assert results == []

    @pytest.mark.asyncio()
    async def test_replace_overwrites(self, tmp_path: Path) -> None:
        store = IndexStore(tmp_path / "index")
        emb = _make_embedding()
        await store.create_or_replace([_make_chunk(file_path="old.py", embedding=emb)])
        stats1 = await store.status()
        assert stats1["chunks"] == 1

        await store.create_or_replace(
            [
                _make_chunk(file_path="new1.py", embedding=emb),
                _make_chunk(file_path="new2.py", embedding=emb),
            ]
        )
        stats2 = await store.status()
        assert stats2["chunks"] == 2


class TestIncremental:
    """Audit finding C6: incremental upsert/delete + content-hash tracking."""

    def _chunk(self, cid: str, chash: str, text: str) -> CodeChunk:
        from mita.index.store import CodeChunk

        return CodeChunk(
            file_path="a.py",
            start_line=1,
            end_line=1,
            content=text,
            language="python",
            chunk_id=cid,
            content_hash=chash,
            embedding=[0.1, 0.2, 0.3],
        )

    @pytest.mark.asyncio()
    async def test_load_content_hashes(self, tmp_path: Path) -> None:
        from mita.index.store import IndexStore

        store = IndexStore(tmp_path)
        await store.create_or_replace(
            [self._chunk("id1", "h1", "a"), self._chunk("id2", "h2", "b")], "nomic", 1.0
        )
        hashes = await store.load_content_hashes()
        assert hashes == {"id1": "h1", "id2": "h2"}

    @pytest.mark.asyncio()
    async def test_apply_incremental_upserts_and_deletes(self, tmp_path: Path) -> None:
        from mita.index.store import IndexStore

        store = IndexStore(tmp_path)
        await store.create_or_replace(
            [self._chunk("id1", "h1", "a"), self._chunk("id2", "h2", "b")], "nomic", 1.0
        )

        # id1 removed (not in present_ids); id2 updated; id3 added.
        changed = [self._chunk("id2", "h2b", "b2"), self._chunk("id3", "h3", "c")]
        await store.apply_incremental(changed, {"id2", "id3"}, "nomic", 2.0)

        hashes = await store.load_content_hashes()
        assert hashes == {"id2": "h2b", "id3": "h3"}

    @pytest.mark.asyncio()
    async def test_needs_full_rebuild_on_model_change(self, tmp_path: Path) -> None:
        from mita.index.store import IndexStore

        store = IndexStore(tmp_path)
        await store.create_or_replace([self._chunk("id1", "h1", "a")], "nomic-embed-text", 1.0)
        assert store.needs_full_rebuild("nomic-embed-text") is False
        assert store.needs_full_rebuild("bge-small") is True


class TestHybridSearch:
    """Audit finding C6: hybrid FTS + vector search with a relevance floor."""

    @pytest.mark.asyncio()
    async def test_hybrid_search_returns_results(self, tmp_path: Path) -> None:
        from mita.index.store import CodeChunk, IndexStore

        store = IndexStore(tmp_path)
        chunks = [
            CodeChunk(
                file_path="a.py",
                start_line=1,
                end_line=1,
                content="def calculate_total(items): return sum(items)",
                language="python",
                symbol="calculate_total",
                chunk_id="id1",
                content_hash="h1",
                embedding=[0.1, 0.2, 0.3],
            ),
            CodeChunk(
                file_path="b.py",
                start_line=1,
                end_line=1,
                content="def render_html(template): return template",
                language="python",
                symbol="render_html",
                chunk_id="id2",
                content_hash="h2",
                embedding=[0.9, 0.8, 0.7],
            ),
        ]
        await store.create_or_replace(chunks, "nomic", 1.0)

        results = await store.hybrid_search([0.1, 0.2, 0.3], "calculate total", top_k=5)
        assert results
        assert any(r.chunk.symbol == "calculate_total" for r in results)

    @pytest.mark.asyncio()
    async def test_relevance_floor_filters(self, tmp_path: Path) -> None:
        from mita.index.store import CodeChunk, IndexStore

        store = IndexStore(tmp_path)
        await store.create_or_replace(
            [
                CodeChunk(
                    file_path="a.py",
                    start_line=1,
                    end_line=1,
                    content="x = 1",
                    language="python",
                    chunk_id="id1",
                    content_hash="h1",
                    embedding=[0.1, 0.2, 0.3],
                )
            ],
            "nomic",
            1.0,
        )
        # An impossibly high floor drops everything.
        results = await store.hybrid_search([0.1, 0.2, 0.3], "x", top_k=5, floor=999.0)
        assert results == []


class TestStaleness:
    """Audit finding C6: detect a stale index against the working tree."""

    @pytest.mark.asyncio()
    async def test_fresh_index_not_stale(self, tmp_path: Path) -> None:
        import subprocess

        from mita.index.store import CodeChunk, IndexStore

        repo = tmp_path / "repo"
        repo.mkdir()
        subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
        (repo / "a.py").write_text("x = 1\n")
        subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)

        store = IndexStore(tmp_path / "index")
        import time as _t

        await store.create_or_replace(
            [
                CodeChunk(
                    file_path="a.py",
                    start_line=1,
                    end_line=1,
                    content="x = 1",
                    language="python",
                    chunk_id="id1",
                    content_hash="h1",
                    embedding=[0.1, 0.2, 0.3],
                )
            ],
            "nomic",
            _t.time() + 5,  # built "after" the file
        )
        assert store.is_stale(repo) is False

    @pytest.mark.asyncio()
    async def test_changed_file_makes_index_stale(self, tmp_path: Path) -> None:
        import subprocess
        import time as _t

        from mita.index.store import CodeChunk, IndexStore

        repo = tmp_path / "repo"
        repo.mkdir()
        subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
        (repo / "a.py").write_text("x = 1\n")

        store = IndexStore(tmp_path / "index")
        await store.create_or_replace(
            [
                CodeChunk(
                    file_path="a.py",
                    start_line=1,
                    end_line=1,
                    content="x = 1",
                    language="python",
                    chunk_id="id1",
                    content_hash="h1",
                    embedding=[0.1, 0.2, 0.3],
                )
            ],
            "nomic",
            _t.time() - 100,  # built in the past; the untracked file is newer
        )
        assert store.is_stale(repo) is True
