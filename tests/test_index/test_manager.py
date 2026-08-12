"""Tests for the index manager CLI handlers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mita.index.manager import build_index, clear_index, search_index, show_index_status


class TestBuildIndex:
    @pytest.mark.asyncio()
    async def test_incremental_only_embeds_changed(self, tmp_path: Path) -> None:
        """Audit finding C6: an incremental build re-embeds only new/changed chunks."""
        from mita.config.schema import MitaConfig
        from mita.index.store import CodeChunk

        config = MitaConfig()
        c1 = CodeChunk(
            file_path="a.py",
            start_line=1,
            end_line=2,
            content="def a(): pass",
            language="python",
            chunk_id="id1",
            content_hash="h1",
        )
        c2 = CodeChunk(
            file_path="a.py",
            start_line=3,
            end_line=4,
            content="def b(): pass",
            language="python",
            chunk_id="id2",
            content_hash="h2-new",
        )

        with (
            patch("mita.index.manager.load_config", return_value=config),
            patch("mita.index.get_index_dir", return_value=tmp_path / "index"),
            patch("mita.index.manager.IndexStore") as mock_store_cls,
            patch("mita.index.manager.EmbeddingClient") as mock_emb_cls,
            patch("mita.index.manager.parse_codebase", return_value=[c1, c2]),
            patch("mita.index.manager.console"),
        ):
            mock_store = MagicMock()
            mock_store.exists.return_value = True
            mock_store.needs_full_rebuild.return_value = False
            # c1 unchanged (h1 matches), c2 changed (stored hash differs).
            mock_store.load_content_hashes = AsyncMock(return_value={"id1": "h1", "id2": "h2-old"})
            mock_store.apply_incremental = AsyncMock()
            mock_store_cls.return_value = mock_store

            mock_emb = AsyncMock()
            mock_emb.is_model_available = AsyncMock(return_value=True)
            mock_emb.embed_texts = AsyncMock(return_value=[[0.1, 0.2]])
            mock_emb_cls.return_value = mock_emb

            await build_index(force=False)

            # Only the changed chunk (c2) is embedded.
            mock_emb.embed_texts.assert_awaited_once()
            assert mock_emb.embed_texts.call_args.args[0] == ["def b(): pass"]
            mock_store.apply_incremental.assert_awaited_once()
            # present_ids passed to apply_incremental covers BOTH current chunks.
            present_ids = mock_store.apply_incremental.call_args.args[1]
            assert present_ids == {"id1", "id2"}

    @pytest.mark.asyncio()
    async def test_model_not_available_no_callback(self) -> None:
        with (
            patch("mita.index.manager.load_config"),
            patch("mita.index.get_index_dir", return_value=Path("/tmp/index")),
            patch("mita.index.manager.IndexStore") as mock_store_cls,
            patch("mita.index.manager.EmbeddingClient") as mock_emb_cls,
            patch("mita.index.manager.console") as mock_console,
        ):
            mock_store = MagicMock()
            mock_store.exists.return_value = False
            mock_store_cls.return_value = mock_store
            mock_emb = AsyncMock()
            mock_emb.is_model_available = AsyncMock(return_value=False)
            mock_emb_cls.return_value = mock_emb

            await build_index(force=False)
            assert "not available" in str(mock_console.print.call_args)

    @pytest.mark.asyncio()
    async def test_model_not_available_with_callback_declines(self) -> None:
        with (
            patch("mita.index.manager.load_config"),
            patch("mita.index.get_index_dir", return_value=Path("/tmp/index")),
            patch("mita.index.manager.IndexStore") as mock_store_cls,
            patch("mita.index.manager.EmbeddingClient") as mock_emb_cls,
            patch("mita.index.manager.console"),
        ):
            mock_store = MagicMock()
            mock_store.exists.return_value = False
            mock_store_cls.return_value = mock_store
            mock_emb = AsyncMock()
            mock_emb.is_model_available = AsyncMock(return_value=False)
            mock_emb_cls.return_value = mock_emb

            async def decline_pull(config: object) -> bool:
                return False

            await build_index(force=False, pull_model_fn=decline_pull)

    @pytest.mark.asyncio()
    async def test_no_chunks_found(self) -> None:
        with (
            patch("mita.index.manager.load_config"),
            patch("mita.index.get_index_dir", return_value=Path("/tmp/index")),
            patch("mita.index.manager.IndexStore") as mock_store_cls,
            patch("mita.index.manager.EmbeddingClient") as mock_emb_cls,
            patch("mita.index.manager.parse_codebase", return_value=[]),
            patch("mita.index.manager.console") as mock_console,
        ):
            mock_store = MagicMock()
            mock_store.exists.return_value = False
            mock_store_cls.return_value = mock_store
            mock_emb = AsyncMock()
            mock_emb.is_model_available = AsyncMock(return_value=True)
            mock_emb_cls.return_value = mock_emb

            await build_index(force=False)
            assert "No code chunks" in str(mock_console.print.call_args)

    @pytest.mark.asyncio()
    async def test_embedding_failure(self) -> None:
        from mita.index.store import CodeChunk

        chunks = [
            CodeChunk(
                file_path="test.py",
                start_line=1,
                end_line=5,
                content="def hello(): pass",
                language="python",
            )
        ]
        with (
            patch("mita.index.manager.load_config"),
            patch("mita.index.get_index_dir", return_value=Path("/tmp/index")),
            patch("mita.index.manager.IndexStore") as mock_store_cls,
            patch("mita.index.manager.EmbeddingClient") as mock_emb_cls,
            patch("mita.index.manager.parse_codebase", return_value=chunks),
            patch("mita.index.manager.console") as mock_console,
        ):
            mock_store = MagicMock()
            mock_store.exists.return_value = False
            mock_store_cls.return_value = mock_store
            mock_emb = AsyncMock()
            mock_emb.is_model_available = AsyncMock(return_value=True)
            mock_emb.embed_texts = AsyncMock(side_effect=ConnectionError("failed"))
            mock_emb_cls.return_value = mock_emb

            await build_index(force=False)
            assert (
                "failed" in str(mock_console.print.call_args).lower()
                or "aborted" in str(mock_console.print.call_args).lower()
            )

    @pytest.mark.asyncio()
    async def test_successful_build(self) -> None:
        from mita.index.store import CodeChunk

        chunks = [
            CodeChunk(
                file_path="test.py",
                start_line=1,
                end_line=5,
                content="def hello(): pass",
                language="python",
            )
        ]
        with (
            patch("mita.index.manager.load_config"),
            patch("mita.index.get_index_dir", return_value=Path("/tmp/index")),
            patch("mita.index.manager.IndexStore") as mock_store_cls,
            patch("mita.index.manager.EmbeddingClient") as mock_emb_cls,
            patch("mita.index.manager.parse_codebase", return_value=chunks),
            patch("mita.index.manager.console") as mock_console,
        ):
            mock_store = MagicMock()
            mock_store.exists.return_value = False
            mock_store.create_or_replace = AsyncMock()
            mock_store_cls.return_value = mock_store
            mock_emb = AsyncMock()
            mock_emb.is_model_available = AsyncMock(return_value=True)
            mock_emb.embed_texts = AsyncMock(return_value=[[0.1] * 768])
            mock_emb_cls.return_value = mock_emb

            await build_index(force=True)
            mock_store.create_or_replace.assert_called_once()
            assert "Indexed" in str(mock_console.print.call_args)


class TestShowIndexStatus:
    @pytest.mark.asyncio()
    async def test_no_index(self) -> None:
        with (
            patch("mita.index.manager.load_config"),
            patch("mita.index.get_index_dir", return_value=Path("/tmp/index")),
            patch("mita.index.manager.IndexStore") as mock_store_cls,
            patch("mita.index.manager.console") as mock_console,
        ):
            mock_store = MagicMock()
            mock_store.status = AsyncMock(return_value={"exists": False, "chunks": 0, "files": 0})
            mock_store_cls.return_value = mock_store

            await show_index_status()
            assert "No index" in str(mock_console.print.call_args)

    @pytest.mark.asyncio()
    async def test_with_index(self) -> None:
        from mita.config.schema import MitaConfig

        config = MitaConfig()
        with (
            patch("mita.index.manager.load_config", return_value=config),
            patch("mita.index.get_index_dir", return_value=Path("/tmp/index")),
            patch("mita.index.manager.IndexStore") as mock_store_cls,
            patch("mita.index.manager.console") as mock_console,
        ):
            mock_store = MagicMock()
            mock_store.status = AsyncMock(return_value={"exists": True, "chunks": 100, "files": 10})
            mock_store_cls.return_value = mock_store

            await show_index_status()
            mock_console.print.assert_called()


class TestSearchIndex:
    @pytest.mark.asyncio()
    async def test_no_index(self) -> None:
        with (
            patch("mita.index.manager.load_config"),
            patch("mita.index.get_index_dir", return_value=Path("/tmp/index")),
            patch("mita.index.manager.Retriever") as mock_ret_cls,
            patch("mita.index.manager.console") as mock_console,
        ):
            mock_ret = MagicMock()
            mock_ret.is_available.return_value = False
            mock_ret_cls.return_value = mock_ret

            await search_index("test query")
            assert "No index" in str(mock_console.print.call_args)

    @pytest.mark.asyncio()
    async def test_no_results(self) -> None:
        with (
            patch("mita.index.manager.load_config"),
            patch("mita.index.get_index_dir", return_value=Path("/tmp/index")),
            patch("mita.index.manager.Retriever") as mock_ret_cls,
            patch("mita.index.manager.console") as mock_console,
        ):
            mock_ret = MagicMock()
            mock_ret.is_available.return_value = True
            mock_ret.retrieve = AsyncMock(return_value=[])
            mock_ret_cls.return_value = mock_ret

            await search_index("test query")
            assert "No results" in str(mock_console.print.call_args)

    @pytest.mark.asyncio()
    async def test_with_results(self) -> None:
        from mita.index.store import CodeChunk, SearchResult

        chunk = CodeChunk(
            file_path="test.py",
            start_line=1,
            end_line=5,
            content="def hello(): pass",
            language="python",
            symbol="hello",
        )
        results = [SearchResult(chunk=chunk, score=0.95)]

        with (
            patch("mita.index.manager.load_config"),
            patch("mita.index.get_index_dir", return_value=Path("/tmp/index")),
            patch("mita.index.manager.Retriever") as mock_ret_cls,
            patch("mita.index.manager.console") as mock_console,
        ):
            mock_ret = MagicMock()
            mock_ret.is_available.return_value = True
            mock_ret.retrieve = AsyncMock(return_value=results)
            mock_ret_cls.return_value = mock_ret

            await search_index("hello", top_k=5)
            mock_console.print.assert_called()


class TestClearIndex:
    @pytest.mark.asyncio()
    async def test_no_index(self) -> None:
        with (
            patch("mita.index.get_index_dir", return_value=Path("/tmp/index")),
            patch("mita.index.manager.IndexStore") as mock_store_cls,
            patch("mita.index.manager.console") as mock_console,
        ):
            mock_store = MagicMock()
            mock_store.exists.return_value = False
            mock_store_cls.return_value = mock_store

            await clear_index()
            assert "No index" in str(mock_console.print.call_args)

    @pytest.mark.asyncio()
    async def test_confirm_yes(self) -> None:
        with (
            patch("mita.index.get_index_dir", return_value=Path("/tmp/index")),
            patch("mita.index.manager.IndexStore") as mock_store_cls,
            patch("mita.index.manager.console") as mock_console,
        ):
            mock_store = MagicMock()
            mock_store.exists.return_value = True
            mock_store.clear = AsyncMock()
            mock_store_cls.return_value = mock_store
            mock_console.input.return_value = "y"

            await clear_index()
            mock_store.clear.assert_called_once()

    @pytest.mark.asyncio()
    async def test_confirm_no(self) -> None:
        with (
            patch("mita.index.get_index_dir", return_value=Path("/tmp/index")),
            patch("mita.index.manager.IndexStore") as mock_store_cls,
            patch("mita.index.manager.console") as mock_console,
        ):
            mock_store = MagicMock()
            mock_store.exists.return_value = True
            mock_store_cls.return_value = mock_store
            mock_console.input.return_value = "n"

            await clear_index()
            assert "Cancelled" in str(mock_console.print.call_args)
