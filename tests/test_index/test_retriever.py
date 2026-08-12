"""Tests for RAG retrieval pipeline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from mita.config.schema import MitaConfig
from mita.index.retriever import Retriever, format_results
from mita.index.store import CodeChunk, SearchResult


def _make_result(file_path: str = "test.py", symbol: str = "func") -> SearchResult:
    return SearchResult(
        chunk=CodeChunk(
            file_path=file_path,
            start_line=1,
            end_line=5,
            content="def func():\n    pass",
            language="python",
            symbol=symbol,
        ),
        score=0.95,
    )


class TestFormatResults:
    def test_basic_format(self) -> None:
        results = [_make_result()]
        text = format_results(results)
        assert "test.py:1-5" in text
        assert "func" in text
        assert "```python" in text

    def test_multiple_results(self) -> None:
        results = [_make_result("a.py", "foo"), _make_result("b.py", "bar")]
        text = format_results(results)
        assert "a.py" in text
        assert "b.py" in text

    def test_no_symbol(self) -> None:
        result = SearchResult(
            chunk=CodeChunk(
                file_path="test.py",
                start_line=1,
                end_line=2,
                content="x = 1",
                language="python",
            ),
            score=0.5,
        )
        text = format_results([result])
        assert "test.py:1-2" in text


class TestRetriever:
    @pytest.mark.asyncio()
    async def test_retrieve(self, tmp_path: Path) -> None:
        config = MitaConfig()
        retriever = Retriever(config, index_dir=tmp_path)

        mock_results = [_make_result()]
        with (
            patch.object(retriever._store, "exists", return_value=True),
            patch.object(
                retriever._store, "hybrid_search", new_callable=AsyncMock, return_value=mock_results
            ),
            patch.object(
                retriever._embedder,
                "embed_single",
                new_callable=AsyncMock,
                return_value=[0.1] * 768,
            ),
        ):
            results = await retriever.retrieve("test query")
            assert len(results) == 1
            assert results[0].chunk.file_path == "test.py"

    @pytest.mark.asyncio()
    async def test_retrieve_formatted(self, tmp_path: Path) -> None:
        config = MitaConfig()
        retriever = Retriever(config, index_dir=tmp_path)

        mock_results = [_make_result()]
        with (
            patch.object(retriever._store, "exists", return_value=True),
            patch.object(
                retriever._store, "hybrid_search", new_callable=AsyncMock, return_value=mock_results
            ),
            patch.object(
                retriever._embedder,
                "embed_single",
                new_callable=AsyncMock,
                return_value=[0.1] * 768,
            ),
        ):
            text = await retriever.retrieve_formatted("test query")
            assert "test.py" in text
            assert "```python" in text

    @pytest.mark.asyncio()
    async def test_retrieve_empty(self, tmp_path: Path) -> None:
        config = MitaConfig()
        retriever = Retriever(config, index_dir=tmp_path)

        with (
            patch.object(retriever._store, "exists", return_value=True),
            patch.object(
                retriever._store, "hybrid_search", new_callable=AsyncMock, return_value=[]
            ),
            patch.object(
                retriever._embedder,
                "embed_single",
                new_callable=AsyncMock,
                return_value=[0.1] * 768,
            ),
        ):
            text = await retriever.retrieve_formatted("test query")
            assert text == ""

    def test_is_available(self, tmp_path: Path) -> None:
        config = MitaConfig()
        retriever = Retriever(config, index_dir=tmp_path)

        with patch.object(retriever._store, "exists", return_value=False):
            assert not retriever.is_available()

        with patch.object(retriever._store, "exists", return_value=True):
            assert retriever.is_available()
