"""Tests for retrieval improvements."""

import pytest
from unittest.mock import MagicMock, patch
from src.rag.retrieval import IntermediateRetriever


class TestRetrieveFullSection:
    """Tests for IntermediateRetriever.retrieve_full_section()."""

    def _make_mock_node(self, text, chunk_index, ticker="AAPL",
                        section="income_statement", fiscal_period="FY2023"):
        node = MagicMock()
        node.text = text
        node.score = 0.8
        node.node.node_id = f"node_{chunk_index}"
        node.node.metadata = {
            "ticker": ticker,
            "section": section,
            "fiscal_period": fiscal_period,
            "chunk_index": chunk_index,
        }
        node.metadata = node.node.metadata
        return node

    def test_returns_chunks_sorted_by_chunk_index(self):
        mock_index = MagicMock()
        retriever = IntermediateRetriever(mock_index)
        chunks = [
            self._make_mock_node("Chunk 2: Cost of sales...", chunk_index=2),
            self._make_mock_node("Chunk 0: Net sales...", chunk_index=0),
            self._make_mock_node("Chunk 1: Gross margin...", chunk_index=1),
        ]
        with patch.object(retriever, 'retrieve', return_value=chunks):
            result = retriever.retrieve_full_section(
                ticker="AAPL", section="income_statement", fiscal_period="FY2023",
            )
        indices = [n.metadata["chunk_index"] for n in result]
        assert indices == [0, 1, 2]

    def test_returns_empty_list_when_no_chunks_found(self):
        mock_index = MagicMock()
        retriever = IntermediateRetriever(mock_index)
        with patch.object(retriever, 'retrieve', return_value=[]):
            result = retriever.retrieve_full_section(
                ticker="TSLA", section="income_statement", fiscal_period="FY2025",
            )
        assert result == []

    def test_respects_max_chunks_parameter(self):
        mock_index = MagicMock()
        retriever = IntermediateRetriever(mock_index)
        chunks = [self._make_mock_node(f"Chunk {i}", chunk_index=i) for i in range(25)]
        with patch.object(retriever, 'retrieve', return_value=chunks):
            result = retriever.retrieve_full_section(
                ticker="AAPL", section="income_statement",
                fiscal_period="FY2023", max_chunks=10,
            )
        assert len(result) <= 10

    def test_filters_by_fiscal_period(self):
        mock_index = MagicMock()
        retriever = IntermediateRetriever(mock_index)
        chunks = [
            self._make_mock_node("FY2023 data", chunk_index=0, fiscal_period="FY2023"),
            self._make_mock_node("FY2022 data", chunk_index=1, fiscal_period="FY2022"),
            self._make_mock_node("FY2023 more", chunk_index=2, fiscal_period="FY2023"),
        ]
        with patch.object(retriever, 'retrieve', return_value=chunks):
            result = retriever.retrieve_full_section(
                ticker="AAPL", section="income_statement", fiscal_period="FY2023",
            )
        assert len(result) == 2
        assert all(n.metadata["fiscal_period"] == "FY2023" for n in result)
