"""Tests for ContextBuilder retrieval cache."""

import pytest
from unittest.mock import MagicMock, patch, call
from src.agents.context import ContextBuilder


def _make_mock_agent(name, queries, sections=None):
    agent = MagicMock()
    agent.AGENT_NAME = name
    agent.RAG_QUERIES = queries
    agent.SECTIONS = sections or []
    return agent


class TestContextBuilderCache:

    def test_duplicate_queries_hit_cache(self):
        mock_rag = MagicMock()
        mock_rag.index = MagicMock()
        builder = ContextBuilder(mock_rag, level="intermediate")
        agent_a = _make_mock_agent("financial_analyst", [
            "Revenue growth trends for {ticker}",
        ], sections=["income_statement"])
        agent_b = _make_mock_agent("earnings_analysis", [
            "Revenue growth trends for {ticker}",
        ], sections=["income_statement"])
        with patch.object(builder, '_intermediate_query', return_value="context data") as mock_query:
            builder.build_all_contexts("AAPL", [agent_a, agent_b])
            assert mock_query.call_count == 1

    def test_different_queries_miss_cache(self):
        mock_rag = MagicMock()
        mock_rag.index = MagicMock()
        builder = ContextBuilder(mock_rag, level="intermediate")
        agent = _make_mock_agent("financial_analyst", [
            "Revenue growth trends for {ticker}",
            "Debt and liquidity for {ticker}",
        ], sections=["income_statement"])
        with patch.object(builder, '_intermediate_query', return_value="data") as mock_query:
            builder.build_all_contexts("AAPL", [agent])
            assert mock_query.call_count == 2

    def test_same_query_different_sections_miss_cache(self):
        mock_rag = MagicMock()
        mock_rag.index = MagicMock()
        builder = ContextBuilder(mock_rag, level="intermediate")
        agent_a = _make_mock_agent("financial_analyst", [
            "Growth trends for {ticker}",
        ], sections=["income_statement"])
        agent_b = _make_mock_agent("risk_assessment", [
            "Growth trends for {ticker}",
        ], sections=["risk_factors"])
        with patch.object(builder, '_intermediate_query', return_value="data") as mock_query:
            builder.build_all_contexts("AAPL", [agent_a, agent_b])
            assert mock_query.call_count == 2

    def test_cache_does_not_persist_across_calls(self):
        mock_rag = MagicMock()
        mock_rag.index = MagicMock()
        builder = ContextBuilder(mock_rag, level="intermediate")
        agent = _make_mock_agent("financial_analyst", [
            "Revenue for {ticker}",
        ], sections=["income_statement"])
        with patch.object(builder, '_intermediate_query', return_value="data") as mock_query:
            builder.build_all_contexts("AAPL", [agent])
            builder.build_all_contexts("AAPL", [agent])
            assert mock_query.call_count == 2

    def test_cache_works_with_corrective_level(self):
        mock_rag = MagicMock()
        mock_rag.index = MagicMock()
        builder = ContextBuilder(mock_rag, level="corrective")
        agent_a = _make_mock_agent("financial_analyst", [
            "Revenue for {ticker}",
        ], sections=["income_statement"])
        agent_b = _make_mock_agent("earnings_analysis", [
            "Revenue for {ticker}",
        ], sections=["income_statement"])
        with patch.object(builder, '_corrective_query', return_value="corrective data") as mock_query:
            builder.build_all_contexts("AAPL", [agent_a, agent_b])
            assert mock_query.call_count == 1

    def test_agents_without_rag_queries_skip_cache(self):
        mock_rag = MagicMock()
        mock_rag.index = MagicMock()
        builder = ContextBuilder(mock_rag, level="intermediate")
        agent_no_rag = _make_mock_agent("technical_analyst", [], sections=[])
        agent_with_rag = _make_mock_agent("financial_analyst", [
            "Revenue for {ticker}",
        ], sections=["income_statement"])
        with patch.object(builder, '_intermediate_query', return_value="data") as mock_query:
            contexts = builder.build_all_contexts("AAPL", [agent_no_rag, agent_with_rag])
            assert contexts["technical_analyst"] == ""
            assert mock_query.call_count == 1
