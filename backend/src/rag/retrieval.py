"""Progressive RAG retrieval: Basic → Intermediate → Advanced."""

import json
import logging
from typing import Optional

import anthropic
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.vector_stores import (
    MetadataFilters,
    MetadataFilter,
    FilterOperator,
)

from config.settings import get_settings

logger = logging.getLogger(__name__)


class BasicRetriever:
    """Level 1: Simple vector similarity search."""

    def __init__(self, index):
        self.index = index

    def retrieve(self, query: str, top_k: int = 5,
                 ticker: str = None, sections: list[str] = None):
        retriever = VectorIndexRetriever(index=self.index, similarity_top_k=top_k)
        return retriever.retrieve(query)


class IntermediateRetriever:
    """Level 2: Metadata filtering + query rewriting."""

    EXPANSIONS = {
        "revenue": "revenue net sales total revenue",
        "debt": "total debt long-term debt term debt borrowings",
        "profit": "net income earnings profit net earnings",
        "cash": "cash and cash equivalents cash position liquidity",
        "margin": "gross margin operating margin profit margin",
        "growth": "growth year over year increase trend",
        "assets": "total assets current assets non-current assets",
        "liabilities": "total liabilities current liabilities",
        "equity": "shareholders equity stockholders equity total equity",
        "eps": "earnings per share diluted EPS basic EPS",
        "dividend": "dividend yield dividend per share payout",
        "buyback": "share repurchase buyback treasury stock",
    }

    def __init__(self, index):
        self.index = index

    def retrieve(self, query: str, top_k: int = 5,
                 ticker: str = None, sections: list[str] = None):
        enhanced_query = self._rewrite_query(query)

        # Try with full filters first, fall back to ticker-only, then no filters
        for filter_set in self._filter_fallback_chain(ticker, sections):
            try:
                kwargs = {"index": self.index, "similarity_top_k": top_k}
                if filter_set and filter_set.filters:
                    kwargs["filters"] = filter_set
                retriever = VectorIndexRetriever(**kwargs)
                return retriever.retrieve(enhanced_query)
            except Exception as e:
                logger.debug(f"Retrieval with filters failed, trying fallback: {e}")

        # Last resort: no filters at all
        retriever = VectorIndexRetriever(index=self.index, similarity_top_k=top_k)
        return retriever.retrieve(enhanced_query)

    def _filter_fallback_chain(self, ticker: str = None,
                               sections: list[str] = None) -> list:
        """Return a list of filter sets to try, from most specific to least."""
        chain = []

        # Most specific: ticker + sections
        if ticker and sections:
            chain.append(MetadataFilters(filters=[
                MetadataFilter(key="ticker", value=ticker.upper()),
                MetadataFilter(key="section", value=sections, operator=FilterOperator.IN),
            ]))

        # Ticker only
        if ticker:
            chain.append(MetadataFilters(filters=[
                MetadataFilter(key="ticker", value=ticker.upper()),
            ]))

        return chain

    def _rewrite_query(self, query: str) -> str:
        query_lower = query.lower()
        for term, expansion in self.EXPANSIONS.items():
            if term in query_lower:
                return f"{query} {expansion}"
        return query


class AdvancedRetriever:
    """Level 3: HyDE + reranking + multi-query retrieval."""

    def __init__(self, index):
        self.index = index
        settings = get_settings()
        self._claude = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self._model = settings.claude_model

    def retrieve(self, query: str, top_k: int = 5,
                 ticker: str = None, sections: list[str] = None):
        # Step 1: Generate multiple query variations
        queries = self._multi_query(query)
        queries.append(query)

        # Step 2: HyDE — hypothetical answer as another query
        hyde_text = self._hyde_transform(query)
        queries.append(hyde_text)

        # Step 3: Retrieve with metadata filters for each query
        intermediate = IntermediateRetriever(self.index)
        all_nodes = []
        for q in queries:
            try:
                nodes = intermediate.retrieve(q, top_k=5, ticker=ticker, sections=sections)
                all_nodes.extend(nodes)
            except Exception as e:
                logger.warning(f"Advanced retrieval sub-query failed: {e}")

        # Step 4: Deduplicate by node ID
        seen = set()
        unique_nodes = []
        for node in all_nodes:
            node_id = node.node.node_id
            if node_id not in seen:
                seen.add(node_id)
                unique_nodes.append(node)

        if not unique_nodes:
            return []

        # Step 5: Rerank using Claude
        return self._rerank(query, unique_nodes, top_n=top_k)

    def _hyde_transform(self, query: str) -> str:
        """Generate a hypothetical answer to improve embedding similarity."""
        try:
            response = self._claude.messages.create(
                model=self._model,
                max_tokens=200,
                temperature=0.0,
                messages=[{"role": "user", "content": (
                    "Write a brief, factual paragraph that would answer this financial question. "
                    "Use realistic but hypothetical numbers.\n"
                    f"Question: {query}\nAnswer:"
                )}],
            )
            return response.content[0].text
        except Exception as e:
            logger.warning(f"HyDE transform failed: {e}")
            return query

    def _multi_query(self, query: str) -> list[str]:
        """Generate multiple query variations for broader retrieval."""
        try:
            response = self._claude.messages.create(
                model=self._model,
                max_tokens=200,
                temperature=0.3,
                messages=[{"role": "user", "content": (
                    "Generate 3 different ways to search for the answer to this financial question. "
                    "Each should emphasize different keywords.\n"
                    f"Question: {query}\n"
                    "Return as a JSON array of 3 strings."
                )}],
            )
            text = response.content[0].text.strip()
            if "```" in text:
                text = text.split("```")[1].split("```")[0]
                if text.startswith("json"):
                    text = text[4:]
            return json.loads(text)
        except Exception as e:
            logger.warning(f"Multi-query generation failed: {e}")
            return []

    def _rerank(self, query: str, nodes, top_n: int = 5):
        """Rerank retrieved nodes using Claude as a cross-encoder."""
        if len(nodes) <= top_n:
            return nodes

        chunks_text = "\n---\n".join(
            f"[{i}] {node.text[:500]}" for i, node in enumerate(nodes)
        )

        try:
            response = self._claude.messages.create(
                model=self._model,
                max_tokens=100,
                temperature=0.0,
                messages=[{"role": "user", "content": (
                    f'Given this query: "{query}"\n'
                    "Rank these chunks by relevance (most relevant first). "
                    "Return ONLY the indices as a JSON array.\n"
                    f"{chunks_text}"
                )}],
            )
            text = response.content[0].text.strip()
            if "```" in text:
                text = text.split("```")[1].split("```")[0]
                if text.startswith("json"):
                    text = text[4:]
            ranked_indices = json.loads(text)
            return [nodes[i] for i in ranked_indices[:top_n] if i < len(nodes)]
        except Exception as e:
            logger.warning(f"Reranking failed, returning top-{top_n} by score: {e}")
            return sorted(nodes, key=lambda n: n.score or 0, reverse=True)[:top_n]
