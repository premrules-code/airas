"""Progressive RAG retrieval: Basic → Intermediate → Advanced → Corrective."""

import json
import logging
from dataclasses import dataclass, field
from typing import Optional, List

import anthropic
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.vector_stores import (
    MetadataFilters,
    MetadataFilter,
    FilterOperator,
)

from config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class CorrectiveResult:
    """Result from CorrectiveRetriever with metadata about corrections."""

    nodes: List  # Retrieved nodes after correction
    grades: List  # GradeResult for each node
    corrections_applied: List[str] = field(default_factory=list)
    initial_relevant_ratio: float = 0.0
    final_relevant_ratio: float = 0.0
    num_correction_attempts: int = 0

    @property
    def confidence(self) -> float:
        """Overall retrieval confidence based on relevance ratio."""
        return self.final_relevant_ratio

    @property
    def improved(self) -> bool:
        """Whether corrections improved relevance."""
        return self.final_relevant_ratio > self.initial_relevant_ratio


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
        extras = []
        for term, expansion in self.EXPANSIONS.items():
            if term in query_lower:
                extras.append(expansion)
        if extras:
            return f"{query} {' '.join(extras)}"
        return query

    def retrieve_full_section(
        self,
        ticker: str,
        section: str,
        fiscal_period: str,
        max_chunks: int = 20,
    ) -> list:
        """Retrieve ALL chunks for a specific filing section, sorted by chunk_index.

        Unlike retrieve() which returns top-k by similarity, this fetches every chunk
        matching the (ticker, section, fiscal_period) triple and returns them in
        document order.

        Args:
            ticker: Stock ticker (e.g., "AAPL")
            section: SEC filing section (e.g., "income_statement", "balance_sheet")
            fiscal_period: Fiscal period (e.g., "FY2023", "Q1 2024")
            max_chunks: Maximum chunks to return (default 20, safety cap)

        Returns:
            List of nodes sorted by chunk_index, or empty list if section not found.
        """
        nodes = self.retrieve(
            query="",
            top_k=max_chunks,
            ticker=ticker,
            sections=[section],
        )

        if not nodes:
            logger.info(f"No chunks found for {ticker}/{section}/{fiscal_period}")
            return []

        # Filter to exact fiscal_period (retrieve() only filters by ticker+section)
        if fiscal_period:
            nodes = [
                n for n in nodes
                if n.metadata.get("fiscal_period") == fiscal_period
            ]

        # Sort by chunk_index for document order
        nodes.sort(key=lambda n: n.metadata.get("chunk_index", 0))

        logger.info(
            f"Full section retrieval: {ticker}/{section}/{fiscal_period} -> {len(nodes)} chunks"
        )

        return nodes[:max_chunks]


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


class CorrectiveRetriever:
    """Level 4: Corrective RAG with self-healing retrieval.

    Adds relevance grading after retrieval. If documents are mostly
    irrelevant, transforms the query and re-retrieves.

    Flow:
    1. Initial retrieval (IntermediateRetriever)
    2. Check vector scores - skip grading if high confidence (hybrid mode)
    3. Grade each document for relevance (RelevanceGrader)
    4. If <threshold relevant: transform query and re-retrieve
    5. Repeat up to max_corrections times
    6. Return graded, filtered results

    Usage:
        retriever = CorrectiveRetriever(index)
        result = retriever.retrieve(
            query="What are Apple's risk factors?",
            top_k=5,
            ticker="AAPL",
        )
        print(f"Confidence: {result.confidence:.0%}")
        print(f"Corrections: {result.corrections_applied}")
        for node in result.nodes:
            print(node.text[:100])
    """

    # Threshold for accepting retrieval (ratio of relevant docs)
    DEFAULT_THRESHOLD = 0.4  # Lowered from 0.5 to reduce unnecessary corrections

    # Vector similarity threshold for skipping grading (hybrid mode)
    # If top doc has score >= this AND avg >= 0.70, assume they're relevant
    HIGH_CONFIDENCE_THRESHOLD = 0.72

    def __init__(
        self,
        index,
        relevance_threshold: float = DEFAULT_THRESHOLD,
        max_corrections: int = 2,
        enable_web_search: bool = False,
        skip_grading_threshold: float = HIGH_CONFIDENCE_THRESHOLD,
    ):
        """Initialize corrective retriever.

        Args:
            index: LlamaIndex VectorStoreIndex
            relevance_threshold: Ratio of relevant docs to accept (0-1)
            max_corrections: Maximum correction attempts
            enable_web_search: Whether to use web search fallback
            skip_grading_threshold: Min vector score to skip LLM grading (hybrid mode)
        """
        self.index = index
        self.threshold = relevance_threshold
        self.max_corrections = max_corrections
        self.enable_web_search = enable_web_search
        self.skip_grading_threshold = skip_grading_threshold

        # Lazy load correction components
        self._grader = None
        self._transformer = None
        self._web_search = None

    @property
    def grader(self):
        """Lazy load RelevanceGrader."""
        if self._grader is None:
            from src.rag.relevance_grader import RelevanceGrader
            self._grader = RelevanceGrader()
        return self._grader

    @property
    def transformer(self):
        """Lazy load QueryTransformer."""
        if self._transformer is None:
            from src.rag.corrections import QueryTransformer
            self._transformer = QueryTransformer()
        return self._transformer

    @property
    def web_search(self):
        """Lazy load WebSearchFallback."""
        if self._web_search is None and self.enable_web_search:
            from src.rag.corrections import WebSearchFallback
            self._web_search = WebSearchFallback()
        return self._web_search

    def _check_high_confidence(self, nodes) -> bool:
        """Check if retrieval results are high confidence based on vector scores.

        Returns True if we should skip LLM grading (hybrid mode).
        Uses a practical threshold based on real embedding score distributions.
        """
        if not nodes:
            return False

        # Get scores from nodes
        scores = [node.score for node in nodes if node.score is not None]
        if not scores:
            return False

        top_score = max(scores)
        avg_score = sum(scores) / len(scores)

        # High confidence if:
        # - Top result is very relevant (>= threshold) AND
        # - Average is reasonably high (>= 0.70)
        # This allows us to trust strong top results while ensuring
        # the rest aren't completely off-topic
        is_high_conf = top_score >= self.skip_grading_threshold and avg_score >= 0.70

        if is_high_conf:
            logger.info(
                f"High confidence retrieval (top={top_score:.3f}, avg={avg_score:.3f}), "
                f"skipping LLM grading"
            )

        return is_high_conf

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        ticker: str = None,
        sections: list[str] = None,
    ) -> CorrectiveResult:
        """Retrieve with self-correction loop.

        Args:
            query: Search query
            top_k: Number of results to return
            ticker: Stock ticker for filtering
            sections: SEC sections to filter

        Returns:
            CorrectiveResult with nodes, grades, and correction metadata
        """
        base_retriever = IntermediateRetriever(self.index)

        # Step 1: Initial retrieval
        nodes = base_retriever.retrieve(query, top_k=top_k, ticker=ticker, sections=sections)

        if not nodes:
            logger.warning("Initial retrieval returned no results")
            return CorrectiveResult(
                nodes=[],
                grades=[],
                initial_relevant_ratio=0.0,
                final_relevant_ratio=0.0,
            )

        # Step 2: Hybrid check - skip grading if vector scores are high
        if self._check_high_confidence(nodes):
            # Create synthetic "relevant" grades for high-confidence results
            from src.rag.relevance_grader import GradeResult
            synthetic_grades = [
                GradeResult(
                    grade="relevant",
                    confidence=node.score or 0.9,
                    reason="High vector similarity (skipped LLM grading)",
                    doc_index=i,
                )
                for i, node in enumerate(nodes)
            ]
            return CorrectiveResult(
                nodes=nodes,
                grades=synthetic_grades,
                corrections_applied=["hybrid_skip"],
                initial_relevant_ratio=1.0,
                final_relevant_ratio=1.0,
                num_correction_attempts=0,
            )

        # Step 3: Grade documents with LLM
        doc_texts = [node.text for node in nodes]
        grades = self.grader.grade_batch(query, doc_texts)

        # Calculate relevance ratio
        relevant_count = sum(1 for g in grades if g.grade == "relevant")
        partial_count = sum(1 for g in grades if g.grade == "partial")
        # Count relevant and partial as "usable"
        usable_ratio = (relevant_count + partial_count * 0.5) / len(grades)
        initial_ratio = usable_ratio

        logger.info(
            f"Initial retrieval: {relevant_count} relevant, {partial_count} partial, "
            f"{len(grades) - relevant_count - partial_count} irrelevant "
            f"(ratio: {usable_ratio:.0%})"
        )

        # Step 3: Correction loop if below threshold
        corrections_applied = []
        current_query = query
        best_nodes = nodes
        best_grades = grades
        best_ratio = usable_ratio

        for attempt in range(self.max_corrections):
            if usable_ratio >= self.threshold:
                break

            logger.info(f"Correction attempt {attempt + 1}: ratio {usable_ratio:.0%} < {self.threshold:.0%}")

            # Transform query
            failed_docs = [
                nodes[i].text for i, g in enumerate(grades)
                if g.grade == "irrelevant"
            ]
            transform_result = self.transformer.transform(
                current_query, failed_docs, ticker
            )

            if transform_result.transformed_query == current_query:
                logger.info("Query transformation produced no change, stopping")
                break

            current_query = transform_result.transformed_query
            corrections_applied.append(f"{transform_result.strategy}: {current_query[:50]}...")

            logger.info(f"Transformed query: {current_query[:100]}")

            # Re-retrieve
            new_nodes = base_retriever.retrieve(
                current_query, top_k=top_k, ticker=ticker, sections=sections
            )

            if not new_nodes:
                logger.warning("Re-retrieval returned no results")
                continue

            # Re-grade
            new_doc_texts = [node.text for node in new_nodes]
            new_grades = self.grader.grade_batch(query, new_doc_texts)  # Grade against ORIGINAL query

            new_relevant = sum(1 for g in new_grades if g.grade == "relevant")
            new_partial = sum(1 for g in new_grades if g.grade == "partial")
            new_ratio = (new_relevant + new_partial * 0.5) / len(new_grades)

            logger.info(f"After correction: {new_relevant} relevant, {new_partial} partial (ratio: {new_ratio:.0%})")

            # Keep best results
            if new_ratio > best_ratio:
                best_nodes = new_nodes
                best_grades = new_grades
                best_ratio = new_ratio
                usable_ratio = new_ratio
                nodes = new_nodes
                grades = new_grades

        # Step 4: Optional web search fallback
        if best_ratio < self.threshold and self.web_search and self.web_search.enabled:
            logger.info("Trying web search fallback")
            web_results = self.web_search.search(query, ticker)
            if web_results:
                corrections_applied.append(f"web_search: {len(web_results)} results")
                # Note: Web results would need to be converted to nodes
                # For now, just log that we tried

        # Step 5: Filter to relevant nodes only (optional - can return all with grades)
        # For now, return all nodes with their grades so caller can decide

        return CorrectiveResult(
            nodes=best_nodes,
            grades=best_grades,
            corrections_applied=corrections_applied,
            initial_relevant_ratio=initial_ratio,
            final_relevant_ratio=best_ratio,
            num_correction_attempts=len(corrections_applied),
        )
