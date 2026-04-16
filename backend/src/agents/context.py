"""Context builder — gathers RAG results and structures them for agents."""

import logging
from typing import Optional
from src.rag.supabase_rag import SupabaseRAG
from src.rag.retrieval import (
    BasicRetriever,
    IntermediateRetriever,
    AdvancedRetriever,
    CorrectiveRetriever,
)
from src.agents.tracing import TracingManager
from src.utils.galileo_setup import log_rag_query
from src.guardrails.galileo_guardrails import check_context_relevance

logger = logging.getLogger(__name__)


class ContextBuilder:
    """Gathers RAG context and structures it for each agent."""

    def __init__(self, rag: SupabaseRAG, level: str = "intermediate"):
        self.rag = rag
        self.level = level

    def build_all_contexts(
        self, ticker: str, agents: list, tracer: Optional[TracingManager] = None
    ) -> dict[str, str]:
        """Run all RAG queries for all active agents, return {agent_name: context_string}.

        Uses a per-call cache to avoid duplicate pgvector queries when multiple
        agents share the same RAG query + ticker + sections combination.
        """
        context_span = tracer.span_context_gathering() if tracer else None
        contexts = {}
        cache: dict[tuple, str] = {}

        for agent_cls in agents:
            if not agent_cls.RAG_QUERIES:
                contexts[agent_cls.AGENT_NAME] = ""
                continue

            sections = []
            for i, query_template in enumerate(agent_cls.RAG_QUERIES):
                query = query_template.format(ticker=ticker)
                cache_key = (
                    query,
                    ticker,
                    tuple(sorted(agent_cls.SECTIONS)) if agent_cls.SECTIONS else (),
                )

                if cache_key in cache:
                    result = cache[cache_key]
                    logger.debug(
                        f"Cache hit for {agent_cls.AGENT_NAME} query: {query[:50]}..."
                    )
                else:
                    try:
                        if self.level == "basic":
                            result = self._basic_query(query)
                        elif self.level == "intermediate":
                            result = self._intermediate_query(
                                query, ticker, agent_cls.SECTIONS
                            )
                        elif self.level == "corrective":
                            result = self._corrective_query(
                                query, ticker, agent_cls.SECTIONS
                            )
                        else:  # advanced
                            result = self._advanced_query(
                                query, ticker, agent_cls.SECTIONS
                            )
                    except Exception as e:
                        logger.warning(f"RAG query failed for {agent_cls.AGENT_NAME}: {e}")
                        result = "No relevant data found."

                    cache[cache_key] = result

                sections.append(f'[Query: "{query}"]\n{result}')

                # Log to Galileo
                result_chunks = result.split("\n\n") if result else []
                log_rag_query(
                    query=query,
                    retrieved_chunks=result_chunks[:10],
                    response=result[:2000],
                    metadata={"agent": agent_cls.AGENT_NAME, "ticker": ticker},
                )

                if tracer and context_span:
                    tracer.log_rag_query(
                        context_span, agent_cls.AGENT_NAME, query, str(result), i
                    )

            contexts[agent_cls.AGENT_NAME] = "\n\n".join(sections)

        if context_span:
            context_span.end()

        logger.info(
            f"Context building complete: {len(cache)} unique queries "
            f"(served {sum(len(a.RAG_QUERIES) for a in agents if a.RAG_QUERIES)} total requests)"
        )

        return contexts

    def _basic_query(self, query: str) -> str:
        """Basic RAG: simple vector similarity search via LlamaIndex query engine."""
        return self.rag.query(query)

    def _intermediate_query(
        self, query: str, ticker: str, sections: Optional[list]
    ) -> str:
        """Intermediate RAG: metadata filtering + query rewriting."""
        if self.rag.index is None:
            return self._basic_query(query)
        retriever = IntermediateRetriever(self.rag.index)
        nodes = retriever.retrieve(query, top_k=5, ticker=ticker, sections=sections)
        if nodes:
            return "\n".join(node.text for node in nodes)
        return "No relevant data found."

    def _advanced_query(
        self, query: str, ticker: str, sections: Optional[list]
    ) -> str:
        """Advanced RAG: HyDE + reranking + multi-query."""
        if self.rag.index is None:
            return self._basic_query(query)
        retriever = AdvancedRetriever(self.rag.index)
        nodes = retriever.retrieve(query, top_k=5, ticker=ticker, sections=sections)
        if nodes:
            return "\n".join(node.text for node in nodes)
        return "No relevant data found."

    def _corrective_query(
        self, query: str, ticker: str, sections: Optional[list]
    ) -> str:
        """Corrective RAG: grading + self-correction loop.

        Returns context with confidence metadata so agents know retrieval quality.
        """
        if self.rag.index is None:
            return self._basic_query(query)

        retriever = CorrectiveRetriever(
            self.rag.index,
            relevance_threshold=0.5,
            max_corrections=2,
        )

        result = retriever.retrieve(
            query, top_k=5, ticker=ticker, sections=sections
        )

        if not result.nodes:
            return "No relevant data found."

        # Build context with confidence metadata
        lines = []

        # Add confidence header so agent knows retrieval quality
        lines.append(f"[Retrieval confidence: {result.confidence:.0%}]")

        if result.corrections_applied:
            lines.append(f"[Corrections applied: {len(result.corrections_applied)}]")

        # Add document content, marking relevance
        for i, node in enumerate(result.nodes):
            grade = result.grades[i] if i < len(result.grades) else None
            if grade:
                grade_marker = f"[{grade.grade.upper()}]"
            else:
                grade_marker = ""
            lines.append(f"{grade_marker}\n{node.text}")

        return "\n\n".join(lines)
