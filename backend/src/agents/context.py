"""Context builder — gathers RAG results and structures them for agents."""

import logging
from typing import Optional
from src.rag.supabase_rag import SupabaseRAG
from src.rag.retrieval import BasicRetriever, IntermediateRetriever, AdvancedRetriever
from src.agents.tracing import TracingManager

logger = logging.getLogger(__name__)


class ContextBuilder:
    """Gathers RAG context and structures it for each agent."""

    def __init__(self, rag: SupabaseRAG, level: str = "intermediate"):
        self.rag = rag
        self.level = level

    def build_all_contexts(
        self, ticker: str, agents: list, tracer: Optional[TracingManager] = None
    ) -> dict[str, str]:
        """Run all RAG queries for all active agents, return {agent_name: context_string}."""
        context_span = tracer.span_context_gathering() if tracer else None
        contexts = {}

        for agent_cls in agents:
            if not agent_cls.RAG_QUERIES:
                contexts[agent_cls.AGENT_NAME] = ""
                continue

            sections = []
            for i, query_template in enumerate(agent_cls.RAG_QUERIES):
                query = query_template.format(ticker=ticker)

                try:
                    if self.level == "basic":
                        result = self._basic_query(query)
                    elif self.level == "intermediate":
                        result = self._intermediate_query(
                            query, ticker, agent_cls.SECTIONS
                        )
                    else:
                        result = self._advanced_query(
                            query, ticker, agent_cls.SECTIONS
                        )
                except Exception as e:
                    logger.warning(f"RAG query failed for {agent_cls.AGENT_NAME}: {e}")
                    result = "No relevant data found."

                sections.append(f'[Query: "{query}"]\n{result}')

                if tracer and context_span:
                    tracer.log_rag_query(
                        context_span, agent_cls.AGENT_NAME, query, str(result), i
                    )

            contexts[agent_cls.AGENT_NAME] = "\n\n".join(sections)

        if context_span:
            context_span.end()

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
