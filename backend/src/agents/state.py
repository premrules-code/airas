"""LangGraph state definition for analysis pipeline."""

from typing import TypedDict, Annotated, Optional
import operator
from src.models.structured_outputs import AgentOutput, InvestmentRecommendation


class AnalysisState(TypedDict):
    """State that flows through the LangGraph analysis pipeline."""
    ticker: str
    query: Optional[str]                                       # Natural language query (None = full analysis)
    active_agents: list[str]                                   # Agent names selected by router
    mode: str                                                  # "full" | "focused"
    rag_level: str                                             # "basic" | "intermediate" | "advanced"
    rag_context: dict[str, str]                                # agent_name → RAG context string
    agent_outputs: Annotated[list[AgentOutput], operator.add]  # Reducer: auto-merges parallel outputs
    recommendation: Optional[InvestmentRecommendation]         # Final synthesis result
    trace_id: Optional[str]                                    # Langfuse trace ID
    errors: Annotated[list[str], operator.add]                 # Reducer: collects errors from all nodes
