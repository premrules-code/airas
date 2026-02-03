"""LangGraph StateGraph for analysis pipeline."""

import logging
import time
from langgraph.graph import StateGraph, END
from langgraph.types import Send

from src.agents.state import AnalysisState
from src.agents.router import route_query, ALL_AGENT_NAMES
from src.agents.context import ContextBuilder
from src.agents.tracing import TracingManager
from src.agents import ALL_AGENTS, AGENT_MAP
from src.rag.supabase_rag import SupabaseRAG

logger = logging.getLogger(__name__)


def route_query_node(state: AnalysisState) -> dict:
    """Route the query to determine which agents to run."""
    if state.get("query"):
        active = route_query(state["query"])
        mode = "focused" if len(active) < len(ALL_AGENT_NAMES) else "full"
    else:
        active = ALL_AGENT_NAMES
        mode = "full"

    # Start Langfuse trace
    tracer = TracingManager()
    trace_id = tracer.start_trace(state["ticker"])

    logger.info(f"Router: mode={mode}, agents={active}")
    return {"active_agents": active, "mode": mode, "trace_id": trace_id}


def gather_context_node(state: AnalysisState) -> dict:
    """Gather RAG context for all active agents."""
    rag = SupabaseRAG()
    rag.load_index()
    tracer = TracingManager.from_trace_id(state.get("trace_id"), ticker=state.get("ticker"))

    active_agent_classes = [
        AGENT_MAP[name] for name in state["active_agents"] if name in AGENT_MAP
    ]

    # Only build context for agents that have RAG queries
    rag_agents = [a for a in active_agent_classes if a.RAG_QUERIES]
    if not rag_agents:
        logger.info("No RAG agents active, skipping context gathering")
        return {"rag_context": {}}

    builder = ContextBuilder(rag, level=state.get("rag_level", "intermediate"))
    contexts = builder.build_all_contexts(state["ticker"], active_agent_classes, tracer)

    return {"rag_context": contexts}


def fan_out_to_agents(state: AnalysisState) -> list[Send]:
    """Fan out to active agents in parallel via LangGraph Send API."""
    return [
        Send(name, state)
        for name in state["active_agents"]
        if name in AGENT_MAP
    ]


def make_agent_node(agent_cls, index: int = 0):
    """Create a LangGraph node function for an agent class."""

    def node_fn(state: AnalysisState) -> dict:
        # Stagger agent starts to avoid rate limit bursts (5s between each)
        if index > 0:
            time.sleep(index * 5)

        agent = agent_cls()
        tracer = TracingManager.from_trace_id(state.get("trace_id"), ticker=state.get("ticker"))
        rag_context = state.get("rag_context", {}).get(agent.AGENT_NAME, "")

        try:
            output = agent.analyze(state["ticker"], rag_context, tracer)
            return {"agent_outputs": [output]}
        except Exception as e:
            logger.error(f"Agent {agent.AGENT_NAME} failed: {e}")
            return {"errors": [f"{agent.AGENT_NAME}: {e}"]}

    return node_fn


def synthesis_node(state: AnalysisState) -> dict:
    """Combine agent outputs into final recommendation."""
    from src.agents.synthesis import synthesize

    tracer = TracingManager.from_trace_id(state.get("trace_id"), ticker=state.get("ticker"))
    recommendation = synthesize(
        state["agent_outputs"], state["ticker"], state["mode"], tracer
    )

    if tracer:
        tracer.log_recommendation(recommendation)
        tracer.end_trace()

    return {"recommendation": recommendation}


def build_analysis_graph():
    """Build the full analysis LangGraph."""
    graph = StateGraph(AnalysisState)

    # Nodes
    graph.add_node("route_query", route_query_node)
    graph.add_node("gather_context", gather_context_node)
    for i, agent_cls in enumerate(ALL_AGENTS):
        graph.add_node(agent_cls.AGENT_NAME, make_agent_node(agent_cls, index=i))
    graph.add_node("synthesis", synthesis_node)

    # Edges
    graph.set_entry_point("route_query")
    graph.add_edge("route_query", "gather_context")
    graph.add_conditional_edges("gather_context", fan_out_to_agents)
    for agent_cls in ALL_AGENTS:
        graph.add_edge(agent_cls.AGENT_NAME, "synthesis")
    graph.add_edge("synthesis", END)

    return graph.compile()
