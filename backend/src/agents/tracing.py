"""Deep Langfuse tracing for the full analysis pipeline.

Uses the Langfuse Python SDK:
  - Langfuse.create_trace_id() for valid 32-hex-char trace IDs
  - langfuse.start_span(trace_context={'trace_id': id}) for top-level spans
  - span.start_span() / span.start_generation() for nested children
  - span.update_trace(name=..., input=..., output=...) for trace metadata
  - langfuse.create_score(trace_id=...) for scores
"""

import json
import logging
from typing import Optional
from src.models.structured_outputs import AgentOutput, InvestmentRecommendation

logger = logging.getLogger(__name__)

_langfuse_client = None


def _get_langfuse():
    global _langfuse_client
    if _langfuse_client is not None:
        return _langfuse_client
    try:
        from src.utils.langfuse_setup import _langfuse_client as client
        _langfuse_client = client
        return _langfuse_client
    except Exception:
        return None


def _safe_json(obj, max_len: int = 5000) -> str:
    """Serialize obj to a JSON string, falling back to str() on failure."""
    try:
        text = json.dumps(obj, default=str, indent=2)
    except Exception:
        text = str(obj)
    return text[:max_len]


class TracingManager:
    """Deep Langfuse tracing for the full analysis pipeline."""

    def __init__(self):
        self.langfuse = _get_langfuse()
        self._trace_id: Optional[str] = None
        self._ticker: Optional[str] = None

    @classmethod
    def from_trace_id(cls, trace_id: Optional[str], ticker: Optional[str] = None) -> "TracingManager":
        """Reconstruct a TracingManager from a trace_id."""
        manager = cls()
        if trace_id and manager.langfuse:
            manager._trace_id = trace_id
            manager._ticker = ticker
        return manager

    def _trace_context(self) -> dict:
        """Return the trace_context dict for associating spans with this trace."""
        return {"trace_id": self._trace_id}

    def start_trace(self, ticker: str) -> Optional[str]:
        """Create top-level trace in Langfuse for a full analysis run.

        Creates a trace by generating a valid hex ID via create_trace_id(),
        then creating a brief root span whose update_trace() sets the
        trace name and input.
        """
        if not self.langfuse:
            return None
        self._ticker = ticker
        try:
            from langfuse import Langfuse
            self._trace_id = Langfuse.create_trace_id()

            # Create a root span to establish the trace and set its metadata
            root = self.langfuse.start_span(
                trace_context=self._trace_context(),
                name="analysis_pipeline",
                input={"ticker": ticker},
            )
            root.update_trace(
                name=f"{ticker}_analysis",
                input={"ticker": ticker},
                metadata={"ticker": ticker},
            )
            root.end()
        except Exception as e:
            logger.debug(f"Langfuse trace creation error: {e}")
            # Fallback: generate a hex ID manually
            import uuid
            self._trace_id = uuid.uuid4().hex
        return self._trace_id

    def span_context_gathering(self):
        """Create span for RAG context gathering phase."""
        if not self.langfuse or not self._trace_id:
            return _NullSpan()
        try:
            return self.langfuse.start_span(
                trace_context=self._trace_context(),
                name="gather_context",
            )
        except Exception as e:
            logger.debug(f"Langfuse span error: {e}")
            return _NullSpan()

    def log_rag_query(self, parent_span, agent_name: str, query: str,
                      response: str, index: int):
        """Log individual RAG query as a generation."""
        if isinstance(parent_span, _NullSpan):
            return
        try:
            gen = parent_span.start_generation(
                name=f"rag_query_{agent_name}_{index}",
                input=query,
                output=response[:2000],
                model="text-embedding-3-small",
            )
            gen.end()
        except Exception as e:
            logger.debug(f"Langfuse rag log error: {e}")

    def span_agent(self, agent_name: str):
        """Create span for one agent's execution."""
        if not self.langfuse or not self._trace_id:
            return _NullSpan()
        try:
            return self.langfuse.start_span(
                trace_context=self._trace_context(),
                name=f"agent_{agent_name}",
                metadata={"agent": agent_name},
                input={"ticker": self._ticker, "agent": agent_name},
            )
        except Exception as e:
            logger.debug(f"Langfuse agent span error: {e}")
            return _NullSpan()

    def log_llm_call(self, parent_span, messages, response, model: str, usage: dict):
        """Log Claude API call as a generation."""
        if isinstance(parent_span, _NullSpan):
            return
        try:
            # Extract clean input: last few messages for context
            input_data = []
            for msg in messages[-3:]:
                if isinstance(msg, dict):
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        input_data.append({"role": role, "content": content[:2000]})
                    elif isinstance(content, list):
                        input_data.append({"role": role, "content": _safe_json(content, 2000)})

            # Extract clean output: text content from response
            output_text = ""
            if hasattr(response, "content"):
                for block in response.content:
                    if hasattr(block, "text"):
                        output_text += block.text
                    elif hasattr(block, "type") and block.type == "tool_use":
                        output_text += f"\n[tool_use: {block.name}({_safe_json(block.input, 500)})]"
            output_text = output_text[:5000] or _safe_json(response, 5000)

            gen = parent_span.start_generation(
                name="llm_call",
                input=input_data,
                output=output_text,
                model=model,
                usage_details={
                    "input": usage.get("input_tokens", 0),
                    "output": usage.get("output_tokens", 0),
                },
            )
            gen.end()
        except Exception as e:
            logger.debug(f"Langfuse llm log error: {e}")

    def log_tool_call(self, parent_span, tool_name: str, tool_input: dict, tool_output):
        """Log tool execution as a span."""
        if isinstance(parent_span, _NullSpan):
            return
        try:
            output = tool_output if isinstance(tool_output, dict) else _safe_json(tool_output, 2000)
            span = parent_span.start_span(
                name=f"tool_{tool_name}",
                input=tool_input,
                output=output,
            )
            span.end()
        except Exception as e:
            logger.debug(f"Langfuse tool log error: {e}")

    def log_agent_score(self, parent_span, agent_output: AgentOutput):
        """Log agent's score and confidence."""
        if not self.langfuse or not self._trace_id:
            return
        try:
            self.langfuse.create_score(
                name=f"{agent_output.agent_name}_score",
                value=agent_output.score,
                trace_id=self._trace_id,
            )
            self.langfuse.create_score(
                name=f"{agent_output.agent_name}_confidence",
                value=agent_output.confidence,
                trace_id=self._trace_id,
            )
        except Exception as e:
            logger.debug(f"Langfuse score log error: {e}")

    def log_recommendation(self, recommendation: InvestmentRecommendation):
        """Log final recommendation scores and update trace output."""
        if not self.langfuse or not self._trace_id:
            return
        try:
            for name, value in [
                ("overall_score", recommendation.overall_score),
                ("financial_score", recommendation.financial_score),
                ("technical_score", recommendation.technical_score),
                ("sentiment_score", recommendation.sentiment_score),
                ("risk_score", recommendation.risk_score),
            ]:
                self.langfuse.create_score(
                    name=name,
                    value=value,
                    trace_id=self._trace_id,
                )

            # Update the trace with the final recommendation as output
            output_span = self.langfuse.start_span(
                trace_context=self._trace_context(),
                name="recommendation",
                input={
                    "num_agents": recommendation.num_agents,
                    "agent_scores": recommendation.agent_scores,
                },
                output={
                    "recommendation": recommendation.recommendation,
                    "overall_score": recommendation.overall_score,
                    "confidence": recommendation.confidence,
                    "thesis": recommendation.thesis,
                },
            )
            trace_name = f"{self._ticker}_analysis" if self._ticker else "analysis"
            output_span.update_trace(
                name=trace_name,
                output={
                    "recommendation": recommendation.recommendation,
                    "overall_score": recommendation.overall_score,
                    "confidence": recommendation.confidence,
                    "thesis": recommendation.thesis,
                    "agent_scores": recommendation.agent_scores,
                },
            )
            output_span.end()
        except Exception as e:
            logger.debug(f"Langfuse recommendation log error: {e}")

    def end_trace(self):
        """Flush and finalize."""
        if self.langfuse:
            try:
                self.langfuse.flush()
            except Exception as e:
                logger.debug(f"Langfuse flush error: {e}")


class _NullSpan:
    """No-op span when Langfuse is not configured."""

    def start_span(self, **kwargs):
        return _NullSpan()

    def start_generation(self, **kwargs):
        return _NullSpan()

    def start_observation(self, **kwargs):
        return _NullSpan()

    def update(self, **kwargs):
        return self

    def update_trace(self, **kwargs):
        return self

    def end(self):
        pass

    def score(self, **kwargs):
        pass
