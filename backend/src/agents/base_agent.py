"""Base agent class with Claude tool-use loop."""

import json
import logging
import time
from typing import Optional
import anthropic

from config.settings import get_settings
from src.tools.financial_tools import FINANCIAL_TOOLS, execute_tool
from src.models.structured_outputs import AgentOutput
from src.agents.tracing import TracingManager

logger = logging.getLogger(__name__)

# Retry config for rate limits
MAX_RETRIES = 3
RETRY_BASE_DELAY = 30  # seconds


class BaseAgent:
    """Base class for all analysis agents."""

    AGENT_NAME: str = ""
    SYSTEM_PROMPT: str = ""
    RAG_QUERIES: list[str] = []
    TOOLS: list[str] = []
    SECTIONS: Optional[list[str]] = None
    WEIGHT: float = 0.0
    TEMPERATURE: float = 0.2
    MAX_TOOL_ITERATIONS: int = 10

    def analyze(self, ticker: str, rag_context: str, tracer: TracingManager) -> AgentOutput:
        """Run analysis with Claude tool-use loop."""
        settings = get_settings()
        agent_span = tracer.span_agent(self.AGENT_NAME) if tracer else None

        try:
            user_message = self._build_user_message(ticker, rag_context)
            messages = [{"role": "user", "content": user_message}]

            # Filter tools to only this agent's tools
            tools = [t for t in FINANCIAL_TOOLS if t["name"] in self.TOOLS]

            client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
            for iteration in range(self.MAX_TOOL_ITERATIONS):
                kwargs = {
                    "model": settings.claude_model,
                    "max_tokens": 2048,
                    "temperature": self.TEMPERATURE,
                    "system": self.SYSTEM_PROMPT,
                    "messages": messages,
                }
                if tools:
                    kwargs["tools"] = tools

                response = self._call_with_retry(client, kwargs)

                # Log to Langfuse
                if tracer and agent_span:
                    tracer.log_llm_call(
                        agent_span, messages, response, settings.claude_model,
                        {"input_tokens": response.usage.input_tokens,
                         "output_tokens": response.usage.output_tokens},
                    )

                if response.stop_reason == "tool_use":
                    tool_blocks = [b for b in response.content if b.type == "tool_use"]
                    messages.append({"role": "assistant", "content": response.content})

                    tool_results = []
                    for tool_block in tool_blocks:
                        try:
                            result = execute_tool(tool_block.name, tool_block.input)
                        except Exception as e:
                            result = {"error": str(e)}
                        if tracer and agent_span:
                            tracer.log_tool_call(
                                agent_span, tool_block.name, tool_block.input, result
                            )
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_block.id,
                            "content": json.dumps(result, default=str),
                        })
                    messages.append({"role": "user", "content": tool_results})

                else:
                    # End turn — extract JSON from response
                    text = next(
                        (b.text for b in response.content if hasattr(b, "text")), ""
                    )
                    output = self._try_parse_or_retry(
                        client, settings, messages, tools, text, ticker
                    )
                    if tracer and agent_span:
                        tracer.log_agent_score(agent_span, output)
                        agent_span.update(output={
                            "score": output.score,
                            "confidence": output.confidence,
                            "summary": output.summary,
                        })
                        agent_span.end()
                    return output

            # Max iterations reached — make one final call WITHOUT tools to
            # force Claude to output the JSON summary instead of calling more tools.
            logger.warning(
                f"{self.AGENT_NAME}: max tool iterations reached, forcing final output"
            )
            output = self._force_final_output(client, settings, messages, ticker)
            if tracer and agent_span:
                tracer.log_agent_score(agent_span, output)
                agent_span.update(output={
                    "score": output.score,
                    "confidence": output.confidence,
                    "summary": output.summary,
                })
                agent_span.end()
            return output

        except Exception as e:
            logger.error(f"{self.AGENT_NAME} error: {e}")
            if agent_span:
                agent_span.update(output={"status": "error", "error": str(e)})
                agent_span.end()
            return self._fallback_output(ticker, str(e))

    def _build_user_message(self, ticker: str, rag_context: str) -> str:
        """Build the user message with RAG context and task instructions."""
        parts = []

        if rag_context:
            parts.append(f"=== SEC FILING DATA (RAG) ===\n\n{rag_context}")

        parts.append(f"""=== YOUR TASK ===

Analyze {ticker}. Reason step-by-step:
1. Review the data provided above (if any)
2. Use your available tools to get current market data
3. Identify 1-3 key strengths and 1-3 key weaknesses
4. Assign a score from -1.0 (very bearish) to +1.0 (very bullish)
5. Assign a confidence from 0.0 to 1.0

Respond with ONLY valid JSON matching this schema:
{{
  "agent_name": "{self.AGENT_NAME}",
  "ticker": "{ticker}",
  "score": <float -1.0 to 1.0>,
  "confidence": <float 0.0 to 1.0>,
  "metrics": {{"key": "value"}},
  "strengths": ["strength 1", "strength 2"],
  "weaknesses": ["weakness 1", "weakness 2"],
  "summary": "One sentence summary.",
  "sources": ["source 1", "source 2"]
}}""")

        return "\n\n".join(parts)

    def _call_with_retry(self, client, kwargs) -> object:
        """Call Claude API with retry on rate limit errors."""
        for attempt in range(MAX_RETRIES + 1):
            try:
                return client.messages.create(**kwargs)
            except anthropic.RateLimitError as e:
                if attempt < MAX_RETRIES:
                    delay = RETRY_BASE_DELAY * (2 ** attempt)
                    logger.info(
                        f"{self.AGENT_NAME}: rate limited, retrying in {delay}s "
                        f"(attempt {attempt + 1}/{MAX_RETRIES})"
                    )
                    time.sleep(delay)
                else:
                    raise

    def _force_final_output(self, client, settings, messages, ticker: str) -> AgentOutput:
        """Make one final API call WITHOUT tools to force a JSON summary."""
        messages.append({
            "role": "user",
            "content": (
                "You have gathered enough data. Now provide your final analysis as ONLY "
                "valid JSON matching the required output schema. Do not call any more tools."
            ),
        })
        try:
            response = self._call_with_retry(client, {
                "model": settings.claude_model,
                "max_tokens": 2048,
                "temperature": self.TEMPERATURE,
                "system": self.SYSTEM_PROMPT,
                "messages": messages,
                # No tools — forces text output
            })
            text = next(
                (b.text for b in response.content if hasattr(b, "text")), ""
            )
            return self._parse_output(text, ticker)
        except Exception as e:
            logger.error(f"{self.AGENT_NAME}: forced final output failed: {e}")
            return self._fallback_output(ticker, str(e))

    def _try_parse_or_retry(self, client, settings, messages, tools,
                            text: str, ticker: str) -> AgentOutput:
        """Try to parse the response text. If empty or invalid JSON, retry once
        without tools to get a clean JSON response."""
        if text.strip():
            result = self._parse_output(text, ticker)
            # If parsing succeeded (not a fallback), return it
            if result.confidence > 0.1 or "Parse error" not in result.summary:
                return result
            logger.info(f"{self.AGENT_NAME}: bad JSON, retrying for clean output")

        else:
            logger.info(f"{self.AGENT_NAME}: empty response text, retrying for JSON")

        # Retry: append the assistant message and ask for JSON explicitly
        return self._force_final_output(client, settings, messages, ticker)

    def _parse_output(self, text: str, ticker: str) -> AgentOutput:
        """Parse JSON from Claude's response into AgentOutput."""
        json_str = text.strip()
        if not json_str:
            return self._fallback_output(ticker, "Empty response")

        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0]
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0]

        # Sometimes Claude wraps JSON in prose — try to find { ... }
        json_str = json_str.strip()
        if not json_str.startswith("{"):
            start = json_str.find("{")
            end = json_str.rfind("}")
            if start != -1 and end != -1 and end > start:
                json_str = json_str[start:end + 1]

        try:
            data = json.loads(json_str)
            # Force correct agent_name and ticker
            data["agent_name"] = self.AGENT_NAME
            data["ticker"] = ticker
            return AgentOutput(**data)
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"{self.AGENT_NAME}: JSON parse error: {e}")
            return self._fallback_output(ticker, f"Parse error: {e}")

    def _fallback_output(self, ticker: str, reason: str = "") -> AgentOutput:
        """Return neutral output when analysis fails."""
        return AgentOutput(
            agent_name=self.AGENT_NAME,
            ticker=ticker,
            score=0.0,
            confidence=0.1,
            metrics={},
            strengths=[],
            weaknesses=[],
            summary=f"Analysis incomplete: {reason}" if reason else "Analysis incomplete.",
            sources=[],
        )
