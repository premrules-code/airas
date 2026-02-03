"""Mock Claude API response objects for testing."""

import json
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MockUsage:
    input_tokens: int = 500
    output_tokens: int = 300


@dataclass
class MockTextBlock:
    type: str = "text"
    text: str = ""


@dataclass
class MockToolUseBlock:
    type: str = "tool_use"
    id: str = "toolu_01"
    name: str = ""
    input: dict = field(default_factory=dict)


@dataclass
class MockMessage:
    content: list = field(default_factory=list)
    stop_reason: str = "end_turn"
    usage: MockUsage = field(default_factory=MockUsage)
    model: str = "claude-sonnet-4-20250514"


def make_claude_response(text: str, stop_reason: str = "end_turn") -> MockMessage:
    """Create a mock Claude Message with text content."""
    return MockMessage(
        content=[MockTextBlock(text=text)],
        stop_reason=stop_reason,
    )


def make_tool_use_response(tool_name: str, tool_input: dict, tool_use_id: str = "toolu_01") -> MockMessage:
    """Create a mock Claude Message with tool_use content."""
    return MockMessage(
        content=[MockToolUseBlock(id=tool_use_id, name=tool_name, input=tool_input)],
        stop_reason="tool_use",
    )


def _agent_json(agent_name: str, ticker: str, score: float, confidence: float,
                summary: str, metrics: Optional[dict] = None) -> str:
    """Build a valid AgentOutput JSON string."""
    return json.dumps({
        "agent_name": agent_name,
        "ticker": ticker,
        "score": score,
        "confidence": confidence,
        "metrics": metrics or {"pe_ratio": 15.2},
        "strengths": ["Strong fundamentals", "Good growth"],
        "weaknesses": ["High debt"],
        "summary": summary,
        "sources": ["SEC filings", "FMP"],
    })


# --- Per-agent bullish responses ---

FINANCIAL_ANALYST_BULLISH = make_claude_response(
    _agent_json("financial_analyst", "AAPL", 0.65, 0.85,
                "AAPL shows strong fundamentals with healthy margins and low debt.")
)

FINANCIAL_ANALYST_BEARISH = make_claude_response(
    _agent_json("financial_analyst", "AAPL", -0.55, 0.80,
                "AAPL shows weakening margins and rising debt levels.")
)

FINANCIAL_ANALYST_NEUTRAL = make_claude_response(
    _agent_json("financial_analyst", "AAPL", 0.05, 0.65,
                "AAPL presents a mixed picture with stable but unremarkable fundamentals.")
)

# --- Edge cases ---

MALFORMED_JSON = make_claude_response("This is not JSON at all. {broken")

MARKDOWN_WRAPPED_JSON = make_claude_response(
    "Here is my analysis:\n\n```json\n"
    + _agent_json("financial_analyst", "AAPL", 0.5, 0.8, "Analysis complete.")
    + "\n```\n\nLet me know if you need more details."
)

BARE_FENCED_JSON = make_claude_response(
    "```\n"
    + _agent_json("financial_analyst", "AAPL", 0.4, 0.7, "Moderate outlook.")
    + "\n```"
)

EXTRA_TEXT_BEFORE_JSON = make_claude_response(
    "Based on my analysis of the financial data, here are my findings:\n\n"
    + _agent_json("financial_analyst", "AAPL", 0.3, 0.6, "Mixed signals.")
)

WRONG_AGENT_NAME_JSON = make_claude_response(
    _agent_json("wrong_agent", "WRONG", 0.5, 0.8, "Should be overridden.")
)

SCORE_OUT_OF_BOUNDS_JSON = make_claude_response(
    json.dumps({
        "agent_name": "financial_analyst",
        "ticker": "AAPL",
        "score": 1.5,
        "confidence": 0.8,
        "metrics": {},
        "strengths": [],
        "weaknesses": [],
        "summary": "Out of bounds score.",
        "sources": [],
    })
)

# --- Thesis response for synthesis tests ---

THESIS_RESPONSE = make_claude_response(
    json.dumps({
        "thesis": "AAPL is well-positioned for growth given strong fundamentals.",
        "bullish_factors": ["Strong revenue", "High margins", "Brand loyalty"],
        "bearish_factors": ["Premium valuation", "Supply chain risk", "Regulatory pressure"],
        "risks": ["Macro downturn", "Competition", "Rate hikes"],
    })
)
