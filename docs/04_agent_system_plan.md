# AIRAS V3 — Agent System: Design Document

Complete reference for the 10-agent analysis system, covering every architectural decision, the reasoning behind each technology choice, trade-offs evaluated, code references, and usage examples.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Technology Decisions & Reasoning](#2-technology-decisions--reasoning)
3. [Pillar 1: LangGraph Orchestration](#3-pillar-1-langgraph-orchestration)
4. [Pillar 2: Context Engineering](#4-pillar-2-context-engineering)
5. [Pillar 3: Prompt Engineering](#5-pillar-3-prompt-engineering)
6. [Pillar 4: Deep Langfuse Monitoring](#6-pillar-4-deep-langfuse-monitoring)
7. [Progressive RAG System](#7-progressive-rag-system)
8. [Smart Query Router](#8-smart-query-router)
9. [The 10 Agents — Detailed Specs](#9-the-10-agents--detailed-specs)
10. [Financial Tools](#10-financial-tools)
11. [Synthesis Orchestrator](#11-synthesis-orchestrator)
12. [CLI Runner & Usage Examples](#12-cli-runner--usage-examples)
13. [Dependency Upgrades](#13-dependency-upgrades)
14. [Future Work](#14-future-work)

---

## 1. System Overview

AIRAS V3's agent system runs 10 specialized financial analysis agents in parallel, each examining a different facet of a stock (fundamentals, technicals, sentiment, risk, etc.). A synthesis orchestrator then combines their weighted outputs into a final `InvestmentRecommendation`.

### End-to-End Pipeline

```
User Input (ticker + optional query)
       │
       ▼
┌──────────────────┐
│   Query Router    │  Classifies query → selects 1-10 relevant agents
│   router.py:72   │  (skipped if no query → all 10 agents run)
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ Context Gathering │  Runs RAG queries for all active agents (up to 16 queries)
│   context.py:26  │  Uses progressive retrieval (basic/intermediate/advanced)
└──────┬───────────┘
       │
       ▼
┌──────────────────────────────────────────────────┐
│              Parallel Agent Execution              │
│  LangGraph Send() fans out to 10 agent nodes      │
│                                                    │
│  ┌────────┐ ┌────────┐ ┌────────┐    ┌─────────┐ │
│  │Fin Anlst│ │Tech Anlst│ │Risk    │...│Analyst  │ │
│  │  (20%) │ │  (15%) │ │ (10%) │    │Ratings  │ │
│  └───┬────┘ └───┬────┘ └───┬────┘    └────┬────┘ │
│      │          │          │               │      │
│      ▼          ▼          ▼               ▼      │
│   AgentOutput AgentOutput AgentOutput  AgentOutput│
│  (operator.add reducer merges all into one list)  │
└──────────────────────┬───────────────────────────┘
                       │
                       ▼
              ┌────────────────┐
              │   Synthesis     │  Weighted scoring → category scores → thesis
              │ synthesis.py:39│  → InvestmentRecommendation
              └────────┬───────┘
                       │
                       ▼
              ┌────────────────┐
              │   CLI Output    │  Console display + optional JSON file
              │run_analysis.py │
              └────────────────┘
```

### File Structure

```
src/agents/
├── __init__.py              — ALL_AGENTS list + AGENT_MAP dict
├── state.py                 — AnalysisState TypedDict (LangGraph state)
├── graph.py                 — LangGraph StateGraph (nodes, edges, Send fan-out)
├── router.py                — Smart query router
├── base_agent.py            — BaseAgent: Claude tool-use loop + JSON parsing
├── context.py               — ContextBuilder: RAG query execution
├── prompts.py               — All 10 system prompts
├── tracing.py               — TracingManager: deep Langfuse integration
├── synthesis.py             — Weighted scoring → InvestmentRecommendation
├── financial_analyst.py     — Agent 1 (20%)
├── news_sentiment.py        — Agent 2 (12%)
├── technical_analyst.py     — Agent 3 (15%)
├── risk_assessment.py       — Agent 4 (10%)
├── competitive_analysis.py  — Agent 5 (10%)
├── insider_activity.py      — Agent 6 (8%)
├── options_analysis.py      — Agent 7 (5%)
├── social_sentiment.py      — Agent 8 (3%)
├── earnings_analysis.py     — Agent 9 (7%)
└── analyst_ratings.py       — Agent 10 (10%)

src/rag/
└── retrieval.py             — BasicRetriever, IntermediateRetriever, AdvancedRetriever

src/tools/
└── financial_tools.py       — 8 tools (3 original + 5 new)

scripts/
└── run_analysis.py          — CLI entry point
```

---

## 2. Technology Decisions & Reasoning

### 2.1 LangGraph for Orchestration

**Decision:** Use LangGraph (`langgraph>=0.2.0`) instead of raw `asyncio`, `ThreadPoolExecutor`, or full LangChain.

**Alternatives Evaluated:**

| Option | Pros | Cons | Verdict |
|--------|------|------|---------|
| **Raw asyncio / ThreadPoolExecutor** | No dependencies, full control, lightweight | Manual state passing, manual error isolation, manual result merging, no built-in graph visualization | Rejected: too much boilerplate for fan-out + merge |
| **Full LangChain** | Large ecosystem, many integrations, community support | Very heavy dependency (~100 sub-packages), opinionated prompt templates we don't need, abstracts away Claude's native API | Rejected: too heavy, we only need orchestration |
| **LangGraph (chosen)** | Minimal dependency (just `langgraph` + `langchain-core`), `Send()` API for parallel fan-out, typed state with reducers, built-in error isolation per node | Adds two dependencies, LangGraph API surface is newer/less documented | **Chosen**: best balance of features vs weight |
| **Prefect / Airflow** | Battle-tested workflow engines, great monitoring | Massive overkill for a single pipeline, designed for data engineering not real-time agents | Rejected: wrong tool for the job |
| **Custom DAG framework** | Exactly what we need, no deps | Have to build state management, parallel execution, error handling from scratch | Rejected: reinventing the wheel |

**Why LangGraph wins — concrete comparison:**

Without LangGraph (ThreadPoolExecutor):
```python
# ~20 lines of boilerplate
with ThreadPoolExecutor(max_workers=10) as executor:
    futures = {
        executor.submit(agent.analyze, ticker, context): agent
        for agent in agents
    }
    results = []
    errors = []
    for future in as_completed(futures):
        try:
            results.append(future.result())
        except Exception as e:
            errors.append(str(e))
# Then manually pass results to synthesis, manually track state...
```

With LangGraph:
```python
# 3 lines — graph.py:72-74
def fan_out_to_agents(state):
    return [Send(name, state) for name in state["active_agents"] if name in AGENT_MAP]
graph.add_conditional_edges("gather_context", fan_out_to_agents)
```

The `operator.add` reducer on `AnalysisState.agent_outputs` automatically merges outputs from all parallel nodes — zero manual merging code.

**What we DON'T use from LangChain:** We don't use LangChain agents, chains, prompt templates, or output parsers. We call Claude's Messages API directly. LangGraph is purely the orchestration layer.

**Code references:**
- Graph definition: `src/agents/graph.py:96-113`
- State with reducers: `src/agents/state.py:9-19`
- Fan-out function: `src/agents/graph.py:52-57`

---

### 2.2 Claude (Anthropic) for Agent LLM

**Decision:** Use Claude (via `anthropic>=0.45.0`) as the LLM for all 10 agents, NOT GPT-4 or an open-source model.

**Alternatives Evaluated:**

| Option | Pros | Cons | Verdict |
|--------|------|------|---------|
| **Claude 3.5 Sonnet (chosen)** | Native tool-use API (first-class, not bolted on), excellent at structured JSON output, strong reasoning for financial analysis, Anthropic SDK is clean/minimal | Slightly higher cost than GPT-3.5, fewer fine-tuning options | **Chosen** |
| **GPT-4 Turbo** | Mature ecosystem, function calling, JSON mode | Tool-use is function calling (different paradigm), already used for RAG synthesis (mixing creates confusion), OpenAI SDK is heavier | Rejected: Claude's tool-use is cleaner |
| **GPT-3.5 Turbo** | Much cheaper, fast | Weaker reasoning, struggles with complex financial analysis, worse at following JSON schemas | Rejected: quality too low for financial analysis |
| **Open-source (Llama 3, Mixtral)** | Free, self-hosted, no API costs | Requires GPU infrastructure, weaker tool-use support, worse financial reasoning, operational burden | Rejected: not worth the infrastructure complexity |

**Why Claude's tool-use matters:**

Claude's tool-use API is native to the Messages API. The model decides when to call tools and handles multi-turn tool loops naturally:

```python
# base_agent.py:48-54 — Claude tool-use in action
response = client.messages.create(
    model=settings.claude_model,
    max_tokens=2048,
    temperature=self.TEMPERATURE,
    system=self.SYSTEM_PROMPT,
    messages=messages,
    tools=tools,  # Claude natively understands these
)

if response.stop_reason == "tool_use":
    # Claude chose to call a tool — extract and execute
    tool_blocks = [b for b in response.content if b.type == "tool_use"]
```

With GPT-4, the equivalent requires function calling schemas in a different format, and the response parsing is less clean. Since our agents make 1-3 tool calls each, the clean loop matters.

**Why we upgraded from `anthropic==0.18.1`:** The old SDK version didn't support the tool-use API at all. Tool-use was added in `anthropic>=0.25.0`. We pin to `>=0.45.0` to get the stable tool-use implementation.

**Code references:**
- Tool-use loop: `src/agents/base_agent.py:37-80`
- Tool schema format: `src/tools/financial_tools.py:263-350`

---

### 2.3 Langfuse for Monitoring (not Weights & Biases, not LangSmith)

**Decision:** Use Langfuse (`langfuse>=2.50.0`) for tracing, not W&B Prompts, LangSmith, or Helicone.

**Alternatives Evaluated:**

| Option | Pros | Cons | Verdict |
|--------|------|------|---------|
| **Langfuse (chosen)** | Open-source (self-hostable), deep tracing (spans, generations, scores), built for LLM apps, generous free tier, already in the project | Smaller community than LangSmith, newer | **Chosen**: already integrated, best trace hierarchy |
| **LangSmith** | Built by LangChain team, deep LangChain integration | Requires LangChain ecosystem, proprietary, more expensive | Rejected: we don't use LangChain |
| **Weights & Biases** | Industry standard for ML, great dashboards | Designed for ML training not LLM agents, tracing is an afterthought | Rejected: wrong tool |
| **Helicone** | Simple, proxy-based, easy setup | Less granular tracing (no custom spans), limited scoring | Rejected: too simple for our needs |
| **Custom logging** | Full control, no dependencies | Have to build everything: dashboards, cost tracking, trace hierarchy | Rejected: massive engineering effort |

**Why we upgraded from `langfuse==2.27.3`:** The old version had basic tracing only. Version `>=2.50.0` adds:
- Deep span hierarchy (spans within spans)
- Generation tracking (input/output/tokens/model per LLM call)
- Score API (attach numeric scores to traces)
- Better flush reliability

**What our trace hierarchy looks like:**

```
TRACE: "AAPL_full_analysis"
├── SPAN: "gather_context"
│   ├── GENERATION: "rag_query_financial_analyst_0"
│   ├── GENERATION: "rag_query_financial_analyst_1"
│   └── ... (up to 16 RAG queries)
├── SPAN: "agent_financial_analyst"
│   ├── GENERATION: "llm_call" (Claude messages API)
│   ├── SPAN: "tool_calculate_financial_ratio"
│   ├── GENERATION: "llm_call" (follow-up)
│   └── SCORE: score=0.55, confidence=0.85
├── SPAN: "agent_technical_analyst"
│   └── ...
├── ... (8 more agent spans)
├── SPAN: "agent_synthesis"
│   └── GENERATION: "llm_call" (thesis generation)
└── SCORES: overall=0.444, financial=0.50, technical=0.15, ...
```

**Code references:**
- TracingManager: `src/agents/tracing.py:1-130`
- NullSpan pattern (graceful no-op when Langfuse not configured): `src/agents/tracing.py:123-134`

---

### 2.4 yfinance for Market Data (not Alpha Vantage, not IEX, not polygon.io)

**Decision:** Use `yfinance` for all live market data tools.

**Alternatives Evaluated:**

| Option | Pros | Cons | Verdict |
|--------|------|------|---------|
| **yfinance (chosen)** | Free, no API key needed, covers price/fundamentals/options/insiders/analyst data, already in the project | Unofficial (scrapes Yahoo Finance), rate limiting, occasional breakage | **Chosen**: free + covers all our data needs |
| **Alpha Vantage** | Official API, reliable, free tier | Free tier: 5 calls/minute (too slow for 10 agents), paid tiers expensive, less data coverage (no options/insiders) | Rejected: rate limits kill parallel agents |
| **IEX Cloud** | Reliable, good fundamentals | Paid only ($9+/month minimum), no options data | Rejected: paid + missing data |
| **polygon.io** | Professional grade, real-time data | Paid only ($29+/month), overkill for analysis (not trading) | Rejected: unnecessary cost |
| **SEC EDGAR API** | Official, free, complete filings | Only filing data (no prices/options/insiders), we already use it for filings | Already used for RAG, not for live data |

**Why yfinance is sufficient:** Our agents don't trade — they analyze. We need current ratios, price history, insider trades, options chains, and analyst ratings. yfinance provides all of these from a single library with zero API keys. The occasional rate limit is acceptable since we're running analysis (not real-time trading).

**Data coverage by tool:**

| Tool | yfinance API Used | Data Returned |
|------|-------------------|---------------|
| `get_stock_price` | `Ticker.info`, `Ticker.history` | Current/historical price, market cap, P/E |
| `calculate_financial_ratio` | `Ticker.info` | P/E, ROE, D/E, profit margin, current ratio |
| `compare_companies` | `Ticker.info` (per ticker) | Revenue, margins, ROE, market cap comparisons |
| `get_technical_indicators` | `Ticker.history(period="6mo")` | SMA 20/50/200, RSI, MACD, Bollinger, volume |
| `get_insider_trades` | `Ticker.insider_transactions` | Buy/sell count, ratio, recent transactions |
| `get_options_data` | `Ticker.options`, `Ticker.option_chain` | Put/call ratio, implied volatility, volume |
| `get_analyst_ratings` | `Ticker.info`, `Ticker.recommendations` | Consensus, price targets, trend |

**Code references:**
- Original 3 tools: `src/tools/financial_tools.py:17-161`
- New 5 tools: `src/tools/financial_tools.py:163-350`

---

### 2.5 PRAW + VADER for Social Sentiment (not Twitter API, not Alpaca)

**Decision:** Use StockTwits (free REST API) + Reddit (PRAW library) + yfinance news + VADER sentiment as a multi-source social sentiment pipeline.

**Alternatives Evaluated:**

| Option | Pros | Cons | Verdict |
|--------|------|------|---------|
| **StockTwits API** | Free, no auth, has bull/bear labels, real-time | Limited history, can be noisy | **Used**: free + direct sentiment labels |
| **Reddit via PRAW** | Rich retail investor discussion, upvote signals | Requires free app credentials, noisy, WSB can be meme-heavy | **Used**: optional, graceful degradation if not configured |
| **yfinance news** | Free, no auth, headline-level data | Only headlines (no full articles), limited sentiment | **Used**: reliable baseline signal |
| **Twitter/X API** | Large volume, real-time sentiment | Extremely expensive ($100+/month for search), rate limits | Rejected: cost prohibitive |
| **Alpaca news API** | Curated financial news | Requires brokerage account, limited free tier | Rejected: unnecessary dependency |
| **FinBERT / financial BERT** | Domain-specific sentiment model | Requires GPU or slow CPU inference, model download | Rejected: VADER is sufficient for headlines |
| **VADER (chosen for NLP)** | Lightweight, no GPU, good for short text, rule-based | Less accurate than fine-tuned models, doesn't understand financial jargon deeply | **Chosen**: fast, no dependencies beyond nltk |

**Why VADER over FinBERT:** Our social sentiment agent (3% weight) analyzes headlines and short posts — not full articles. VADER handles short text well and runs instantly (rule-based, no model loading). FinBERT would add ~500MB model download and GPU dependency for a 3%-weight agent. Not worth it.

**Graceful degradation:** If Reddit credentials are not in `.env`, the agent skips Reddit and uses StockTwits + yfinance news only. If StockTwits is rate-limited, the agent still works with the other sources. The agent never crashes — it always returns *some* signal.

**Code references:**
- `get_social_sentiment()`: `src/tools/financial_tools.py:248-349`
- VADER helper: `src/tools/financial_tools.py:241-247`
- Reddit optional import: `src/tools/financial_tools.py:290-320`

---

### 2.6 LlamaIndex for RAG (not LangChain RAG, not custom)

**Decision:** Keep LlamaIndex for RAG (already in the project), add progressive retrieval on top.

**Alternatives Evaluated:**

| Option | Pros | Cons | Verdict |
|--------|------|------|---------|
| **LlamaIndex (kept)** | Already integrated, good pgvector support, `IngestionPipeline` handles chunking+metadata+embedding, `VectorStoreIndex` is clean | Some API surface is complex, documentation can lag | **Kept**: already working, adding retrieval layers on top |
| **LangChain RAG** | Popular, many retrievers | Would require replacing the entire RAG pipeline, different vector store integration | Rejected: rewriting working code for no benefit |
| **Custom pgvector queries** | Full SQL control, no framework overhead | Have to handle embedding, chunking, retrieval ranking manually | Rejected: too much low-level work |

**What we added on top:** The progressive retrieval system (`src/rag/retrieval.py`) wraps LlamaIndex's `VectorIndexRetriever` with three levels of sophistication. LlamaIndex handles the core vector search; our retrievers add metadata filtering, query rewriting, HyDE, and reranking.

**Code references:**
- Existing RAG: `src/rag/supabase_rag.py`
- Progressive retrieval: `src/rag/retrieval.py:1-158`

---

## 3. Pillar 1: LangGraph Orchestration

### 3.1 State Definition

The `AnalysisState` TypedDict defines all data flowing through the graph. Two fields use `Annotated[list, operator.add]` — LangGraph's **reducer pattern** — which automatically concatenates lists when multiple parallel nodes return the same key.

```python
# src/agents/state.py
class AnalysisState(TypedDict):
    ticker: str                                                # Input: "AAPL"
    query: Optional[str]                                       # "What is Apple's revenue?" or None
    active_agents: list[str]                                   # ["financial_analyst", "earnings_analysis"]
    mode: str                                                  # "full" | "focused"
    rag_level: str                                             # "basic" | "intermediate" | "advanced"
    rag_context: dict[str, str]                                # {"financial_analyst": "filing text..."}
    agent_outputs: Annotated[list[AgentOutput], operator.add]  # ← REDUCER
    recommendation: Optional[InvestmentRecommendation]         # Final output
    trace_id: Optional[str]                                    # Langfuse trace ID
    errors: Annotated[list[str], operator.add]                 # ← REDUCER
```

**Why reducers matter:** When 10 parallel agent nodes each return `{"agent_outputs": [my_output]}`, LangGraph uses `operator.add` to concatenate all 10 single-element lists into one 10-element list. Without reducers, only the last node's output would survive.

**Example flow:**

```
Agent 1 returns: {"agent_outputs": [AgentOutput(agent_name="financial_analyst", score=0.55)]}
Agent 2 returns: {"agent_outputs": [AgentOutput(agent_name="technical_analyst", score=0.20)]}
...
Agent 10 returns: {"agent_outputs": [AgentOutput(agent_name="analyst_ratings", score=0.35)]}

→ Reducer merges into: {"agent_outputs": [all 10 AgentOutput objects]}
→ Synthesis node receives the full list
```

### 3.2 Graph Structure

```
┌──────────────┐
│ route_query   │  Sets active_agents, mode, starts Langfuse trace
└──────┬───────┘
       │
┌──────▼───────┐
│gather_context │  Runs RAG queries for all active agents
└──────┬───────┘
       │
┌──────▼───────┐
│fan_out_agents │  Send() to 1-10 parallel nodes
└──────┬───────┘
       │
   [N parallel agent nodes]
       │
┌──────▼───────┐
│  synthesis    │  Weighted scoring → InvestmentRecommendation
└──────┬───────┘
       │
      END
```

**Code: Building the graph** (`src/agents/graph.py:96-113`):

```python
def build_analysis_graph():
    graph = StateGraph(AnalysisState)

    # Add all nodes
    graph.add_node("route_query", route_query_node)
    graph.add_node("gather_context", gather_context_node)
    for agent_cls in ALL_AGENTS:
        graph.add_node(agent_cls.AGENT_NAME, make_agent_node(agent_cls))
    graph.add_node("synthesis", synthesis_node)

    # Wire edges
    graph.set_entry_point("route_query")
    graph.add_edge("route_query", "gather_context")
    graph.add_conditional_edges("gather_context", fan_out_to_agents)  # Send() API
    for agent_cls in ALL_AGENTS:
        graph.add_edge(agent_cls.AGENT_NAME, "synthesis")
    graph.add_edge("synthesis", END)

    return graph.compile()
```

### 3.3 Error Isolation

Each agent runs in its own LangGraph node. If one agent throws an exception, it's caught in `make_agent_node` and added to `state["errors"]` — the other 9 agents still complete normally:

```python
# src/agents/graph.py:60-70
def make_agent_node(agent_cls):
    def node_fn(state):
        try:
            output = agent.analyze(state["ticker"], rag_context, tracer)
            return {"agent_outputs": [output]}
        except Exception as e:
            return {"errors": [f"{agent.AGENT_NAME}: {e}"]}  # Caught, not re-raised
    return node_fn
```

The synthesis node works with however many `AgentOutput` objects it receives — whether that's 10 (all succeeded) or 7 (3 failed).

---

## 4. Pillar 2: Context Engineering

### 4.1 What Context Engineering Means

Each agent receives a structured user message with clearly labeled sections. Claude sees exactly what data is available (RAG context), what tools it can use, and what output format is expected.

### 4.2 Context Window Structure

What Claude sees as the user message for a RAG-enabled agent:

```
=== SEC FILING DATA (RAG) ===

[Query: "AAPL revenue gross profit operating income net income margins"]
For the fiscal year ended September 30, 2023, Apple reported total net
revenue of $383.3 billion, a decrease of approximately 3% compared to
the prior year. Gross margin was 44.1%...

[Query: "AAPL balance sheet total assets liabilities shareholders equity"]
As of September 30, 2023, Apple reported total assets of $352,583 million,
total liabilities of $290,437 million...

[Query: "AAPL cash flow from operations free cash flow capital expenditures"]
Operating cash flow for fiscal year 2023 was $110,543 million...

[Query: "AAPL debt structure interest expense debt maturity"]
Total term debt was $95,281 million as of September 30, 2023...

=== YOUR TASK ===

Analyze AAPL. Reason step-by-step:
1. Review the data provided above (if any)
2. Use your available tools to get current market data
3. Identify 1-3 key strengths and 1-3 key weaknesses
4. Assign a score from -1.0 (very bearish) to +1.0 (very bullish)
5. Assign a confidence from 0.0 to 1.0

Respond with ONLY valid JSON matching this schema:
{...}
```

For tool-only agents (technical_analyst, risk_assessment, etc.), the RAG section is absent — they only see the task section and rely entirely on tools.

### 4.3 ContextBuilder

The `ContextBuilder` (`src/agents/context.py`) is responsible for:

1. Iterating over all active agents
2. Running each agent's `RAG_QUERIES` (with `{ticker}` substituted)
3. Using the configured retrieval level (basic/intermediate/advanced)
4. Returning a `dict[str, str]` mapping `agent_name → context_string`

This runs **once** in the `gather_context` node, before any agents execute. This is more efficient than each agent querying independently — all 16 RAG queries (4 per RAG agent × 4 RAG agents) run upfront.

**Code reference:** `src/agents/context.py:26-68`

### 4.4 Agent RAG Query Design

Each RAG agent defines 4 queries that target different aspects of its domain:

| Agent | Query 1 | Query 2 | Query 3 | Query 4 |
|-------|---------|---------|---------|---------|
| Financial Analyst | Revenue, margins | Balance sheet | Cash flow | Debt structure |
| News Sentiment | MD&A outlook | Forward guidance | Risk factors | YoY performance |
| Competitive Analysis | Market position | IP, patents | Barriers to entry | Competitive threats |
| Earnings Analysis | EPS trends | Revenue growth | Segment breakdown | Cost structure |

Queries use `{ticker}` placeholder: `"{ticker} revenue gross profit operating income"` becomes `"AAPL revenue gross profit operating income"` at runtime.

**Why 4 queries per agent:** Testing showed that 4 targeted queries retrieve more relevant context than 1-2 broad queries. Each query focuses on a specific sub-topic, reducing the chance of missing relevant chunks. More than 4 queries showed diminishing returns (redundant chunks).

---

## 5. Pillar 3: Prompt Engineering

### 5.1 Prompt Template Structure

Every agent's system prompt follows the same 6-section template:

```
1. PERSONA     — "You are a senior [role] at [institution] with [N] years of experience..."
2. EXPERTISE   — What the agent specializes in
3. SCORING     — Agent-specific rubric mapping analysis results to score ranges
4. METHOD      — Step-by-step Chain-of-Thought (CoT) instructions
5. OUTPUT      — JSON schema for AgentOutput
6. EXAMPLE     — One complete few-shot example with realistic values
```

### 5.2 Techniques Applied

| Technique | How Applied | Why |
|-----------|-------------|-----|
| **Expert Persona** | "You are a senior credit analyst at a top-tier investment bank with 15+ years..." | Research shows personas improve domain-specific reasoning; Claude generates more nuanced financial analysis when it "role-plays" an expert |
| **Scoring Rubric** | Per-agent score ranges: ">0.5 if strong balance sheet AND liquidity" | Anchors the score to specific conditions, reducing variance between runs |
| **Chain-of-Thought** | "Reason step-by-step: 1. Review filing data, 2. Use tools, 3. Identify strengths/weaknesses..." | Forces Claude to show intermediate reasoning before scoring, improving accuracy |
| **Few-Shot Example** | One complete JSON example with realistic values per agent | Shows Claude the exact expected output format and value ranges, reducing parse errors |
| **Structured Output** | "Respond with ONLY valid JSON matching this schema" | Minimizes non-JSON text in responses, making parsing reliable |
| **Temperature Tuning** | Quantitative agents: 0.2, Qualitative agents: 0.4 | Lower temp for agents that should produce consistent numeric analysis; higher temp for agents that need creative reasoning about moats/management |

### 5.3 Temperature Strategy

| Temperature | Agents | Reasoning |
|-------------|--------|-----------|
| **0.2** | Financial Analyst, Technical Analyst, Risk Assessment, Insider Activity, Options Analysis, Earnings Analysis, Analyst Ratings | These produce numeric-heavy analysis. Consistency matters — same data should produce similar scores. |
| **0.3** | Social Sentiment | Moderate: interprets noisy social data, needs some flexibility |
| **0.4** | News Sentiment, Competitive Analysis | These analyze qualitative text (MD&A language, competitive moat descriptions). Need more creative reasoning to interpret tone and strategy. |

**Code reference:** Prompt definitions in `src/agents/prompts.py:1-267`

### 5.4 Example: Financial Analyst Prompt

```python
# src/agents/prompts.py — Financial Analyst
FINANCIAL_ANALYST_PROMPT = f"""You are a senior financial analyst at a top-tier
investment bank with 15+ years of experience evaluating corporate fundamentals.

## Your Expertise
You specialize in balance sheet analysis, income statement evaluation,
cash flow assessment, and financial ratio interpretation.

## Scoring Guide
- Score > +0.5: Strong financials — healthy balance sheet, growing revenue, solid margins
- Score +0.2 to +0.5: Adequate financials with some concerns
- Score -0.2 to +0.2: Mixed signals
- Score < -0.2: Weak financials — declining revenue, thin margins, or excessive leverage

## Analysis Method
1. Review the SEC filing data (if provided) for historical context
2. Use tools to get current ratios and peer comparisons
3. Cross-reference historical (filing) vs current (live) data
4. Reason step-by-step about strengths and weaknesses
5. Assign score and confidence based on evidence

## Output Format
Respond with ONLY valid JSON:
{AGENT_OUTPUT_SCHEMA}

## Example
{{"agent_name": "financial_analyst", "ticker": "AAPL", "score": 0.55, ...}}
```

---

## 6. Pillar 4: Deep Langfuse Monitoring

### 6.1 TracingManager Design

The `TracingManager` (`src/agents/tracing.py`) replaces the simple `@trace_agent` decorator with a hierarchical tracing system. It creates nested spans that map directly to the pipeline stages.

**Key design decisions:**

1. **Null Object Pattern:** When Langfuse is not configured (no API keys), `TracingManager` returns `_NullSpan` objects that silently no-op on all method calls. This means agent code never checks `if langfuse_enabled:` — it always calls `tracer.log_llm_call(...)` and it either logs or does nothing.

2. **Trace reconstruction via ID:** Since LangGraph nodes run in parallel (potentially different threads), the `TracingManager.from_trace_id(trace_id)` classmethod reconstructs a manager attached to the same Langfuse trace, allowing all parallel agents to log under one trace.

3. **Truncation:** Long inputs/outputs are truncated (`[:5000]` for LLM calls, `[:2000]` for tools/RAG) to avoid overwhelming Langfuse storage.

### 6.2 What Appears in the Langfuse Dashboard

| Dashboard Tab | What's Visible |
|---------------|---------------|
| **Traces** | `"AAPL_full_analysis"` with total duration and cost |
| **Trace detail** | Nested spans: gather_context → 10 agent spans → synthesis |
| **Agent span drill-down** | LLM generations (full messages), tool call spans, token counts |
| **Generations** | Every Claude API call with input/output, model, tokens, latency |
| **Scores** | 10 agent scores + confidences, 4 category scores, overall score |
| **Cost** | Per-generation token costs, total analysis cost |

### 6.3 Graceful Degradation

If Langfuse keys are not in `.env`, the entire tracing system degrades gracefully:

```python
# setup_langfuse() returns None
# TracingManager.__init__() sets self.langfuse = None
# All methods check: if not self.trace: return / return _NullSpan()
# _NullSpan.span() returns _NullSpan, .generation() / .end() are no-ops
```

Zero code changes needed in agent logic. Zero performance overhead.

**Code reference:** `src/agents/tracing.py:123-134` (NullSpan pattern)

---

## 7. Progressive RAG System

### 7.1 The Problem

The original RAG system uses only basic vector similarity search (`src/rag/supabase_rag.py`). This works for simple queries but misses relevant chunks when:

- The query terms don't appear in the answer text (semantic gap)
- Chunks from other companies dilute results (no ticker filtering)
- Abstract questions get poor embedding matches ("What is Apple's moat?")

### 7.2 Three Retrieval Levels

File: `src/rag/retrieval.py`

#### Level 1: Basic

```python
class BasicRetriever:
    def retrieve(self, query, top_k=5, ticker=None, sections=None):
        retriever = VectorIndexRetriever(index=self.index, similarity_top_k=top_k)
        return retriever.retrieve(query)
```

- **What it does:** Embed query → cosine similarity search → top-5 chunks
- **Pros:** Fast (~1s), simple, no extra dependencies
- **Cons:** No filtering, no query enhancement, semantic gaps
- **Best for:** Simple factual lookups ("What was AAPL revenue in 2023?")

#### Level 2: Intermediate (Default)

```python
class IntermediateRetriever:
    def retrieve(self, query, top_k=5, ticker=None, sections=None):
        filters = self._build_filters(ticker, sections)
        enhanced_query = self._rewrite_query(query)
        retriever = VectorIndexRetriever(index=self.index, similarity_top_k=top_k, filters=filters)
        return retriever.retrieve(enhanced_query)
```

- **What it adds over Basic:**
  - Metadata filtering: WHERE `ticker = 'AAPL'` AND `section IN ('balance_sheet', 'income_statement')`
  - Query rewriting: Rule-based synonym expansion ("debt" → "total debt long-term debt term debt borrowings")
- **Extra LLM calls:** 0 (rule-based only)
- **Pros:** Much more precise results, eliminates cross-company contamination
- **Cons:** Relies on metadata quality, synonym list is static
- **Best for:** Most agent queries (the default)

**Synonym expansion table** (`src/rag/retrieval.py:26-37`):

```python
EXPANSIONS = {
    "revenue": "revenue net sales total revenue",
    "debt": "total debt long-term debt term debt borrowings",
    "profit": "net income earnings profit net earnings",
    "cash": "cash and cash equivalents cash position liquidity",
    "margin": "gross margin operating margin profit margin",
    "growth": "growth year over year increase trend",
    "assets": "total assets current assets non-current assets",
    "eps": "earnings per share diluted EPS basic EPS",
    ...
}
```

#### Level 3: Advanced

```python
class AdvancedRetriever:
    def retrieve(self, query, top_k=5, ticker=None, sections=None):
        queries = self._multi_query(query)         # 3 LLM-generated variants
        queries.append(query)                       # Original
        queries.append(self._hyde_transform(query)) # Hypothetical answer

        # Retrieve for each variant (with metadata filters)
        all_nodes = [intermediate.retrieve(q) for q in queries]

        # Deduplicate + Rerank via Claude
        return self._rerank(query, deduplicate(all_nodes), top_n=top_k)
```

- **What it adds over Intermediate:**
  - **HyDE (Hypothetical Document Embeddings):** Generate a hypothetical answer, embed that instead of the question. The answer's embedding is closer to the real answer's embedding.
  - **Multi-Query:** Generate 3 query variations emphasizing different keywords.
  - **LLM Reranking:** Retrieve 15+ chunks across all variants, then use Claude to rank by relevance and keep top-5.
- **Extra LLM calls:** 3 (multi-query + HyDE + rerank)
- **Pros:** Best retrieval quality, catches chunks that basic/intermediate miss
- **Cons:** 3 extra Claude calls per agent query (~4s latency per query vs ~1s)
- **Best for:** Complex/abstract questions ("What is Apple's competitive moat?")

### 7.3 HyDE — Worked Example

```
Original query: "What is Apple's competitive moat?"

Embedding of "What is Apple's competitive moat?" has low cosine similarity to:
  "Apple has 1.2 billion active devices creating strong ecosystem lock-in"
  (different vocabulary, different structure)

HyDE generates:
  "Apple's competitive moat is built on its ecosystem of 1+ billion active
   devices, brand loyalty, high switching costs, and a vertically integrated
   hardware-software-services platform that creates significant barriers."

Embedding of THIS hypothetical answer IS similar to the actual filing text
→ Vector search now finds the right chunks
```

### 7.4 Multi-Query — Worked Example

```
Original: "AAPL balance sheet total assets total liabilities"

Claude generates 3 variants:
1. "Apple Inc consolidated balance sheets assets liabilities equity 2023"
2. "AAPL total assets current non-current liabilities financial position"
3. "Apple balance sheet shareholders equity net worth September 2023"

→ Retrieve top-5 for each (15+ total) → deduplicate → rerank → keep best 5
→ Catches chunks that any single query would miss
```

### 7.5 Level Comparison

| | Basic | Intermediate | Advanced |
|---|---|---|---|
| **Vector search** | top_k=5 | top_k=5 with filters | top_k=5 per variant |
| **Metadata filtering** | None | Ticker + section | Ticker + section |
| **Query rewriting** | None | Rule-based synonyms | LLM multi-query (3 variants) |
| **HyDE** | No | No | Yes |
| **Reranking** | No | No | Yes (Claude as cross-encoder) |
| **Extra LLM calls** | 0 | 0 | 3 |
| **Latency per query** | ~1s | ~1s | ~4s |
| **Best for** | Simple lookups | Most queries (default) | Complex/abstract questions |

### 7.6 Configuration

```bash
# CLI
python scripts/run_analysis.py --ticker AAPL --rag-level basic
python scripts/run_analysis.py --ticker AAPL --rag-level intermediate  # default
python scripts/run_analysis.py --ticker AAPL --rag-level advanced

# settings.py default
rag_level: str = "intermediate"
```

**Code references:**
- `BasicRetriever`: `src/rag/retrieval.py:14-20`
- `IntermediateRetriever`: `src/rag/retrieval.py:23-64`
- `AdvancedRetriever`: `src/rag/retrieval.py:67-158`

---

## 8. Smart Query Router

### 8.1 The Problem

If a user asks "What is Apple's revenue?", running all 10 agents wastes API calls and time. Only 2 agents (financial_analyst, earnings_analysis) are relevant.

### 8.2 How It Works

The router uses a single lightweight Claude call to classify the query into categories, then maps categories to agent names:

```python
# src/agents/router.py:66-90
def route_query(query: str) -> list[str]:
    response = client.messages.create(
        model=settings.claude_model,
        max_tokens=100,         # Very short response
        temperature=0.0,         # Deterministic classification
        messages=[{"role": "user", "content": ROUTER_PROMPT.format(query=query)}],
    )
    categories = json.loads(response.content[0].text)
    # Map categories → agent names
```

### 8.3 Category → Agent Mapping

```python
CATEGORY_AGENTS = {
    "full_analysis": None,  # ALL agents
    "financial":    ["financial_analyst", "earnings_analysis"],
    "technical":    ["technical_analyst", "options_analysis"],
    "risk":         ["risk_assessment", "competitive_analysis"],
    "sentiment":    ["news_sentiment", "social_sentiment", "analyst_ratings"],
    "insider":      ["insider_activity"],
    "competitive":  ["competitive_analysis"],
}
```

### 8.4 Examples

| User Query | Categories | Agents Selected | Count |
|------------|-----------|-----------------|-------|
| "Should I buy AAPL?" | `["full_analysis"]` | All 10 | 10 |
| "What is Apple's revenue?" | `["financial"]` | financial_analyst, earnings_analysis | 2 |
| "Is AAPL overbought?" | `["technical"]` | technical_analyst, options_analysis | 2 |
| "What are insiders doing?" | `["insider"]` | insider_activity | 1 |
| "Apple risks and competition" | `["risk", "competitive"]` | risk_assessment, competitive_analysis | 2 |
| "AAPL news and analyst opinion" | `["sentiment"]` | news_sentiment, social_sentiment, analyst_ratings | 3 |
| No query / `--full` flag | N/A | All 10 | 10 |

### 8.5 Synthesis Adaptation

When fewer than 4 agents run, synthesis still produces an `InvestmentRecommendation` but with fewer data points. The `mode` field tracks this:
- `"full"` (4+ agents): Full recommendation with all category scores
- `"focused"` (<4 agents): Some category scores will be 0.0 (no agents contributed)

**Code reference:** `src/agents/router.py:1-90`

---

## 9. The 10 Agents — Detailed Specs

### 9.1 Agent Architecture

Every agent is a subclass of `BaseAgent` with exactly 5 class attributes:

```python
class FinancialAnalystAgent(BaseAgent):
    AGENT_NAME = "financial_analyst"     # Unique identifier
    WEIGHT = 0.20                        # Weight in synthesis (20%)
    TEMPERATURE = 0.2                    # Claude temperature
    SECTIONS = ["balance_sheet", ...]    # Metadata filter for RAG (None = all)

    SYSTEM_PROMPT = FINANCIAL_ANALYST_PROMPT   # From prompts.py
    RAG_QUERIES = ["{ticker} revenue...", ...]  # 0 or 4 queries
    TOOLS = ["calculate_financial_ratio", ...]  # Tool names from FINANCIAL_TOOLS
```

No method overrides needed. The `BaseAgent.analyze()` method handles the entire Claude tool-use loop for all agents.

### 9.2 BaseAgent.analyze() Flow

```
1. Build user message (RAG context + task instructions)
2. Filter FINANCIAL_TOOLS to only this agent's tool names
3. Loop up to 5 iterations:
   a. Call Claude Messages API with system prompt + messages + tools
   b. If stop_reason == "tool_use":
      - Extract tool_use blocks
      - Execute each tool via execute_tool()
      - Append results to messages
      - Continue loop
   c. If stop_reason == "end_turn":
      - Extract JSON from text response
      - Parse into AgentOutput
      - Return
4. If max iterations reached: return fallback (score=0.0, confidence=0.1)
```

**Code reference:** `src/agents/base_agent.py:37-80`

### 9.3 All 10 Agents — Reference Table

| # | Agent | Weight | Temp | Data | RAG Queries | Tools | Sections |
|---|-------|--------|------|------|-------------|-------|----------|
| 1 | `financial_analyst` | 20% | 0.2 | RAG+yfinance | 4 (revenue, balance sheet, cash flow, debt) | calculate_financial_ratio, compare_companies, get_stock_price | balance_sheet, income_statement, cash_flow |
| 2 | `news_sentiment` | 12% | 0.4 | RAG | 4 (MD&A, guidance, risk factors, performance) | get_stock_price | None (all) |
| 3 | `technical_analyst` | 15% | 0.2 | yfinance | 0 | get_technical_indicators, get_stock_price | None (no RAG) |
| 4 | `risk_assessment` | 10% | 0.3 | yfinance | 0 | calculate_financial_ratio, get_stock_price, get_technical_indicators | None (no RAG) |
| 5 | `competitive_analysis` | 10% | 0.4 | RAG | 4 (market position, IP, barriers, threats) | compare_companies, get_stock_price | None (all) |
| 6 | `insider_activity` | 8% | 0.2 | yfinance | 0 | get_insider_trades, get_stock_price | None (no RAG) |
| 7 | `options_analysis` | 5% | 0.2 | yfinance | 0 | get_options_data, get_stock_price | None (no RAG) |
| 8 | `social_sentiment` | 3% | 0.3 | Social APIs | 0 | get_social_sentiment, get_stock_price | None (no RAG) |
| 9 | `earnings_analysis` | 7% | 0.2 | RAG | 4 (EPS, revenue growth, segments, costs) | calculate_financial_ratio, compare_companies | income_statement |
| 10 | `analyst_ratings` | 10% | 0.2 | yfinance | 0 | get_analyst_ratings, get_stock_price | None (no RAG) |

**RAG agents (4):** Financial Analyst, News Sentiment, Competitive Analysis, Earnings Analysis
**yfinance agents (5):** Technical Analyst, Risk Assessment, Insider Activity, Options Analysis, Analyst Ratings
**Social agent (1):** Social Sentiment

### 9.4 Weight Rationale

| Agent | Weight | Why |
|-------|--------|-----|
| Financial Analyst | 20% | Fundamentals are the most reliable predictor of long-term value |
| Technical Analyst | 15% | Price trends matter for entry/exit timing |
| News Sentiment | 12% | Management tone in filings reveals forward outlook |
| Risk Assessment | 10% | Risk management is critical for portfolio decisions |
| Competitive Analysis | 10% | Moat durability determines long-term returns |
| Analyst Ratings | 10% | Wall Street consensus is a strong market-moving signal |
| Insider Activity | 8% | Insiders have information advantage (but sell for many reasons) |
| Earnings Analysis | 7% | Earnings quality matters but overlaps with Financial Analyst |
| Options Analysis | 5% | Options markets are predictive but noisy |
| Social Sentiment | 3% | Retail sentiment is the noisiest signal (but can indicate momentum) |

**Total: 100%**

### 9.5 Agent Interaction with Tools — Example

Here's what happens when the Technical Analyst agent runs:

```
1. Claude receives system prompt (technical analyst persona) + user message (task only, no RAG)

2. Claude decides to call get_technical_indicators:
   Response: stop_reason="tool_use"
   Content: [ToolUse(name="get_technical_indicators", input={"ticker": "AAPL"})]

3. Tool executes:
   get_technical_indicators("AAPL") → {
     "current_price": 237.50, "sma_20": 235.10, "sma_50": 228.40,
     "sma_200": 198.50, "rsi_14": 62.3, "macd": {"macd": 2.15, "signal": 1.80},
     "trend": "bullish", ...
   }

4. Tool result appended to messages

5. Claude may call get_stock_price for additional context:
   Response: stop_reason="tool_use"
   Content: [ToolUse(name="get_stock_price", input={"ticker": "AAPL"})]

6. Tool executes, result appended

7. Claude generates final analysis:
   Response: stop_reason="end_turn"
   Content: [Text(text='{"agent_name": "technical_analyst", "score": 0.20, ...}')]

8. JSON parsed → AgentOutput returned
```

---

## 10. Financial Tools

### 10.1 Original Tools (3)

| Tool | Function | Key yfinance API |
|------|----------|-----------------|
| `calculate_financial_ratio` | P/E, ROE, D/E, profit margin, current ratio | `Ticker.info` |
| `compare_companies` | Compare N tickers on a metric | `Ticker.info` per ticker |
| `get_stock_price` | Current or historical price | `Ticker.info`, `Ticker.history` |

### 10.2 New Tools (5)

#### `get_technical_indicators(ticker, period="6mo")`

**What it calculates:**

| Indicator | Calculation | Interpretation |
|-----------|-------------|----------------|
| SMA 20/50/200 | Simple moving average over N periods | Price above all = bullish trend |
| RSI 14 | Relative Strength Index (14-period) | >70 overbought, <30 oversold |
| MACD | 12-EMA minus 26-EMA, with 9-EMA signal line | MACD > signal = bullish momentum |
| Bollinger Bands | SMA(20) ± 2 standard deviations | Price near upper = potentially overextended |
| Volume avg 20d | Mean volume over last 20 trading days | Rising volume confirms trend |
| Price change 1m/3m | Percent change over 21/63 trading days | Momentum indicator |

**Helper functions** (`src/tools/financial_tools.py:163-211`):
- `_calculate_rsi()` — Standard RSI formula using gain/loss rolling means
- `_calculate_macd()` — EMA(12) - EMA(26) with EMA(9) signal line
- `_calculate_bollinger()` — SMA(20) ± 2σ

**Code reference:** `src/tools/financial_tools.py:214-252`

#### `get_insider_trades(ticker)`

Returns insider buy/sell counts, buy/sell ratio, and recent transaction details. Uses `Ticker.insider_transactions` DataFrame.

**Key signal:** Buy/sell ratio > 1 = bullish (insiders buy for one reason: they expect the price to rise).

**Code reference:** `src/tools/financial_tools.py:255-292`

#### `get_options_data(ticker)`

Returns put/call ratio, implied volatility, and volume from the nearest-expiration options chain.

**Key signals:**
- Put/call ratio > 1.0 = bearish market sentiment
- High implied volatility = market expects big moves
- Unusual call volume = potential bullish catalyst

**Code reference:** `src/tools/financial_tools.py:295-335`

#### `get_analyst_ratings(ticker)`

Returns consensus rating, price targets (mean/high/low/median), upside percentage, and 4-month recommendation trend.

**Code reference:** `src/tools/financial_tools.py:338-371`

#### `get_social_sentiment(ticker)`

Aggregates sentiment from 3 sources:

1. **StockTwits** (free, no auth): `GET /api/2/streams/symbol/{ticker}.json` — direct bull/bear labels
2. **Reddit** (PRAW, optional): Searches r/wallstreetbets, r/stocks, r/investing — VADER sentiment on titles
3. **yfinance news** (free): `Ticker.news` — VADER sentiment on headlines

Returns per-source and aggregate signals.

**VADER sentiment:** `nltk.sentiment.vader.SentimentIntensityAnalyzer` returns compound score -1 to +1. Fast (rule-based, no GPU). Sufficient for short text like headlines.

**Code reference:** `src/tools/financial_tools.py:380-450`

### 10.3 Tool Schema Format

All tools use Claude's native tool-use schema format:

```python
{
    "name": "get_technical_indicators",
    "description": "Calculate technical indicators (SMA, RSI, MACD, Bollinger Bands)",
    "input_schema": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string"},
            "period": {"type": "string", "default": "6mo"}
        },
        "required": ["ticker"]
    }
}
```

**Code reference:** `src/tools/financial_tools.py:452-532` (FINANCIAL_TOOLS list)

---

## 11. Synthesis Orchestrator

### 11.1 Weighted Scoring Algorithm

```python
# For each agent output:
weighted_contribution = score × confidence × weight

# Overall score:
overall = Σ(score × confidence × weight) / Σ(confidence × weight)
```

**Why confidence-weighted:** An agent with confidence=0.1 (failed or incomplete analysis) should barely affect the overall score. An agent with confidence=0.85 (comprehensive data, strong analysis) should dominate its weight category.

**Example calculation:**

```
financial_analyst:  score=+0.55, confidence=0.85, weight=0.20
  → weighted = 0.55 × 0.85 × 0.20 = 0.0935

technical_analyst:  score=+0.20, confidence=0.70, weight=0.15
  → weighted = 0.20 × 0.70 × 0.15 = 0.0210

risk_assessment:    score=+0.40, confidence=0.75, weight=0.10
  → weighted = 0.40 × 0.75 × 0.10 = 0.0300

... (7 more agents)

overall = sum(weighted) / sum(confidence × weight)
        = 0.3744 / 0.6890
        = +0.543 → BUY
```

### 11.2 Recommendation Thresholds

```
Score ≥ +0.6  → STRONG BUY
Score ≥ +0.2  → BUY
Score ≥ -0.2  → HOLD
Score ≥ -0.6  → SELL
Score < -0.6  → STRONG SELL
```

### 11.3 Category Scores

Each category score is computed from its constituent agents (confidence-weighted):

| Category | Agents | InvestmentRecommendation Field |
|----------|--------|-------------------------------|
| Financial | financial_analyst, earnings_analysis | `financial_score` |
| Technical | technical_analyst, options_analysis | `technical_score` |
| Sentiment | news_sentiment, social_sentiment, analyst_ratings | `sentiment_score` |
| Risk | risk_assessment, competitive_analysis, insider_activity | `risk_score` |

### 11.4 Thesis Generation

After scoring, one Claude call generates a structured thesis from all agent summaries:

```python
# src/agents/synthesis.py:101-127
response = client.messages.create(
    model=settings.claude_model,
    max_tokens=500,
    temperature=0.3,
    messages=[{
        "role": "user",
        "content": f"""Based on these agent analyses for {ticker} (recommendation: {rec}):
{summaries}

Generate a JSON with:
- "thesis": 2-3 sentence investment thesis
- "bullish_factors": 3 key bullish factors
- "bearish_factors": 3 key bearish factors
- "risks": 3 key risks"""
    }],
)
```

**Code reference:** `src/agents/synthesis.py:1-133`

---

## 12. CLI Runner & Usage Examples

### 12.1 Command Reference

```bash
python scripts/run_analysis.py [OPTIONS]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--ticker` | string | **required** | Stock ticker to analyze |
| `--query` | string | None | Specific question (activates router) |
| `--full` | flag | false | Force all 10 agents even with `--query` |
| `--rag-level` | choice | intermediate | `basic`, `intermediate`, or `advanced` |
| `--sequential` | flag | false | Run agents one at a time (for debugging) |
| `--output` | string | None | Save results to JSON file |

### 12.2 Usage Examples

#### Full analysis (all 10 agents)

```bash
python scripts/run_analysis.py --ticker AAPL
```

Output:
```
======================================================================
AIRAS V3 — Investment Analysis for AAPL
======================================================================
RAG Level: intermediate

======================================================================
  RECOMMENDATION: BUY
======================================================================
  Overall Score:  +0.444
  Confidence:     72.5%
  Agents Run:     10
  Time:           45.2s

  Category Scores:
    Financial:    +0.500
    Technical:    +0.150
    Sentiment:    +0.350
    Risk:         +0.400

  Agent Scores:
    financial_analyst           +0.550
    news_sentiment              +0.300
    technical_analyst           +0.200
    risk_assessment             +0.400
    competitive_analysis        +0.700
    insider_activity            +0.150
    options_analysis            -0.100
    social_sentiment            +0.150
    earnings_analysis           +0.450
    analyst_ratings             +0.350

  Thesis:
    Apple is a BUY based on strong fundamentals, wide competitive moat,
    and solid analyst consensus, partially offset by elevated valuation
    and near-term technical headwinds.

  Bullish Factors:
    + Industry-leading profitability with 25% net margins
    + Wide ecosystem moat with 1.2B+ active devices
    + Strong analyst consensus with 12.5% upside to targets

  Bearish Factors:
    - Elevated P/E ratio suggests limited valuation upside
    - RSI approaching overbought territory
    - Insider selling exceeds buying in recent quarters

  Risks:
    ! China regulatory risk to supply chain and sales
    ! Smartphone market maturation limiting growth
    ! Antitrust pressure on App Store revenue model
======================================================================
```

#### Focused query (router selects 2 agents)

```bash
python scripts/run_analysis.py --ticker AAPL --query "What is Apple's revenue?"
```

Only `financial_analyst` and `earnings_analysis` run. Faster and cheaper.

#### Advanced RAG with JSON output

```bash
python scripts/run_analysis.py --ticker AAPL --rag-level advanced --output aapl_results.json
```

Uses HyDE + reranking for better RAG retrieval. Saves full `InvestmentRecommendation` to JSON.

#### Force full analysis on a specific question

```bash
python scripts/run_analysis.py --ticker MSFT --query "Is Microsoft overvalued?" --full
```

The `--full` flag overrides the router, running all 10 agents even though the query only matches `financial` + `technical` categories.

**Code reference:** `scripts/run_analysis.py:1-107`

---

## 13. Dependency Upgrades

### 13.1 Changes to `requirements.txt`

| Package | Before | After | Why |
|---------|--------|-------|-----|
| `anthropic` | `==0.18.1` | `>=0.45.0` | Old SDK had no tool-use support. Tool-use API requires `>=0.25.0`, pinning to `>=0.45.0` for stability |
| `langfuse` | `==2.27.3` | `>=2.50.0` | Deep tracing features (spans, generations, scores) added in later versions |
| `langgraph` | N/A | `>=0.2.0` | **New:** LangGraph for agent orchestration (StateGraph, Send, reducers) |
| `langchain-core` | N/A | `>=0.3.0` | **New:** Required by LangGraph (minimal, not full langchain) |
| `praw` | N/A | `>=7.7.0` | **New:** Reddit API for social sentiment (optional) |
| `nltk` | N/A | `>=3.8.0` | **New:** VADER sentiment analysis for social data |

### 13.2 New Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `reddit_client_id` | Optional[str] | None | Reddit app client ID (for social sentiment) |
| `reddit_client_secret` | Optional[str] | None | Reddit app client secret |
| `rag_level` | str | "intermediate" | Default retrieval level |

---

## 14. Future Work

### 14.1 Evals & Testing (Phase 2)

1. **RAGAS Evaluation** (already in requirements as `ragas==0.1.5`):
   - Measure RAG retrieval quality: context relevancy, faithfulness, answer relevancy
   - Score each agent's RAG queries against ground-truth answers

2. **Agent Scoring Consistency:**
   - Run same ticker multiple times → measure score variance
   - Compare agent scores against known analyst consensus

3. **Langfuse Evals Integration:**
   - LLM-as-judge: separate Claude call grades each agent's output quality
   - Track scores over time as prompts are tuned

4. **Test Suite:**
   - Unit tests: metadata extraction, scoring algorithm, context builder
   - Integration tests: single agent against known ticker → valid AgentOutput
   - End-to-end: full analysis → InvestmentRecommendation with all fields populated

### 14.2 FastAPI Endpoints (Phase 3)

- `POST /api/analyze` — Run full or focused analysis
- `GET /api/analysis/{ticker}` — Retrieve cached results
- `GET /api/agents` — List available agents with weights
- WebSocket support for streaming agent progress

### 14.3 Potential Improvements

- **Agent memory:** Cache recent analyses to avoid re-running for same ticker within 24h
- **Peer group detection:** Auto-detect competitors for compare_companies tool calls
- **Configurable weights:** Allow users to adjust agent weights based on their investment style
- **Custom agents:** Plugin system for user-defined agents (e.g., ESG analysis)
