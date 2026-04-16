# AIRAS v3 — AI-Powered Investment Research & Analysis System

**Live:** [airas-production.up.railway.app](https://airas-production.up.railway.app/)

A multi-agent system that answers **"Should I invest in this stock?"** by running 10 specialized AI analysts in parallel, each examining a different dimension of a company, and synthesizing their findings into a single weighted recommendation.

```
Input: Ticker (e.g. AAPL) + optional query
  ↓
RAG over SEC filings (pgvector + Corrective Retrieval)
  ↓
10 Parallel Claude Agents (each with tool-use loops calling live financial APIs)
  ↓
Weighted Synthesis → STRONG BUY / BUY / HOLD / SELL / STRONG SELL + written thesis
```

---

## Why This Architecture

A single LLM call can't do 10 types of financial analysis well simultaneously. Instead:

- **Specialized agents** with narrow scopes (financials, technicals, sentiment, risk, etc.) produce higher-quality outputs than one generalist
- **Parallel execution** via LangGraph keeps total analysis time under 60 seconds despite running 10 agents
- **Weighted aggregation** reflects real-world analyst importance — financials (20%) matter more than social sentiment (3%)
- **Multi-provider data fallback** (Finnhub → FMP → yfinance) ensures resilience against rate limits and outages
- **Corrective RAG** self-heals bad retrieval — grades chunks for relevance and falls back to web search when context is insufficient

---

## Architecture

### Full Request Lifecycle

```
Client: POST /api/analysis { "ticker": "NVDA" }
│
▼
┌──────────────────────────────────────────────────────────────┐
│  FastAPI + Uvicorn                                            │
│  Async request handling, Pydantic validation, OpenAPI docs    │
│  → Creates background job, returns job_id immediately         │
└────────────────────────────┬─────────────────────────────────┘
                             │
            ═══ BACKGROUND THREAD ═══
                             │
                             ▼
┌──────────────────────────────────────────────────────────────┐
│  1. SEC Data Check                                            │
│     Does this ticker have indexed documents?                  │
│     NO → Download 10-K filings from EDGAR → chunk → embed    │
│     YES → Skip (already indexed)                              │
└────────────────────────────┬─────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────┐
│  2. Cache Warm                                                │
│     Pre-fetch financial data into memory cache                │
│     so agents read from cache instead of each hitting APIs    │
└────────────────────────────┬─────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────┐
│  3. LangGraph StateGraph Execution                            │
│                                                               │
│  ┌─────────────┐   ┌──────────────────┐   ┌──────────────┐  │
│  │ Route Query  │──▶│ Gather Context   │──▶│ Fan-Out      │  │
│  │ Classify     │   │ RAG queries per  │   │ Send() API   │  │
│  │ intent,      │   │ active agent     │   │ 5s stagger   │  │
│  │ select agents│   │ via CRAG         │   │ between      │  │
│  └─────────────┘   └──────────────────┘   │ agents       │  │
│                                            └──────┬───────┘  │
│                                                   │          │
│        ┌──────────────────────────────────────────┘          │
│        ▼                                                      │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  10 Parallel Agent Nodes                                │  │
│  │                                                         │  │
│  │  Each agent runs a Claude tool-use loop:                │  │
│  │  1. Receive RAG context + task instructions             │  │
│  │  2. Call Claude with agent-specific tools               │  │
│  │  3. If tool_use → execute tool → append result → loop   │  │
│  │  4. If end_turn → parse JSON → AgentOutput              │  │
│  │  5. Galileo validates for hallucinations                │  │
│  │  6. Langfuse logs trace spans + scores                  │  │
│  │                                                         │  │
│  │  financial_analyst (20%)   technical_analyst (15%)       │  │
│  │  news_sentiment (12%)     analyst_ratings (10%)         │  │
│  │  risk_assessment (10%)    competitive_analysis (10%)    │  │
│  │  insider_activity (8%)    earnings_analysis (7%)        │  │
│  │  options_analysis (5%)    social_sentiment (3%)         │  │
│  └───────────────────────────────┬─────────────────────────┘  │
│                                  │                             │
│                                  ▼                             │
│  ┌────────────────────────────────────────────────────────┐   │
│  │  Synthesis Node                                         │   │
│  │  overall = Σ(score × confidence × weight) / Σ(c × w)   │   │
│  │  ≥ +0.6 → STRONG BUY   ≥ +0.2 → BUY                   │   │
│  │  ≥ -0.2 → HOLD         ≥ -0.6 → SELL                   │   │
│  │  < -0.6 → STRONG SELL                                   │   │
│  │  + Thesis generation via Claude                         │   │
│  └────────────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────┘
                             │
                             ▼
Client polls: GET /api/analysis/{job_id}
Or streams:   GET /api/analysis/{job_id}/stream (SSE)
```

### Worked Example — NVDA Full Analysis

Suppose the 10 agents return these results:

```
Agent                  Score   Confidence  Weight
─────────────────────  ──────  ──────────  ──────
financial_analyst      +0.55   0.85        0.20
technical_analyst      +0.45   0.80        0.15
news_sentiment         +0.30   0.65        0.12
risk_assessment        +0.40   0.75        0.10
competitive_analysis   +0.60   0.85        0.10
analyst_ratings        +0.50   0.80        0.10
insider_activity       +0.10   0.60        0.08
earnings_analysis      +0.45   0.80        0.07
options_analysis       +0.25   0.65        0.05
social_sentiment       +0.20   0.50        0.03
```

**Scoring formula:** `overall = Σ(score × confidence × weight) / Σ(confidence × weight)`

```
Numerator  = 0.55×0.85×0.20 + 0.45×0.80×0.15 + ... = 0.3330
Denominator = 0.85×0.20 + 0.80×0.15 + ...            = 0.7595
Overall    = 0.3330 / 0.7595 = +0.44 → BUY (≥ +0.2)
Display    = (0.44 + 1) × 50 = 72/100
```

**Category scores** (confidence-weighted average within each group):
- Financial: 75 (financial_analyst + earnings_analysis)
- Technical: 68 (technical_analyst + options_analysis)
- Sentiment: 68 (news + social + analyst_ratings)
- Risk: 70 (risk + competitive + insider)

Low-confidence agents are automatically down-weighted. If an agent fails (score=0.0, confidence=0.1), it barely affects the result.

---

## Agent System

### The 10 Agents

| Agent | Weight | SEC Sections | Tools | Analyzes |
|-------|--------|-------------|-------|----------|
| Financial Analyst | 20% | balance_sheet, income_statement, cash_flow | calculate_financial_ratio, compare_companies, get_stock_price | Revenue trends, margins, debt, FCF |
| Technical Analyst | 15% | — | get_technical_indicators, get_stock_price | RSI, MACD, Bollinger, trends |
| News Sentiment | 12% | risk_factors, analyst_commentary | get_social_sentiment | News tone, recent headlines |
| Analyst Ratings | 10% | analyst_commentary | get_analyst_ratings, get_stock_price | Upgrades, downgrades, price targets |
| Risk Assessment | 10% | risk_factors, contingencies | compare_companies, calculate_financial_ratio | Risk disclosures, mitigation |
| Competitive Analysis | 10% | business_description, market_overview | compare_companies | Market position, moat |
| Insider Activity | 8% | insider_transactions | get_insider_trades, get_stock_price | Insider buys/sells, 10b5-1 plans |
| Earnings Analysis | 7% | income_statement, earnings_guidance | calculate_financial_ratio, get_stock_price | Beat/miss history, EPS trends |
| Options Analysis | 5% | — | get_options_data, get_stock_price | IV, put/call ratio, open interest |
| Social Sentiment | 3% | — | get_social_sentiment | Reddit, StockTwits, social buzz |

### BaseAgent Design

Each agent is a subclass of `BaseAgent` with 5 class attributes — no method overrides needed:

```python
class FinancialAnalystAgent(BaseAgent):
    AGENT_NAME = "financial_analyst"
    SYSTEM_PROMPT = "You are a senior Financial Analyst with 20 years..."
    RAG_QUERIES = ["Revenue and earnings for {ticker}", ...]
    TOOLS = ["calculate_financial_ratio", "compare_companies", "get_stock_price"]
    SECTIONS = ["balance_sheet", "income_statement", "cash_flow"]
```

The base class handles: Claude tool-use loop (max 10 iterations), multi-provider fallback, JSON output parsing, Langfuse tracing, and Galileo validation.

### Agent Output Schema

```python
class AgentOutput(BaseModel):
    agent_name: str       # Which agent produced this
    ticker: str           # Stock ticker analyzed
    score: float          # [-1.0 very bearish ↔ +1.0 very bullish]
    confidence: float     # [0.0 uncertain ↔ 1.0 high confidence]
    metrics: dict         # Key quantitative metrics (P/E, RSI, etc.)
    strengths: list[str]  # 1-3 bullish factors
    weaknesses: list[str] # 1-3 bearish factors
    summary: str          # One-sentence synthesis
    sources: list[str]    # Data sources used
```

### Agent Isolation

Agents do **not** communicate with each other. They run in parallel via LangGraph's `Send()` API and write to shared state using an `operator.add` reducer that auto-concatenates their outputs. Only the synthesis node reads all agent outputs.

```
[Route Query] → writes: active_agents, mode
[Gather Context] → writes: rag_context[agent_name]
[Each Agent] → writes: agent_outputs (appended via reducer)
             → reads: ONLY its own rag_context
[Synthesis] → reads: ALL agent_outputs → produces recommendation
```

---

## Tool Calling

### 8 Financial Tools (Claude tool-use format)

| Tool | Description | Primary | Secondary | Fallback |
|------|-------------|---------|-----------|----------|
| `get_stock_price` | Current/historical price + volume | Finnhub | FMP | yfinance |
| `calculate_financial_ratio` | P/E, P/B, debt/equity, ROE, etc. | FMP | — | yfinance |
| `compare_companies` | Compare metrics against sector peers | FMP | — | yfinance |
| `get_technical_indicators` | RSI, MACD, Bollinger, SMA, EMA | FMP + ta lib | — | yfinance + ta lib |
| `get_insider_trades` | Insider buy/sell from SEC Form 4 | FMP | Finnhub | yfinance |
| `get_options_data` | Options chain, IV, put/call ratio | Tradier | — | yfinance |
| `get_analyst_ratings` | Consensus, price targets, upgrades | FMP | — | yfinance |
| `get_social_sentiment` | Reddit, StockTwits, news sentiment | Finnhub | StockTwits + Reddit | FMP news |

### Tool-Use Loop

```
for iteration in range(max=10):
    response = claude.messages.create(tools=agent.TOOLS)

    if stop_reason == "tool_use":
        result = execute_tool(name, input)    # tries primary → fallback
        messages.append(tool_result)           # feed result back to Claude
        continue                               # loop

    if stop_reason == "end_turn":
        return parse_json(response) → AgentOutput
```

**Caching:** FMP responses cached 5 min, yfinance 30 min. Prevents redundant API calls when multiple agents query the same ticker data.

---

## RAG: Four Retrieval Levels

### Level 1: Basic Retriever
```
Query → cosine similarity (top_k=5) → Retrieved chunks
```
- ~50ms latency. No filtering. Good for prototyping.

### Level 2: Intermediate Retriever (Default)
```
Query → Synonym expansion (revenue → "net sales total revenue operating revenue")
      → Tiered metadata filtering:
          Attempt 1: filter(ticker=X, section IN agent.SECTIONS)
          Attempt 2: filter(ticker=X)
          Attempt 3: no filter
      → Retrieved chunks
```
- ~100-200ms. Prevents cross-ticker contamination.

### Level 3: Advanced Retriever
```
Query → Parallel:
        ├── Multi-query: Claude generates 3 query variations
        ├── HyDE: Claude writes hypothetical SEC passage → embed → search
        └── Original query
      → Merge + deduplicate
      → Claude cross-encoder reranking (score 0-10)
      → Top-K reranked chunks
```
- ~2-4s (3-4 extra LLM calls). HyDE aligns better with SEC document vector space than embedding raw questions.

### Level 4: Corrective Retriever (CRAG) — Production Default
```
Query → Initial retrieval (IntermediateRetriever)
      → Hybrid score check: if top_score ≥ 0.72 → skip grading (fast path)
      → Relevance grading (Claude Haiku, batch 5 docs/call):
          RELEVANT (1.0)   — directly answers with facts/numbers
          PARTIAL (0.5)    — related but missing key details
          IRRELEVANT (0.0) — off-topic or wrong company
      → If usable_ratio < 0.5 → trigger correction:
          Strategy 1: Rule-based (add ticker, expand keywords, add SEC Item refs)
          Strategy 2: LLM-based (Claude rewrites query)
      → Re-retrieve with transformed query
      → Return best results + confidence metadata
```
- ~500ms-3s. Agents receive `[Retrieval confidence: 85%]` and `[Corrections applied: 1]` headers, adjusting their own confidence accordingly.

---

## Tech Stack: Choices, Pros & Cons

### LLM — Claude (Anthropic)

| | |
|---|---|
| **Pros** | Native tool-use API, reliable structured JSON in multi-turn loops, handles long SEC filing contexts well, consistent scoring calibration |
| **Cons** | Higher cost than GPT-4 mini, no fine-tuning available, rate limits under heavy concurrent use |
| **Alternatives considered** | GPT-4 (less reliable tool-use in long loops), Gemini (limited tool-use at the time), open-source (insufficient reasoning quality for financial analysis) |

### Embeddings — OpenAI `text-embedding-3-small`

| | |
|---|---|
| **Pros** | Low cost ($0.02/1M tokens), 1536 dimensions, good domain performance on financial text |
| **Cons** | External API dependency, not fine-tuned for SEC terminology |
| **Alternatives considered** | `text-embedding-3-large` (more expensive, marginal quality gain), Cohere Embed v3 (fewer benchmarks on financial text), local models (deployment complexity) |

### Orchestration — LangGraph

| | |
|---|---|
| **Pros** | `StateGraph` with typed `TypedDict` state, `Send()` API for clean parallel fan-out, `Annotated[list, operator.add]` reducer auto-merges parallel outputs, inspectable/traceable graph |
| **Cons** | Learning curve, abstractions can obscure control flow, tightly coupled to LangChain ecosystem |
| **Alternatives considered** | Raw asyncio (no state management, manual coordination), CrewAI (too opinionated for this use case), Temporal (overkill, adds infrastructure) |

### Vector DB — Supabase PostgreSQL + pgvector

| | |
|---|---|
| **Pros** | Single managed Postgres instance for vectors + JSONB metadata + relational data, HNSW index gives sub-100ms retrieval, GIN index on metadata for fast filtering, free tier |
| **Cons** | No built-in hybrid search (BM25 + vector), scaling requires Supabase plan upgrades, pgvector HNSW recall can degrade at scale |
| **Alternatives considered** | Pinecone (managed but separate infrastructure, no JSONB), Weaviate (self-hosted complexity), ChromaDB (not production-grade), Qdrant (additional service to manage) |

### RAG Framework — LlamaIndex

| | |
|---|---|
| **Pros** | Mature ingestion pipeline (chunking → embedding → storage), `VectorIndexRetriever` with composable metadata filters, easy to swap retrieval strategies |
| **Cons** | Heavy dependency tree, abstractions can be rigid, version churn |
| **Alternatives considered** | LangChain retrieval (less mature ingestion pipeline), custom implementation (reinventing the wheel), Haystack (fewer integrations) |

### API — FastAPI + Uvicorn

| | |
|---|---|
| **Pros** | Async-native (critical for 10 parallel agents), auto-generated OpenAPI docs, Pydantic validation built-in, SSE support for streaming |
| **Cons** | Single-process by default (need Gunicorn workers for CPU-bound), Python GIL limits true parallelism |
| **Alternatives considered** | Flask (no async), Django (too heavy), Express/Node (would lose Python ML ecosystem) |

### Financial Data — Multi-Provider Fallback

| Provider | Rate Limit | Strengths | Weaknesses |
|----------|-----------|-----------|------------|
| **Finnhub** (primary) | 60 calls/min | High rate limit, real-time quotes, insider data | Limited fundamental ratios |
| **FMP** (secondary) | 5 calls/min | Comprehensive fundamentals, analyst ratings | Severe rate limiting on free tier |
| **Tradier** (options) | Free sandbox | Real options chains with Greeks | Sandbox only, no real-time |
| **yfinance** (fallback) | Unlimited | Free, no API key, broad coverage | Unofficial API, can break, slower |

**Why multi-provider?** With 10 agents each making 2-3 tool calls concurrently, a single provider with 5 calls/min would fail systematically. The fallback chain ensures at least one provider succeeds.

### Observability — Langfuse + Galileo

| Tool | Role | What It Tracks |
|------|------|---------------|
| **Langfuse** | LLM tracing | Full trace hierarchy (pipeline → agent → tool call), token costs, latencies, agent scores |
| **Galileo** | Guardrails + evals | Hallucination detection, PII redaction, toxicity scoring, context relevance validation |

---

## LangGraph Orchestration Detail

### State Management

```python
class AnalysisState(TypedDict):
    ticker: str
    query: Optional[str]
    rag_level: str
    active_agents: list[str]              # Set by router
    mode: str                              # "focused" or "full"
    rag_context: dict[str, str]           # {agent_name → context_string}
    agent_outputs: Annotated[list[AgentOutput], operator.add]  # Auto-merge
    recommendation: Optional[InvestmentRecommendation]
    trace_id: Optional[str]               # Shared Langfuse trace
    errors: Annotated[list[str], operator.add]
```

### Graph Definition

```python
builder = StateGraph(AnalysisState)

builder.add_node("route_query", route_query_node)
builder.add_node("gather_context", gather_context_node)
builder.add_node("fan_out", fan_out_node)
# ... 10 agent nodes ...
builder.add_node("synthesis", synthesis_node)

builder.add_edge(START, "route_query")
builder.add_edge("route_query", "gather_context")
builder.add_edge("gather_context", "fan_out")

# Fan-out uses Send() for parallel execution
builder.add_conditional_edges(
    "fan_out",
    lambda state: [Send(name, state) for name in state["active_agents"]]
)

# All agent nodes → synthesis
for agent_name in ALL_AGENT_NAMES:
    builder.add_edge(agent_name, "synthesis")
builder.add_edge("synthesis", END)
```

### Focused vs Full Analysis

| Mode | Trigger | Agents | Use Case |
|------|---------|--------|----------|
| **Full** | No query or `--full` flag | All 10 | Comprehensive investment analysis |
| **Focused** | Query provided (e.g. "What's AAPL's debt?") | 2-3 selected by router | Fast, targeted answers |

---

## Evaluations

- **Retrieval evals** — RAGAS metrics (faithfulness, relevance, precision) against golden datasets
- **Pipeline benchmarking** — Scenario-based: record real pipeline runs as JSON, replay with deterministic mocking, score with a two-gate metric (valid routing × correct execution). Inspired by ["Benchmarking a Multimodal Agent"](https://www.hedra.com/blog/hedra-agent-evaluation).
- **RAG comparison** — A/B testing across Basic vs Intermediate vs Advanced retrieval with delta tables
- **Cost tracking** — Per-analysis cost breakdown by agent and model

### Two-Gate Benchmark Scoring

```
Per-stage:  P(stage) = P(valid_plan) × P(correct_execution | valid_plan)
Pipeline:   P(scenario) = P(routing) × ∏ P(agent_i) × P(synthesis)
Overall:    P(benchmark) = Σ(P(scenario_i) × stages_i) / Σ(stages_i)
```

Gate 1 validates structural correctness (did the router pick valid agents?). Gate 2 validates execution quality (did agents produce reasonable scores?). Longer scenarios are weighted higher since they're harder.

---

## Project Structure

```
airas-v3/
├── backend/
│   ├── config/                  # Pydantic BaseSettings (.env → typed config)
│   ├── src/
│   │   ├── agents/              # 10 agent implementations + orchestration
│   │   │   ├── base_agent.py    # BaseAgent with Claude tool-use loop
│   │   │   ├── graph.py         # LangGraph StateGraph definition
│   │   │   ├── router.py        # Query classification + agent selection
│   │   │   ├── synthesis.py     # Weighted score aggregation + thesis
│   │   │   ├── state.py         # AnalysisState TypedDict with reducers
│   │   │   ├── context.py       # ContextBuilder — RAG queries per agent
│   │   │   ├── prompts.py       # System prompts (persona, rubric, examples)
│   │   │   └── *.py             # Individual agent subclasses
│   │   ├── rag/                 # RAG engine
│   │   │   ├── retrieval.py     # 4 retrieval levels (Basic → CRAG)
│   │   │   ├── supabase_rag.py  # LlamaIndex + pgvector integration
│   │   │   ├── relevance_grader.py  # CRAG relevance scoring
│   │   │   └── corrections.py   # Query transformation strategies
│   │   ├── tools/               # Financial API clients + tool schemas
│   │   │   ├── financial_tools.py   # 8 tools with Claude function-calling schemas
│   │   │   ├── finnhub_client.py    # Primary: 60 calls/min
│   │   │   ├── fmp_client.py        # Secondary: 5 calls/min, cached
│   │   │   └── tradier_client.py    # Options: free sandbox
│   │   ├── models/              # Pydantic structured outputs
│   │   ├── guardrails/          # Galileo guardrail integration
│   │   ├── api/                 # FastAPI endpoints + SSE streaming
│   │   └── utils/               # Langfuse setup, LlamaIndex config
│   ├── evals/                   # Evaluation suite
│   │   ├── benchmark/           # Scenario recording, replay, scoring
│   │   ├── datasets/            # Golden sets for retrieval eval
│   │   └── metrics/             # RAGAS, cost/latency, Galileo metrics
│   ├── scripts/                 # CLI (download filings, build index, run analysis)
│   └── tests/
├── frontend/                    # React + Vite dashboard
├── Dockerfile
└── railway.toml
```

---

## Getting Started

```bash
# Install backend dependencies
cd backend
pip install -r requirements.txt

# Configure environment
cp .env.example .env  # Add your API keys

# Download and index SEC filings
python scripts/download_sec_filings.py --ticker AAPL
python scripts/smart_build_index.py

# Run a full analysis (all 10 agents)
python scripts/run_analysis.py --ticker AAPL

# Run a focused query (router selects 2-3 agents)
python scripts/run_analysis.py --ticker AAPL --query "What are the main risk factors?"

# Run with specific RAG level
python scripts/run_analysis.py --ticker AAPL --rag-level corrective

# Run tests
pytest --asyncio-mode=auto
```

### Required Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes | Claude API access |
| `OPENAI_API_KEY` | Yes | Embeddings (text-embedding-3-small) |
| `SUPABASE_URL` | Yes | Vector database |
| `SUPABASE_KEY` | Yes | Vector database auth |
| `POSTGRES_CONNECTION_STRING` | Yes | Direct DB connection for pgvector |
| `FMP_API_KEY` | No | Financial Modeling Prep (5 calls/min free) |
| `FINNHUB_API_KEY` | No | Finnhub (60 calls/min free) |
| `TRADIER_API_TOKEN` | No | Options data (free sandbox) |
| `LANGFUSE_PUBLIC_KEY` | No | Observability tracing |
| `LANGFUSE_SECRET_KEY` | No | Observability tracing |

Financial data providers gracefully degrade — if a key isn't set, that provider is skipped and the fallback chain continues.

---

## Key Design Decisions

**Why 10 agents instead of 1?** — Each agent has a narrow scope with a tailored system prompt, scoring rubric, and tool set. This produces more calibrated scores than asking one model to consider everything at once. Weights reflect how much each dimension should influence the final recommendation.

**Why LangGraph over raw asyncio?** — LangGraph's `Send()` API provides typed state management with reducer functions for parallel output merging. The state graph is inspectable and traceable, and the fan-out pattern maps cleanly to the "route → gather → analyze → synthesize" pipeline.

**Why Corrective RAG as the production default?** — Standard vector similarity retrieval fails silently when the indexed documents don't contain relevant information. CRAG grades retrieved chunks and falls back to web search, preventing confident-sounding answers built on irrelevant context. The fast-path score check avoids unnecessary grading when retrieval quality is already high.

**Why multi-provider fallback for financial data?** — Free-tier rate limits (FMP: 5 calls/min, Finnhub: 60 calls/min) make single-provider architectures fragile under concurrent agent execution. The fallback chain (Finnhub → FMP → yfinance) ensures at least one provider succeeds.

**Why confidence-weighted synthesis?** — Not all agent outputs are equally reliable. An agent with low retrieval quality or failed tool calls will report low confidence. The synthesis formula `Σ(score × confidence × weight) / Σ(confidence × weight)` automatically down-weights unreliable results, making the system self-healing when individual agents fail.

**Why 5-second stagger between agents?** — Launching 10 agents simultaneously creates burst traffic to both Claude's API and financial data providers. A 5-second stagger smooths the load while keeping total execution under 60 seconds.
