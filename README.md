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

## Why This Architecture

A single LLM call can't do 10 types of financial analysis well simultaneously. Instead:

- **Specialized agents** with narrow scopes (financials, technicals, sentiment, risk, etc.) produce higher-quality outputs than one generalist
- **Parallel execution** via LangGraph keeps total analysis time under 60 seconds despite running 10 agents
- **Weighted aggregation** reflects real-world analyst importance — financials (20%) matter more than social sentiment (3%)
- **Multi-provider data fallback** (Finnhub → FMP → yfinance) ensures resilience against rate limits and outages

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    FastAPI + Uvicorn                                 │
└──────────────────────────┬──────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                LangGraph StateGraph                                  │
│                                                                      │
│  [1] Router ──▶ [2] Context Gathering ──▶ [3] Fan-Out (Send API)    │
│      │                  │                        │                   │
│  Classifies query   RAG queries per         10 agents in parallel    │
│  Selects agents     active agent                 │                   │
│                                                  ▼                   │
│                                         [4] Synthesis                │
│                                         Weighted scoring + thesis    │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Pipeline (Offline)

SEC filings → `SentenceSplitter` (512-token chunks) → `SECMetadataTransform` (section detection, fiscal period extraction) → OpenAI embeddings (`text-embedding-3-small`, 1536 dim) → Supabase PostgreSQL + pgvector (HNSW index)

### Analysis Pipeline (Online)

1. **Router** — Classifies the query intent and selects which agents to activate
2. **Context Gathering** — Runs RAG queries per agent using the configured retrieval level
3. **Agent Execution** — Each agent runs a Claude tool-use loop: receives RAG context → calls financial APIs via tool use → produces a scored `AgentOutput` (score from -1 to 1, confidence, summary, sections)
4. **Synthesis** — Weighted score aggregation across all agents, category scores, thesis generation via Claude → final `InvestmentRecommendation`

### The 10 Agents

| Agent | Weight | Primary Data Sources |
|-------|--------|---------------------|
| Financial Analyst | 20% | SEC filings (RAG) + FMP/yfinance |
| Technical Analyst | 15% | FMP historical prices + `ta` library |
| News Sentiment | 12% | SEC filings (RAG) |
| Analyst Ratings | 10% | FMP/yfinance |
| Risk Assessment | 10% | FMP/yfinance |
| Competitive Analysis | 10% | SEC filings (RAG) |
| Insider Activity | 8% | FMP/Finnhub/yfinance |
| Earnings Analysis | 7% | SEC filings (RAG) |
| Options Analysis | 5% | Tradier/yfinance |
| Social Sentiment | 3% | StockTwits + Reddit + Finnhub/FMP News |

Each agent is a subclass of `BaseAgent` with 5 class attributes — no method overrides needed. The base class handles the full tool-use loop, output parsing, tracing, and guardrail validation.

## RAG: Four Retrieval Levels

| Level | Technique | Use Case |
|-------|-----------|----------|
| **Basic** | Vector similarity search | Fast, simple queries |
| **Intermediate** | Metadata filtering + query rewriting | Targeted section/period lookups |
| **Advanced** | HyDE + multi-query + reranking | Complex analytical queries |
| **Corrective (CRAG)** | Relevance grading → web fallback if context is insufficient | Production default — self-healing retrieval |

The Corrective RAG pipeline grades retrieved chunks for relevance and falls back to web search when the vector store doesn't have sufficient context, preventing hallucinated answers from poor retrieval.

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| LLM | Claude (Anthropic) | Native tool-use, structured output, strong reasoning |
| Embeddings | OpenAI `text-embedding-3-small` | Cost-effective, 1536 dimensions, good financial domain performance |
| Orchestration | LangGraph | `StateGraph` with `Send()` for parallel fan-out, typed state with reducers |
| Vector DB | Supabase PostgreSQL + pgvector | HNSW index, JSONB metadata filtering, managed infrastructure |
| API | FastAPI + Uvicorn | Async, auto-generated OpenAPI docs, Pydantic validation |
| Financial Data | Finnhub (primary), FMP, Tradier, yfinance (fallback) | Multi-provider fallback chain for resilience |
| Observability | Langfuse | Full trace hierarchy: pipeline → agent → tool call, with token costs and latencies |
| Guardrails | Galileo AI | Hallucination detection, PII redaction, toxicity scoring, context relevance |
| Frontend | React + Vite | Lightweight dashboard for triggering and viewing analyses |
| Deployment | Railway + Docker | Containerized, environment-based configuration |

## Observability

Every analysis run produces a full Langfuse trace with:
- Per-agent execution spans (LLM calls, tool calls, scores)
- RAG query performance (retrieval latency, chunk relevance)
- Token usage and cost tracking across all Claude calls
- Agent output scores logged as Langfuse scores for trend analysis

Galileo guardrails run post-agent to validate outputs against the RAG context, flagging hallucinations before they reach the synthesis layer.

## Evaluations

- **Retrieval evals** — RAGAS metrics (faithfulness, relevance, precision) against golden datasets
- **Pipeline benchmarking** — Scenario-based evaluation: record real pipeline runs as JSON, replay with deterministic mocking, score with a two-gate metric (valid routing × correct execution). Inspired by ["Benchmarking a Multimodal Agent"](https://www.hedra.com/blog/hedra-agent-evaluation).
- **RAG comparison** — A/B testing across Basic vs Intermediate vs Advanced retrieval with delta tables
- **Cost tracking** — Per-analysis cost breakdown by agent and model

## Project Structure

```
airas-v3/
├── backend/
│   ├── config/                  # Settings (Pydantic BaseSettings)
│   ├── src/
│   │   ├── agents/              # 10 agent implementations + orchestration
│   │   │   ├── base_agent.py    # BaseAgent with Claude tool-use loop
│   │   │   ├── graph.py         # LangGraph StateGraph definition
│   │   │   ├── router.py        # Query classification + agent selection
│   │   │   ├── synthesis.py     # Weighted score aggregation + thesis
│   │   │   └── *.py             # Individual agent subclasses
│   │   ├── rag/                 # RAG engine (4 retrieval levels)
│   │   ├── tools/               # Financial API clients + tool schemas
│   │   ├── models/              # Pydantic structured outputs
│   │   ├── guardrails/          # Galileo guardrail integration
│   │   ├── api/                 # FastAPI endpoints
│   │   └── utils/               # Langfuse setup, LlamaIndex config
│   ├── evals/                   # Evaluation suite + benchmark framework
│   ├── scripts/                 # CLI tools (download filings, build index, run analysis)
│   └── tests/
├── frontend/                    # React + Vite dashboard
├── docs/                        # Architecture deep dives
├── Dockerfile
└── railway.toml
```

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

# Run a full analysis
python scripts/run_analysis.py --ticker AAPL

# Run a focused query
python scripts/run_analysis.py --ticker AAPL --query "What are the main risk factors?"

# Run with specific RAG level
python scripts/run_analysis.py --ticker AAPL --rag-level corrective
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

## Key Design Decisions

**Why 10 agents instead of 1?** — Each agent has a narrow scope with a tailored system prompt, scoring rubric, and tool set. This produces more calibrated scores than asking one model to consider everything at once. Weights reflect how much each dimension should influence the final recommendation.

**Why LangGraph over raw asyncio?** — LangGraph's `Send()` API provides typed state management with reducer functions for parallel output merging. The state graph is inspectable and traceable, and the fan-out pattern maps cleanly to the "route → gather → analyze → synthesize" pipeline.

**Why Corrective RAG?** — Standard vector similarity retrieval fails silently when the indexed documents don't contain relevant information. CRAG grades retrieved chunks and falls back to web search, preventing confident-sounding answers built on irrelevant context.

**Why multi-provider fallback for financial data?** — Free-tier rate limits (FMP: 5 calls/min, Finnhub: 60 calls/min) make single-provider architectures fragile under concurrent agent execution. The fallback chain (Finnhub → FMP → yfinance) ensures at least one provider succeeds.
