# AIRAS v3 — Deep Dive Architecture & Tech Stack Reference

> **AI-Powered Investment Research & Analysis System**
> Complete guide with examples, pros/cons, alternatives, detailed workflows, and measurement strategies.

---

## Table of Contents

1. [System Overview & Purpose](#1-system-overview--purpose)
2. [Architecture Diagram](#2-architecture-diagram)
3. [Tech Stack: Choices, Pros, Cons & Alternatives](#3-tech-stack-choices-pros-cons--alternatives)
4. [Detailed Workflow: Step-by-Step](#4-detailed-workflow-step-by-step)
   - 4.1 Offline: SEC Ingestion Pipeline
   - 4.2 Online: Analysis Request Full Flow
5. [RAG Techniques: All Four Levels](#5-rag-techniques-all-four-levels)
   - 5.1 BasicRetriever
   - 5.2 IntermediateRetriever
   - 5.3 AdvancedRetriever (HyDE + Multi-Query + Reranking)
   - 5.4 CorrectiveRetriever (CRAG)
6. [Agent System: Architecture & Interaction](#6-agent-system-architecture--interaction)
   - 6.1 Agent Catalog
   - 6.2 BaseAgent Tool-Use Loop (with real code walkthrough)
   - 6.3 Agent-to-Agent Communication via LangGraph State
7. [Tool Calling / Function Calling](#7-tool-calling--function-calling)
8. [LangGraph Orchestration](#8-langgraph-orchestration)
9. [Synthesis: Score Aggregation](#9-synthesis-score-aggregation)
10. [Monitoring & Observability (Langfuse + Galileo)](#10-monitoring--observability-langfuse--galileo)
11. [Guardrails](#11-guardrails)
12. [Evaluations & Metrics](#12-evaluations--metrics)
13. [API Endpoints](#13-api-endpoints)
14. [Configuration Reference](#14-configuration-reference)
15. [How to Measure & Monitor: Metrics Guide](#15-how-to-measure--monitor-metrics-guide)
16. [Known Gaps & Improvement Roadmap](#16-known-gaps--improvement-roadmap)
    - 16.0 Concrete Example: Before & After Improvements 1-3
    - 16.1 RAG Improvements (Mintlify ChromaFs-Inspired)
    - 16.2 Evaluation Improvements
    - 16.3 Architecture Improvements
    - 16.4 Priority Summary

---

## 1. System Overview & Purpose

AIRAS v3 answers the question: **"Should I invest in this stock?"** by running 10 specialized AI analysts in parallel, each examining a different dimension (financials, technicals, sentiment, risk, etc.) and synthesizing their findings into a single weighted recommendation.

**Core pipeline in one sentence:**
> A request for `AAPL` triggers retrieval from a vector database of SEC filings, the context is distributed to 10 Claude agents, each agent calls financial APIs and produces a scored output, and a synthesis layer combines all 10 into a final `BUY / HOLD / SELL` decision with a written thesis.

**Why this architecture exists:**
- No single LLM call has enough context window for 10 types of financial analysis simultaneously
- Specialized agents with narrow scopes produce higher-quality outputs than one generalist
- Parallel execution keeps total time under 60 seconds despite running 10 analyses
- Weighted aggregation reflects real-world analyst importance (financials matter more than social sentiment)

---

## 2. Architecture Diagram

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                         CLIENT REQUEST                                         │
│  POST /api/analysis/analyze                                                    │
│  { "ticker": "AAPL", "query": "investment outlook", "rag_level": "corrective"} │
└──────────────────────────────────────┬─────────────────────────────────────────┘
                                       │
                                       ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│                    FastAPI + Uvicorn (backend/src/api/)                        │
│  Async request handling, Pydantic validation, OpenAPI docs auto-generated      │
└──────────────────────────────────────┬─────────────────────────────────────────┘
                                       │
                                       ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│                   LangGraph StateGraph (src/agents/graph.py)                   │
│                                                                                │
│  ┌─────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐    │
│  │  [1] Router     │───▶│  [2] Gather Context  │───▶│  [3] Fan-Out        │    │
│  │  route_query    │    │  gather_context      │    │  (Send API)         │    │
│  │                 │    │                      │    │  5s stagger         │    │
│  │  Classifies     │    │  RAG queries per     │    │  between agents     │    │
│  │  query intent   │    │  active agent        │    └────────┬────────────┘    │
│  │  Selects agents │    │  Builds rag_context  │             │                 │
│  └─────────────────┘    │  dict                │             │                 │
│                         └─────────────────────┘             │                 │
│                                  │                           │ parallel        │
│                    ┌─────────────┘                           │ Send()          │
│                    │                                         ▼                 │
│              RAG Engine                    ┌─────────────────────────────────┐ │
│         (src/rag/retrieval.py)             │   10 Parallel Agent Nodes       │ │
│                    │                       │                                 │ │
│    ┌───────────────▼────────────────────┐  │  financial_analyst    (20%)     │ │
│    │  Supabase PostgreSQL + pgvector    │  │  technical_analyst    (15%)     │ │
│    │  Table: airas_documents            │  │  news_sentiment       (12%)     │ │
│    │  Embeddings: text-embedding-3-small│  │  analyst_ratings      (10%)     │ │
│    │  Index: HNSW (cosine)             │  │  risk_assessment      (10%)     │ │
│    │  Metadata: JSONB (ticker, section) │  │  competitive_analysis (10%)     │ │
│    └────────────────────────────────────┘  │  insider_activity     ( 8%)     │ │
│                                            │  earnings_analysis    ( 7%)     │ │
│                                            │  options_analysis     ( 5%)     │ │
│                                            │  social_sentiment     ( 3%)     │ │
│                                            └────────────────┬────────────────┘ │
│                                                             │                  │
│                                            ┌────────────────▼────────────────┐ │
│                                            │  [5] Synthesis Node             │ │
│                                            │  Weighted score aggregation     │ │
│                                            │  Category scores                │ │
│                                            │  Thesis generation via Claude   │ │
│                                            └─────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────────────┘

Each Agent Node (zoomed in):
┌──────────────────────────────────────────────────────────────────────┐
│  Agent Node (e.g. financial_analyst_node)                            │
│                                                                      │
│  1. Receive: { ticker, rag_context, trace_id }                       │
│  2. Build user message (RAG context + task schema + instructions)    │
│                                                                      │
│  ┌── Claude Tool-Use Loop ─────────────────────────────────────────┐ │
│  │  Iter 0: claude.messages.create(tools=agent.TOOLS)              │ │
│  │    → stop_reason="tool_use" → execute_tool() → append result    │ │
│  │  Iter 1: claude.messages.create(same tools + tool result)       │ │
│  │    → stop_reason="tool_use" → execute_tool() → append result    │ │
│  │  Iter 2: claude.messages.create(same tools + tool result)       │ │
│  │    → stop_reason="end_turn" → parse JSON → AgentOutput          │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                      │
│  3. Galileo validate(summary, rag_context) → hallucination check    │
│  4. Langfuse log scores                                              │
│  5. Return AgentOutput to LangGraph state                           │
└──────────────────────────────────────────────────────────────────────┘

Financial API fallback chain (inside execute_tool):
┌──────────────┐    failure    ┌──────────┐    failure    ┌──────────┐
│   Finnhub    │ ────────────▶ │   FMP    │ ────────────▶ │ yfinance │
│ 60 calls/min │               │ 5 c/min  │               │ free     │
└──────────────┘               └──────────┘               └──────────┘

Observability Layer (wraps entire pipeline):
┌─────────────────────────┐    ┌──────────────────────────────────┐
│  Langfuse               │    │  Galileo AI                      │
│  - Full trace hierarchy │    │  - Hallucination detection        │
│  - Token costs          │    │  - PII detection + redaction      │
│  - Tool call latencies  │    │  - Toxicity scoring               │
│  - Agent scores         │    │  - Context relevance evaluation   │
└─────────────────────────┘    └──────────────────────────────────┘
```

---

## 3. Tech Stack: Choices, Pros, Cons & Alternatives

### 3.1 Python 3.11+

**Why chosen:** Dominant language for AI/ML tooling. `asyncio` native, extensive financial libraries.

| Pros | Cons | Alternatives |
|------|------|-------------|
| Richest AI/ML ecosystem | GIL limits true multi-threading | TypeScript (Node.js) — less AI tooling |
| asyncio for concurrent agents | Slower than compiled languages | Go — faster, but immature AI libs |
| Pydantic, LangGraph, LlamaIndex all Python-first | Dynamic typing can hide bugs | Java/Kotlin — mature but verbose |
| Rapid prototyping | — | — |

---

### 3.2 FastAPI + Uvicorn

**Why chosen:** Async-native HTTP framework with automatic OpenAPI docs and tight Pydantic integration. Uvicorn is the high-performance ASGI server.

**Example endpoint:**
```python
@router.post("/analyze", response_model=InvestmentRecommendation)
async def analyze(request: AnalysisRequest) -> InvestmentRecommendation:
    graph = get_analysis_graph()
    result = await graph.ainvoke({"ticker": request.ticker, ...})
    return result["recommendation"]
```

| Pros | Cons | Alternatives |
|------|------|-------------|
| Auto-generates OpenAPI/Swagger docs | Newer than Django/Flask — less community resources | Flask — simpler but no async, no auto-docs |
| Pydantic v2 validation built-in | Async bugs harder to debug | Django REST — batteries included, heavier |
| ~3x faster than Flask for I/O-bound tasks | — | Starlette — FastAPI is built on it |
| Dependency injection system | — | — |

---

### 3.3 LangGraph

**Why chosen:** Provides a `StateGraph` with typed state (`TypedDict`), built-in parallel fan-out via `Send()`, and automatic state aggregation via reducers. Critical for running 10 agents in parallel and collecting their outputs.

**Example — parallel fan-out:**
```python
# In fan_out_node: create parallel Send() for each active agent
def fan_out_node(state: AnalysisState):
    sends = []
    for i, agent_name in enumerate(state["active_agents"]):
        time.sleep(i * 5)  # Stagger to avoid rate limit bursts
        sends.append(Send(agent_name, state))
    return sends

# State reducer auto-merges parallel results:
class AnalysisState(TypedDict):
    agent_outputs: Annotated[list[AgentOutput], operator.add]
    #                         ↑ Each parallel agent appends to this list
    #                           operator.add is the reducer function
```

| Pros | Cons | Alternatives |
|------|------|-------------|
| Typed state with `TypedDict` → IDE support | Steeper learning curve than bare asyncio | LangChain LCEL — less suited for parallel fan-out |
| `Send()` API for clean parallel execution | Debugging requires understanding graph execution model | Raw `asyncio.gather()` — works but no state management |
| `operator.add` reducer handles concurrent writes safely | — | Celery — full task queue, overkill for this use case |
| Handles retry and error propagation at node level | — | Temporal — production workflow engine, more complex |

---

### 3.4 LlamaIndex

**Why chosen:** Provides a complete ingestion pipeline (chunking → metadata extraction → embedding → storage) and `VectorIndexRetriever` with composable metadata filters. Avoids writing raw SQL for every retrieval pattern.

**Example — ingestion pipeline:**
```python
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=512, chunk_overlap=50),
        SECMetadataTransform(),           # Custom: extracts ticker, section, date
        OpenAIEmbedding(model="text-embedding-3-small"),
    ],
    vector_store=PGVectorStore(...)
)
pipeline.run(documents=sec_documents)
```

**Example — filtered retrieval:**
```python
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=5,
    filters=MetadataFilters(filters=[
        MetadataFilter(key="ticker", value="AAPL"),
        MetadataFilter(key="section", value=["risk_factors", "cash_flow"],
                      operator=FilterOperator.IN),
    ])
)
nodes = retriever.retrieve("What are Apple's liquidity risks?")
```

| Pros | Cons | Alternatives |
|------|------|-------------|
| Abstracts ingestion pipeline cleanly | Abstraction leaks when debugging low-level vector issues | LangChain — more popular but heavier deps |
| Built-in pgvector integration | Version upgrades break APIs frequently | Direct pgvector SQL — full control but more code |
| Composable metadata filters | Opinionated about document structure | Haystack — good alternative, similar scope |
| Supports HyDE, multi-query natively | — | ChromaDB/Pinecone direct — no pipeline abstraction |

---

### 3.5 Claude (Anthropic) as Primary LLM

**Why chosen:** Superior structured JSON output reliability, handles long SEC filing contexts well, reliable tool-use in multi-turn loops, lower hallucination rate for financial facts.

**Example — tool-use call:**
```python
response = client.messages.create(
    model="claude-opus-4-5-20251101",
    max_tokens=2048,
    temperature=0.2,          # Low temp for consistent structured output
    system=SYSTEM_PROMPT,     # Expert persona + scoring rubric + few-shot
    tools=agent_tools,        # Only this agent's permitted tools
    messages=conversation_history
)
# response.stop_reason == "tool_use" → agent wants to call a tool
# response.stop_reason == "end_turn"  → agent is done, parse JSON output
```

| Pros | Cons | Alternatives |
|------|------|-------------|
| Best-in-class structured JSON output | Higher cost than GPT-3.5/Gemini Flash | GPT-4o — comparable quality, different pricing |
| Excellent long-context comprehension | Rate limits require retry logic | Gemini 1.5 Pro — 1M token context, cheaper |
| Reliable tool-use in multi-turn loops | — | Llama 3 (self-hosted) — no API cost, but needs infra |
| Low hallucination rate on financial data | — | Mistral — good quality, EU-hosted option |

---

### 3.6 OpenAI text-embedding-3-small for Embeddings

**Why chosen:** 1536-dimensional embeddings with excellent semantic understanding. Low cost at $0.02/1M tokens. No need for 3-large for financial text retrieval.

**Example:**
```python
# During ingestion
embed_model = OpenAIEmbedding(model="text-embedding-3-small", dimensions=1536)
# Query: "Apple revenue growth FY2023"
# → [0.023, -0.147, 0.891, ...] (1536 floats)
# Cosine similarity search in pgvector finds semantically similar SEC chunks
```

| Pros | Cons | Alternatives |
|------|------|-------------|
| Excellent quality at low cost | Requires OpenAI dependency | text-embedding-3-large — better quality, 3x cost |
| Widely benchmarked | Not open-source / self-hostable | Cohere embed-v3 — competitive quality |
| Supports Matryoshka (truncatable dimensions) | API latency for large batches | sentence-transformers (local) — free, slightly lower quality |
| OpenAI ecosystem consistency | — | BGE-M3 — open-source, multilingual |

---

### 3.7 PostgreSQL + pgvector (Supabase)

**Why chosen:** Keeps all data — vector store, metadata, application data — in a single managed Postgres instance. HNSW index gives sub-100ms vector retrieval. JSONB metadata columns allow flexible filtering without schema migrations.

**Example — what the table looks like:**
```sql
-- Table: airas_documents
-- id: uuid, content: text, embedding: vector(1536), metadata: jsonb
SELECT content, metadata->>'section', 1 - (embedding <=> query_vector) as score
FROM airas_documents
WHERE metadata->>'ticker' = 'AAPL'
  AND metadata->>'section' = ANY(ARRAY['risk_factors', 'cash_flow'])
ORDER BY embedding <=> query_vector
LIMIT 5;

-- HNSW index makes this sub-100ms even with 100k+ documents:
CREATE INDEX ON airas_documents USING hnsw (embedding vector_cosine_ops);
-- GIN index makes metadata filtering fast:
CREATE INDEX ON airas_documents USING gin (metadata);
```

| Pros | Cons | Alternatives |
|------|------|-------------|
| Single infrastructure piece (no separate vector DB) | Vector search slower than specialized DBs at 10M+ scale | Pinecone — fastest vector search, managed, no SQL |
| Full SQL expressiveness + vector search | pgvector HNSW less mature than FAISS | Weaviate — feature-rich, adds infra complexity |
| JSONB for flexible metadata without migrations | — | Qdrant — open-source, fast, more config |
| Supabase free tier covers development | — | Chroma — simplest, no auth, dev-only |
| Familiar to any backend engineer | — | Milvus — enterprise-grade, most complex |

**At current scale (10K–500K documents):** pgvector + HNSW is perfectly adequate.
**At 10M+ documents:** Consider migrating to Pinecone or Qdrant for retrieval, keeping Postgres for metadata.

---

### 3.8 Finnhub as Primary Financial Data Provider

**Why chosen (promoted from secondary):** 60 API calls/minute on the free tier vs FMP's 5/minute. With 10 parallel agents each making 2-3 tool calls, FMP caused systematic `429` rate limit errors. Finnhub's higher limit eliminated this bottleneck.

**Real problem that forced this change:**
```
# Before (FMP as primary):
# 10 agents × 2 tool calls = 20 calls in ~5 seconds
# FMP rate limit: 5/min = ~0.08 calls/second
# Result: ~75% of tool calls failed with 429, agents returned null data

# After (Finnhub as primary):
# Finnhub rate limit: 60/min = 1 call/second
# 20 calls spread over ~5 seconds with 5s stagger → no rate limit failures
```

| Pros | Cons | Alternatives |
|------|------|-------------|
| 60 calls/min free tier | Less comprehensive than FMP for fundamentals | FMP — better fundamental data, lower rate limit |
| Real-time prices and news | No options data | Alpha Vantage — free but very low limits |
| Good insider transaction data | — | Polygon.io — excellent, paid |
| WebSocket support for streaming | — | yfinance — free, unlimited, but unofficial/fragile |

---

### 3.9 Langfuse for LLM Tracing

**Why chosen:** Purpose-built for LLM observability. Provides hierarchical trace spans, token cost tracking, quality scoring, and a web dashboard. Open-source with a managed cloud tier.

**Example trace structure in Langfuse UI:**
```
Trace: AAPL_analysis_2026-03-03        [total: 47s, $0.84, 42,000 tokens]
  ├── route_query                       [12ms]
  ├── gather_context                    [3.2s]
  │     ├── rag_financial_analyst       [620ms, 5 chunks, confidence=82%]
  │     ├── rag_technical_analyst       [180ms, 5 chunks, confidence=91%]
  │     └── ...
  ├── financial_analyst                 [8.4s, $0.19, 12,600 tokens]
  │     ├── llm_call_0                  [2.1s, 3,420 in / 156 out]
  │     ├── tool: get_stock_price       [340ms, Finnhub]
  │     ├── tool: calculate_financial_ratio [280ms, FMP]
  │     ├── llm_call_1                  [3.8s, 4,100 in / 312 out]
  │     └── score: 0.72                 [comment: "Strong FCF growth, improving margins"]
  ├── technical_analyst                 [5.1s, $0.11]
  └── synthesis                         [4.2s, $0.09]
        └── recommendation: BUY (0.58)
```

| Pros | Cons | Alternatives |
|------|------|-------------|
| Purpose-built for LLM tracing | Not a general-purpose APM | Arize AI — ML observability, less LLM-specific |
| Hierarchical spans with costs | Requires setup of SDK | Weights & Biases — broad ML tracking |
| Open-source (self-hostable) | — | LangSmith — LangChain ecosystem lock-in |
| Web dashboard out of the box | — | OpenTelemetry — vendor-neutral, more manual |
| Dataset & annotation features | — | Helicone — simpler proxy-based approach |

---

### 3.10 Galileo AI for Guardrails & Evaluation

**Why chosen:** Provides automated hallucination detection (groundedness), PII detection with redaction, and context relevance scoring — all per LLM call. No need to build custom evaluation infrastructure.

**Example — hallucination detection:**
```python
# After each agent produces its summary:
result = check_hallucination(
    response="Apple's revenue grew 15% in FY2023",   # Agent claim
    context=["Apple reported net sales of $383,285M, up 5.5% YoY"]  # RAG source
)
# result = {
#   "is_grounded": False,         # 15% ≠ 5.5%
#   "score": 0.31,                # Low groundedness
#   "flagged_claims": ["revenue grew 15%"]
# }
# → Logs warning, agent confidence penalized
```

| Pros | Cons | Alternatives |
|------|------|-------------|
| Purpose-built hallucination detection | Proprietary, adds external dependency | RAGAS — open-source, batch-oriented |
| PII detection + auto-redaction | Additional cost per call | Guardrails AI — open-source, more rule-based |
| Real-time (per-call) evaluation | — | Custom LLM-as-judge with GPT-4 — flexible but DIY |
| Context relevance scoring | — | TruLens — open-source, TruLens Evals |
| Dashboard for reviewing flagged outputs | — | DeepEval — open-source alternative |

---

### 3.11 RAGAS for RAG Evaluation

**Why chosen:** The standard open-source framework for evaluating RAG pipelines. Provides faithfulness, answer relevancy, context precision, and context recall as quantitative metrics.

**Example:**
```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

result = evaluate(
    dataset=Dataset.from_dict({
        "question": ["What was Apple's FY2023 revenue?"],
        "answer": ["Apple's revenue was $383.3B"],
        "contexts": [["Apple net sales $383,285M for FY2023..."]],
        "ground_truth": ["$383.3 billion"]
    }),
    metrics=[faithfulness, answer_relevancy, context_precision]
)
# Output:
# faithfulness: 0.95      (answer is supported by context)
# answer_relevancy: 0.98  (answer addresses the question)
# context_precision: 0.80 (retrieved context is relevant)
```

| Pros | Cons | Alternatives |
|------|------|-------------|
| Standard benchmark for RAG quality | Uses LLM internally → costs money per eval run | DeepEval — similar metrics, more customizable |
| Works with any LLM/embedding | Slow for large test sets | ARES — LLM-free evaluation |
| Easy to integrate with pytest | Metrics can be misleading without golden set | Custom metrics — full control |
| Actively maintained | — | TruLens — similar scope |

---

### 3.12 Tenacity for Retry Logic

**Why chosen:** Declarative retry with exponential backoff. Single decorator handles all rate limit scenarios without manual `try/except/sleep` chains.

**Example from base_agent.py:**
```python
# Actual implementation in base_agent.py:
def _call_with_retry(self, client, kwargs):
    for attempt in range(MAX_RETRIES + 1):  # MAX_RETRIES = 3
        try:
            return client.messages.create(**kwargs)
        except anthropic.RateLimitError:
            if attempt < MAX_RETRIES:
                delay = RETRY_BASE_DELAY * (2 ** attempt)  # 30s, 60s, 120s
                logger.info(f"Rate limited, retrying in {delay}s")
                time.sleep(delay)
            else:
                raise
```

| Pros | Cons | Alternatives |
|------|------|-------------|
| Simple, declarative | Not async-native (requires asyncio wrapper) | backoff — similar, slightly more concise |
| Handles many exception types | — | httpx built-in retry — only for HTTP calls |
| Configurable backoff strategies | — | Manual try/except loops — verbose |

---

## 4. Detailed Workflow: Step-by-Step

### 4.1 Offline: SEC Ingestion Pipeline

This runs once per ticker to build the knowledge base. Triggered via `POST /api/sec/download` then `POST /api/sec/index`, or via `python scripts/download_sec_filings.py --ticker AAPL`.

```
STEP 1: Download SEC Filings
───────────────────────────────────────────────────────────────────
Command: python scripts/download_sec_filings.py --ticker AAPL
Library: sec-edgar-downloader
         Uses: SEC_USER_EMAIL as User-Agent header (EDGAR requirement)

What it downloads:
  - Form 10-K (annual reports)
  - Form 10-Q (quarterly reports)
  - Last 4 years by default

Output structure:
  data/raw/AAPL/
    ├── 10-K/
    │     ├── 2023-11-03/  (fiscal year end date)
    │     │     └── full-submission.txt  (raw HTML + XBRL)
    │     └── 2022-10-28/
    └── 10-Q/
          ├── 2024-08-02/
          └── 2024-05-03/

Example: Apple's 10-K is ~200 pages of HTML with:
  - Item 1: Business description
  - Item 1A: Risk factors (typically 30-50 risks listed)
  - Item 7: Management Discussion & Analysis (MD&A)
  - Item 8: Financial Statements (income statement, balance sheet, cash flows)

STEP 2: Parse HTML → Plain Text Sections
───────────────────────────────────────────────────────────────────
Library: BeautifulSoup4 + lxml

Process:
  1. Strip HTML tags, table markup, XBRL metadata
  2. Identify SEC sections by heading patterns:
     "ITEM 1A." → section="risk_factors"
     "ITEM 7."  → section="management_discussion"
     "ITEM 8."  → section="financial_statements"
  3. Preserve section structure in plain text

Example raw input:
  <p><b>ITEM 1A. RISK FACTORS</b></p>
  <p>The following risk factors may affect our business...</p>

Example output:
  "ITEM 1A. RISK FACTORS\n\nThe following risk factors may affect our business..."

STEP 3: Chunk → Extract Metadata → Embed → Store
───────────────────────────────────────────────────────────────────
Library: LlamaIndex IngestionPipeline

Sub-step 3a: SentenceSplitter
  chunk_size = 512 tokens
  chunk_overlap = 50 tokens

  Input: "The following risk factors may affect our business. We operate in
          highly competitive markets. Our results may be affected by global
          economic conditions. We rely on third-party manufacturers..."

  Output chunk 1 (512 tokens): "The following risk factors...global economic conditions."
  Output chunk 2 (512 tokens, 50 overlap): "...global economic conditions. We rely on third-party..."

  Why 512 tokens?
  - Large enough to contain a complete risk disclosure or financial paragraph
  - Small enough that retrieval returns specific, focused content
  - 50 token overlap prevents splitting a concept across chunk boundaries

Sub-step 3b: SECMetadataTransform (custom)
  Input: chunk text + filename "AAPL_10-K_2023-11-03.txt"

  Output metadata (stored as JSONB):
  {
    "ticker": "AAPL",
    "filing_type": "10-K",
    "date": "2023-11-03",
    "section": "risk_factors",          ← detected from "ITEM 1A" heading
    "fiscal_period": "FY2023",
    "content_type": "risk_disclosure",
    "metrics": {                         ← extracted if numbers present
      "mentioned_revenue": true,
      "mentioned_debt": false
    }
  }

Sub-step 3c: OpenAI Embeddings
  model: text-embedding-3-small
  dimensions: 1536

  Input: chunk text
  Output: [0.023, -0.147, 0.891, ...] (1536 floats)
  Cost: ~$0.02 per 1M tokens

  Example: "Apple faces risks from supply chain disruptions in China"
  → 1536-dimensional vector that places this near other supply chain risk text
  → Far from text about revenue or dividends

Sub-step 3d: Store in pgvector
  INSERT INTO airas_documents (content, embedding, metadata)
  VALUES (chunk_text, embedding_vector, metadata_jsonb)

  After full AAPL 10-K + 10-Q indexing:
  ~2,000-4,000 chunks per ticker
  ~5MB of embeddings (4,000 × 1,536 × 4 bytes)
  ~2MB of text content
```

### 4.2 Online: Analysis Request Full Flow

```
REQUEST ARRIVES:
  POST /api/analysis/analyze
  { "ticker": "AAPL", "rag_level": "corrective" }

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NODE 1: route_query_node
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
File: src/agents/router.py

Input:  { ticker: "AAPL", query: None, rag_level: "corrective" }

Process:
  1. If no query → run all 10 agents ("full" mode)
  2. If query present → classify intent:
     "What are the risk factors?" → risk_assessment, competitive_analysis
     "How is the technical setup?"→ technical_analyst, options_analysis
     "Is there insider buying?"   → insider_activity

  3. Initialize Langfuse trace:
     trace_id = langfuse.trace(name="AAPL_analysis", metadata={...})

Output: { active_agents: ["financial_analyst", "technical_analyst", ...all 10],
          mode: "full",
          trace_id: "trace_abc123" }

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NODE 2: gather_context_node
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
File: src/agents/context.py

For each active agent, run all its RAG_QUERIES:

  financial_analyst.RAG_QUERIES = [
    "Revenue growth trends for {ticker}",
    "Operating margins and profitability for {ticker}",
    "Balance sheet debt and liquidity for {ticker}",
    "Cash flow from operations for {ticker}",
  ]
  → substituted: "Revenue growth trends for AAPL"

  Each query runs CorrectiveRetriever (because rag_level="corrective"):

  Query 1: "Revenue growth trends for AAPL"
    → Initial retrieval: 5 chunks from airas_documents WHERE ticker='AAPL'
    → Top vector score: 0.76, avg: 0.71
    → Hybrid check: 0.76 >= 0.72 AND 0.71 >= 0.70 → HIGH CONFIDENCE → skip grading
    → Return 5 chunks labeled [RELEVANT] (synthetic grades)

  Query 2: "Operating margins and profitability for AAPL"
    → Initial retrieval: 5 chunks
    → Top score: 0.63, avg: 0.58 → NOT high confidence
    → Grade 5 chunks via Claude Haiku (batch call):
      Chunk 1: "Apple's gross margin was 44.1%..." → RELEVANT (score=1.0)
      Chunk 2: "The company sells iPhone, Mac, iPad..." → PARTIAL (score=0.5)
      Chunk 3: "Competition from Android devices..." → PARTIAL (score=0.5)
      Chunk 4: "Apple's headquarters in Cupertino..." → IRRELEVANT (score=0.0)
      Chunk 5: "Net sales of services segment grew 16%..." → RELEVANT (score=1.0)
    → usable_ratio = (2 + 2×0.5) / 5 = 0.60 >= 0.40 threshold → ACCEPT

  Context string built for financial_analyst:
    "[Retrieval confidence: 76%]

     [RELEVANT]
     Apple's gross margin was 44.1% in FY2023, up from 43.3% in FY2022...
     Net sales of services grew 16% to $85.2 billion...

     [PARTIAL]
     The company sells iPhone, Mac, iPad, Apple Watch and services...
     Competition from Android devices remains a factor...

     [IRRELEVANT]
     Apple's headquarters is located in Cupertino, California..."

Output: rag_context = {
  "financial_analyst": "[Retrieval confidence: 76%]\n\n[RELEVANT]\n...",
  "technical_analyst": "[Retrieval confidence: 91%]\n\n[RELEVANT]\n...",
  "risk_assessment": "[Retrieval confidence: 68%]\n\n[RELEVANT]\n...",
  ...10 entries total
}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NODE 3: fan_out
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
File: src/agents/graph.py

Creates parallel Send() for each agent with 5s stagger:
  t=0s:  Send("financial_analyst", state)
  t=5s:  Send("technical_analyst", state)
  t=10s: Send("news_sentiment", state)
  ...
  t=45s: Send("social_sentiment", state)

Why stagger? Without it, all 10 agents send their first Claude call
simultaneously. Even with Finnhub's 60/min limit, 10 agents making
2-3 tool calls = 20-30 simultaneous API calls → guaranteed 429 errors.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NODE 4: financial_analyst_node (typical agent, parallel with others)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
File: src/agents/base_agent.py

STEP 4a: Build user message
  user_message = f"""
  === SEC FILING DATA (RAG) ===

  [Retrieval confidence: 76%]

  [RELEVANT]
  Apple's gross margin was 44.1% in FY2023...
  Net sales of services grew 16% to $85.2B...

  [PARTIAL]
  Apple sells iPhone, Mac, iPad, Apple Watch and services...

  === YOUR TASK ===

  Analyze AAPL. Reason step-by-step:
  1. Review the data provided above (if any)
  2. Use your available tools to get current market data
  3. Identify 1-3 key strengths and 1-3 key weaknesses
  4. Assign a score from -1.0 (very bearish) to +1.0 (very bullish)
  5. Assign a confidence from 0.0 to 1.0

  Respond with ONLY valid JSON matching this schema: {...}
  """

STEP 4b: First Claude call
  claude.messages.create(
    model="claude-opus-4-5-20251101",
    temperature=0.2,
    system="You are a senior Financial Analyst with 20 years of experience...",
    tools=[calculate_financial_ratio, compare_companies, get_stock_price],
    messages=[{role: "user", content: user_message}]
  )

  → Response stop_reason = "tool_use"
  → Claude wants to call: get_stock_price({"ticker": "AAPL"})

  Log to Langfuse: llm_call_0, 3420 input tokens, 156 output tokens, 2.1s

STEP 4c: Execute tool
  result = execute_tool("get_stock_price", {"ticker": "AAPL"})

  Tries Finnhub first:
    GET https://finnhub.io/api/v1/quote?symbol=AAPL&token=...
    Response: {"c": 189.30, "h": 191.20, "l": 187.50, "o": 188.00, ...}
    ✓ Success (Finnhub is primary, 60 calls/min)

  Returns: {"price": 189.30, "change_pct": 0.68, "volume": 42_000_000, ...}

  Log to Langfuse: tool_call "get_stock_price", 340ms, Finnhub provider

STEP 4d: Append tool result, second Claude call
  messages = [
    {role: "user", content: [user_message]},
    {role: "assistant", content: [tool_use_block]},
    {role: "user", content: [{type: "tool_result", content: '{"price": 189.30...}'}]}
  ]

  claude.messages.create(same params, updated messages)
  → stop_reason = "tool_use"
  → Claude wants: calculate_financial_ratio({"ticker": "AAPL", "ratio": "pe_ratio"})

STEP 4e: Execute second tool
  execute_tool("calculate_financial_ratio", {"ticker": "AAPL", "ratio": "pe_ratio"})

  Tries FMP (secondary for ratios):
    GET https://financialmodelingprep.com/api/v3/ratios/AAPL?...
    Response: {"peRatio": 29.4, "debtToEquity": 1.73, "roe": 1.47, ...}
    ✓ Success

  Returns: {"pe_ratio": 29.4, "debt_to_equity": 1.73, "roe": 147%}

STEP 4f: Third Claude call — end turn
  messages (now 5 messages: user, assistant, tool_result, assistant, tool_result)

  claude.messages.create(same params, updated messages)
  → stop_reason = "end_turn"
  → Response text contains JSON:

  {
    "agent_name": "financial_analyst",
    "ticker": "AAPL",
    "score": 0.72,
    "confidence": 0.85,
    "metrics": {
      "pe_ratio": 29.4,
      "gross_margin": 44.1,
      "services_growth": 16.0,
      "current_price": 189.30
    },
    "strengths": [
      "Services segment growing 16% YoY with high margins",
      "Gross margin expansion to 44.1% from 43.3%",
      "Strong free cash flow generation supports buybacks"
    ],
    "weaknesses": [
      "Hardware revenue flat, iPhone growth stagnating",
      "China market exposure amid geopolitical tensions"
    ],
    "summary": "Apple shows strong financial health with margin expansion and growing high-margin services.",
    "sources": ["SEC 10-K FY2023", "Finnhub price data", "FMP financial ratios"]
  }

STEP 4g: Galileo validation
  galileo.validate_agent_output(
    summary="Apple shows strong financial health...",
    rag_context="Apple gross margin 44.1%... services grew 16%..."
  )
  → groundedness: 0.92 ✓ (summary matches SEC data)
  → pii_detected: None ✓

STEP 4h: Return AgentOutput
  AgentOutput(
    agent_name="financial_analyst",
    ticker="AAPL",
    score=0.72,
    confidence=0.85,
    ...
  )
  → LangGraph operator.add appends to state["agent_outputs"]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
(other 9 agents run in parallel, same pattern)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Typical results after all 10 agents complete:
  financial_analyst:    score=0.72, confidence=0.85
  technical_analyst:    score=0.45, confidence=0.78   ← slightly overbought RSI
  news_sentiment:       score=0.60, confidence=0.72
  analyst_ratings:      score=0.65, confidence=0.88   ← average PT above current
  risk_assessment:      score=-0.15, confidence=0.70  ← China risk flagged
  competitive_analysis: score=0.40, confidence=0.65
  insider_activity:     score=0.55, confidence=0.80   ← recent CEO buys
  earnings_analysis:    score=0.68, confidence=0.82
  options_analysis:     score=0.20, confidence=0.60   ← high IV, mixed signals
  social_sentiment:     score=0.35, confidence=0.55

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NODE 5: synthesis_node
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
File: src/agents/synthesis.py

STEP 5a: Calculate overall score (confidence-weighted)
  For each agent:
    w = AGENT_WEIGHTS[agent_name]        # e.g., 0.20 for financial_analyst
    contribution = score × confidence × w

  financial_analyst:    0.72 × 0.85 × 0.20 = 0.1224
  technical_analyst:    0.45 × 0.78 × 0.15 = 0.0527
  news_sentiment:       0.60 × 0.72 × 0.12 = 0.0518
  analyst_ratings:      0.65 × 0.88 × 0.10 = 0.0572
  risk_assessment:     -0.15 × 0.70 × 0.10 = -0.0105
  competitive_analysis: 0.40 × 0.65 × 0.10 = 0.0260
  insider_activity:     0.55 × 0.80 × 0.08 = 0.0352
  earnings_analysis:    0.68 × 0.82 × 0.07 = 0.0390
  options_analysis:     0.20 × 0.60 × 0.05 = 0.0060
  social_sentiment:     0.35 × 0.55 × 0.03 = 0.0058

  total_weighted = sum = 0.3856
  total_weight   = sum(confidence × weight) = 0.6833
  overall_score  = 0.3856 / 0.6833 = 0.565

  → 0.565 >= 0.20 → recommendation = "BUY"

STEP 5b: Category scores
  financial_score = (financial_analyst×0.20 + earnings×0.07) / (0.20+0.07)
                  = confidence-weighted average = 0.71
  technical_score  = 0.39
  sentiment_score  = 0.58
  risk_score       = 0.27 (dragged down by risk_assessment's -0.15)

STEP 5c: Generate thesis via Claude
  summaries = """
  - financial_analyst (0.72): Apple shows strong financial health with margin expansion
  - technical_analyst (0.45): RSI at 68 suggests mild overbought, trend remains bullish
  - news_sentiment (0.60): Recent positive coverage on Vision Pro and India expansion
  - analyst_ratings (0.65): 32 of 40 analysts rate Buy, avg PT $210 vs current $189
  - risk_assessment (-0.15): Significant China supply chain and regulatory exposure
  - ...
  """

  Claude generates:
  {
    "thesis": "Apple presents a compelling BUY opportunity driven by its expanding
               high-margin services segment and strong shareholder returns program.
               While hardware growth remains muted and China exposure represents
               a material risk, the overall financial health and analyst consensus
               support a constructive outlook at current valuation levels.",
    "bullish_factors": [
      "Services revenue growing 16% YoY with 70%+ gross margins",
      "32/40 analysts rate Buy with average price target of $210",
      "Gross margin expansion from 43.3% to 44.1% demonstrates pricing power"
    ],
    "bearish_factors": [
      "iPhone revenue growth stagnating, hardware dependence remains",
      "RSI approaching overbought territory",
      "China supply chain and regulatory risks increasing"
    ],
    "risks": [...]
  }

STEP 5d: Galileo thesis validation
  check_hallucination(thesis, context=all_agent_summaries)
  → groundedness: 0.88 ✓ (thesis claims supported by agent analyses)
  check_pii(thesis) → no PII ✓

OUTPUT: InvestmentRecommendation
  {
    "ticker": "AAPL",
    "recommendation": "BUY",
    "overall_score": 0.565,
    "confidence": 0.735,
    "financial_score": 0.71,
    "technical_score": 0.39,
    "sentiment_score": 0.58,
    "risk_score": 0.27,
    "bullish_factors": [...],
    "bearish_factors": [...],
    "thesis": "Apple presents a compelling BUY opportunity...",
    "analysis_time_seconds": 47.3,
    "num_agents": 10
  }
```

---

## 5. RAG Techniques: All Four Levels

### 5.1 BasicRetriever — Level 1

**What it does:** Pure cosine similarity search. No filters, no rewriting.

**Code:**
```python
class BasicRetriever:
    def retrieve(self, query, top_k=5, ticker=None, sections=None):
        retriever = VectorIndexRetriever(index=self.index, similarity_top_k=top_k)
        return retriever.retrieve(query)   # ticker/sections ignored
```

**Example:**
```
Query: "Apple revenue growth"
Retrieved (from ANY ticker in database):
  - [score=0.89] "Apple net sales were $383B, up 5.5% YoY..."  ← correct
  - [score=0.87] "Microsoft revenue grew 18% to $212B..."       ← WRONG TICKER
  - [score=0.85] "Amazon's AWS revenue growth accelerated..."   ← WRONG TICKER
  - [score=0.84] "Apple iPhone revenue declined 3%..."          ← correct
  - [score=0.83] "Tesla revenue grew 8% in fiscal year..."      ← WRONG TICKER
```

| Pros | Cons |
|------|------|
| Fastest (~50ms) | Ignores ticker — retrieves any company's data |
| Zero LLM overhead | No query enhancement |
| Good for benchmarking retrieval quality | Poor precision for multi-company databases |

**Alternatives:** No real alternatives at this level — it's the baseline.

---

### 5.2 IntermediateRetriever — Level 2 (Default)

**What it does:** Adds synonym expansion to the query, then tries metadata filtering in priority order: (ticker + section) → (ticker only) → (no filter).

**Code:**
```python
EXPANSIONS = {
    "revenue": "revenue net sales total revenue",
    "debt": "total debt long-term debt term debt borrowings",
    "profit": "net income earnings profit net earnings",
    "margin": "gross margin operating margin profit margin",
    "eps": "earnings per share diluted EPS basic EPS",
    # ... 7 more terms
}

def retrieve(self, query, top_k=5, ticker=None, sections=None):
    enhanced_query = self._rewrite_query(query)
    # "Apple margin growth" → "Apple margin growth gross margin operating margin profit margin"

    for filter_set in self._filter_fallback_chain(ticker, sections):
        try:
            retriever = VectorIndexRetriever(index=self.index,
                                            similarity_top_k=top_k,
                                            filters=filter_set)
            return retriever.retrieve(enhanced_query)
        except Exception:
            continue  # Try next filter level

    # Final fallback: no filters
    return VectorIndexRetriever(index=self.index, similarity_top_k=top_k)\
               .retrieve(enhanced_query)
```

**Example:**
```
Query: "Apple margin growth"
↓ Query rewriting:
Enhanced: "Apple margin growth gross margin operating margin profit margin"

↓ Attempt 1: ticker=AAPL, section IN [income_statement, cash_flow]
  → Retrieved 5 chunks, all from Apple's income statement sections ✓

vs. without enhancement:
Query: "Apple margin growth"  (no expansion)
  → Might miss chunks that say "operating margin" without using "margin growth"
  → Enhancement improves recall by ~15-20% on financial queries
```

**Real fallback example:**
```
Query about insider trades for a company not yet indexed:
  Attempt 1: ticker=TSLA, section=insider_transactions → 0 results
  Attempt 2: ticker=TSLA                              → 0 results
  Attempt 3: no filter                               → 3 results (from any company)
  → Returns 3 results with warning logged
```

| Pros | Cons |
|------|------|
| ~100-200ms, still fast | Synonym list is predefined, not dynamic |
| Eliminates cross-ticker contamination | Fallback to no-filter still possible if ticker not indexed |
| Graceful degradation via fallback chain | — |

**Alternatives:**
- **LLM query rewriting** (instead of static synonyms) — better but adds LLM latency
- **BM25 + vector hybrid** — combines keyword and semantic search, better recall

---

### 5.3 AdvancedRetriever — Level 3

**What it does:** Combines multi-query generation, HyDE (Hypothetical Document Embeddings), and Claude cross-encoder reranking for maximum retrieval accuracy.

**Full workflow:**

```
Original Query: "What are Apple's main liquidity risks?"

STEP 1: Multi-query generation (Claude call)
  Claude generates 3 variations:
  [
    "Apple cash position and liquidity concerns",
    "AAPL short-term debt obligations and credit facilities",
    "Apple risk factors related to financial flexibility"
  ]

STEP 2: HyDE — generate hypothetical document (Claude call)
  Prompt: "Write a brief factual paragraph answering this financial question:
           What are Apple's main liquidity risks?"

  Claude writes:
  "Apple's liquidity risks stem from its significant commercial paper program
   and term debt obligations. As of Q3 2024, Apple held $51.5B in cash and
   investments against $95B in total debt. The company manages liquidity risk
   through its revolving credit facilities and strong free cash flow generation
   of approximately $90B annually. Near-term risks include refinancing $13B in
   debt maturing within 12 months..."

  → This hypothetical text is embedded and used as a 4th query
  → Why? The embedding of an answer-like text matches better to the vector
     space of actual SEC documents than the embedding of a question

STEP 3: Retrieve with all 4 queries (each via IntermediateRetriever)
  Query 1 (original):   → 5 chunks
  Query 2 (variation 1): → 5 chunks
  Query 3 (variation 2): → 5 chunks
  Query 4 (HyDE):        → 5 chunks
  Total before dedup:     20 chunks
  After dedup by node_id: ~12-15 unique chunks

STEP 4: Claude cross-encoder reranking
  Prompt to Claude:
  "Given query: 'What are Apple's main liquidity risks?'
   Rank these 14 chunks by relevance. Return ONLY indices as JSON array.
   [0] 'Apple's cash and investments totaled $162B...'
   [1] 'Apple faces intense competition in all segments...'
   [2] 'Apple's total term debt is $95.3B with $13B maturing...'
   ..."

  Claude returns: [2, 0, 9, 4, 7, ...]  ← by relevance rank

  Final output: top 5 chunks in order of relevance
```

**Example — why HyDE helps:**
```
Query embedding:      "What are Apple's liquidity risks?"
                      ← points toward QUESTION space in embedding model

HyDE embedding:       "Apple's liquidity risks stem from commercial paper
                       program and $95B term debt..."
                      ← points toward ANSWER/DOCUMENT space

SEC filing text:      "Apple's total term debt obligations... commercial
                       paper program... credit facilities..."
                      ← also in ANSWER/DOCUMENT space

→ HyDE embedding closer to SEC text than query embedding
→ ~10-15% improvement in recall for complex financial questions
```

| Pros | Cons |
|------|------|
| Highest retrieval accuracy | 3-4 extra LLM calls → 2-4s additional latency |
| HyDE bridges question-answer embedding gap | Higher cost (Claude calls for multi-query + HyDE + reranking) |
| Catches documents missed by single-query retrieval | Overkill for simple factual queries |

**Alternatives:**
- **Cohere Rerank API** — specialized reranking model, no LLM needed, faster
- **BGE Reranker (local)** — open-source cross-encoder, zero cost but needs GPU
- **ColBERT** — token-level interaction model, theoretically better than cross-encoder

---

### 5.4 CorrectiveRetriever (CRAG) — Level 4

**What it does:** After initial retrieval, grades each document for relevance. If too many are irrelevant, transforms the query and re-retrieves. Up to 2 correction attempts.

**Detailed workflow with real thresholds:**

```
Query: "Apple's debt maturity schedule and refinancing risks"
ticker: "AAPL"

STEP 1: Initial retrieval (IntermediateRetriever)
  Retrieved 5 chunks with vector scores:
    Chunk 1: score=0.65  "Apple's long-term debt of $95.3B..."
    Chunk 2: score=0.63  "Competition in the smartphone market..."
    Chunk 3: score=0.61  "Apple Watch and wearables revenue..."
    Chunk 4: score=0.59  "Supply chain concentration in Asia..."
    Chunk 5: score=0.57  "Apple term debt maturing within 12 months..."

  top_score = 0.65, avg_score = 0.61

STEP 2: Hybrid confidence check
  Is top_score (0.65) >= 0.72?  NO
  → Must proceed to LLM grading (can't skip)

STEP 3: Grade via Claude Haiku (batch of 5 in one call)
  Cost: ~$0.000025 per call (5 docs × $0.00000025/token × ~100 tokens each)

  Haiku receives:
  "Query: 'Apple debt maturity schedule and refinancing risks'
   Grade each document as RELEVANT, PARTIAL, or IRRELEVANT:
   [1] 'Apple's long-term debt of $95.3B includes notes payable...'
   [2] 'Competition in the smartphone market intensified...'
   [3] 'Apple Watch and wearables revenue declined 3%...'
   [4] 'Supply chain concentration in Asia creates risk...'
   [5] 'Apple term debt maturing within 12 months is $13B...'"

  Grades:
    Chunk 1: RELEVANT   (directly addresses debt)        score=1.0
    Chunk 2: IRRELEVANT (about competition, not debt)     score=0.0
    Chunk 3: IRRELEVANT (about products, not debt)        score=0.0
    Chunk 4: PARTIAL    (supply chain risk, tangential)   score=0.5
    Chunk 5: RELEVANT   (directly addresses maturity)     score=1.0

  usable_ratio = (2 relevant + 1×0.5 partial) / 5 = 0.50

STEP 4: Is 0.50 >= threshold (0.40)?  YES → ACCEPT
  (threshold is 0.40, not 0.50 as often documented)

  Return CorrectiveResult:
    nodes: [Chunk1, Chunk2, Chunk3, Chunk4, Chunk5]
    grades: [RELEVANT, IRRELEVANT, IRRELEVANT, PARTIAL, RELEVANT]
    initial_relevant_ratio: 0.50
    final_relevant_ratio: 0.50
    corrections_applied: []
    num_correction_attempts: 0

  Context string built:
    "[Retrieval confidence: 50%]
     [Corrections applied: 0]

     [RELEVANT]
     Apple's long-term debt of $95.3B includes notes payable...
     Apple term debt maturing within 12 months is $13B...

     [PARTIAL]
     Supply chain concentration in Asia creates operational risk...

     [IRRELEVANT]
     Competition in the smartphone market intensified...
     Apple Watch and wearables revenue declined 3%..."
```

**Example where corrections ARE triggered:**
```
Query: "Apple's Options Chain implied volatility analysis"
  (no options data in SEC filings — wrong retrieval domain)

STEP 1: Initial retrieval
  5 chunks about Apple's business, all about products/strategy
  top_score=0.55, avg=0.48 → NOT high confidence

STEP 3: Grading
  All 5 chunks graded IRRELEVANT (options IV not in SEC filings)
  usable_ratio = 0 / 5 = 0.0 → below threshold 0.40

STEP 4: Correction attempt 1
  QueryTransformer.transform():
    Strategy: rule_based
    - Detect "options" → add "derivatives", "put/call", "volatility"
    - Detect "implied volatility" → add "IV", "options premium"

    Transformed: "Apple AAPL options chain implied volatility IV derivatives put call"

  Re-retrieve:
    Still gets 5 chunks about business/products (options IV not in SEC docs)
    Regraded: all IRRELEVANT
    new_ratio = 0.0 → best_ratio unchanged

STEP 5: Correction attempt 2
  Strategy: llm_transform
    Claude rewrites: "Apple options market data volatility surface"

  Re-retrieve: Still no options data in SEC filings
  → best_ratio stays 0.0

Result:
  CorrectiveResult:
    nodes: original 5 chunks (best available)
    corrections_applied: ["rule_based: Apple AAPL options...", "llm_transform: Apple options..."]
    initial_relevant_ratio: 0.0
    final_relevant_ratio: 0.0
    num_correction_attempts: 2

  Context string:
    "[Retrieval confidence: 0%]
     [Corrections applied: 2]

     [IRRELEVANT]
     Apple designs, manufactures, and markets smartphones...
     ..."

  Agent (OptionsAnalysisAgent) sees confidence=0% and falls back entirely
  to tool calls (get_options_data → Tradier API) for live options data.
  The agent's output confidence will be lowered (0.4-0.5 instead of 0.8+).
```

| Pros | Cons |
|------|------|
| Self-correcting: catches retrieval failures | Adds 500ms-3s latency for grading |
| Explicit quality signals for agents | Claude Haiku cost per analysis (~$0.001) |
| Reduces agent hallucination (agents know when context is weak) | Doesn't help when domain mismatch is fundamental (options in SEC filings) |
| Hybrid fast-path skips grading for high-confidence results | — |

**Alternatives:**
- **Reranking-only** — skip grading, just rerank by score (AdvancedRetriever approach)
- **Threshold filtering** — drop chunks below score X (simpler but less intelligent)
- **Query decomposition first** — break complex queries before retrieval
- **Self-RAG** (original paper) — interleave generation and retrieval token-by-token (more complex)

---

## 6. Agent System: Architecture & Interaction

### 6.1 Agent Catalog

| Agent | Weight | SEC Sections Used | Tools Permitted | Primary Analysis Focus |
|-------|--------|------------------|-----------------|----------------------|
| financial_analyst | **20%** | balance_sheet, income_statement, cash_flow | calculate_financial_ratio, compare_companies, get_stock_price | Revenue, margins, debt, FCF, ROE |
| technical_analyst | **15%** | — | get_technical_indicators, get_stock_price | RSI, MACD, Bollinger Bands, SMA/EMA |
| news_sentiment | **12%** | risk_factors, analyst_commentary | get_social_sentiment | Recent news tone, headline analysis |
| analyst_ratings | **10%** | analyst_commentary | get_analyst_ratings, get_stock_price | Buy/sell ratings, price targets, consensus |
| risk_assessment | **10%** | risk_factors, contingencies | compare_companies, calculate_financial_ratio | Risk disclosures, mitigation plans |
| competitive_analysis | **10%** | business_description, market_overview | compare_companies | Market share, competitive moat |
| insider_activity | **8%** | insider_transactions | get_insider_trades, get_stock_price | Form 4 trades, 10b5-1 plans |
| earnings_analysis | **7%** | income_statement, earnings_guidance | calculate_financial_ratio, get_stock_price | EPS beat/miss history, guidance |
| options_analysis | **5%** | — | get_options_data, get_stock_price | IV skew, put/call ratio, open interest |
| social_sentiment | **3%** | — | get_social_sentiment | Reddit WallStreetBets, StockTwits buzz |

**Weights sum to 100%.** Financial fundamentals (20% + 7%) are most important; social sentiment (3%) is the least.

### 6.2 BaseAgent Tool-Use Loop

The real implementation from `base_agent.py`:

```python
def analyze(self, ticker: str, rag_context: str, tracer: TracingManager) -> AgentOutput:
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    # Step 1: Build initial messages
    user_message = self._build_user_message(ticker, rag_context)
    messages = [{"role": "user", "content": user_message}]

    # Step 2: Filter tools to only this agent's permitted tools
    tools = [t for t in FINANCIAL_TOOLS if t["name"] in self.TOOLS]
    # e.g., financial_analyst only gets [calculate_financial_ratio, compare_companies, get_stock_price]
    # not get_options_data or get_insider_trades

    # Step 3: Tool-use loop
    for iteration in range(self.MAX_TOOL_ITERATIONS):  # max 10
        response = self._call_with_retry(client, {
            "model": settings.claude_model,
            "max_tokens": 2048,
            "temperature": self.TEMPERATURE,  # 0.2 for consistency
            "system": self.SYSTEM_PROMPT,
            "messages": messages,
            "tools": tools,
        })

        # Log to Langfuse (LLM call span)
        tracer.log_llm_call(agent_span, messages, response, model, usage)
        # Log to Galileo (model call)
        log_llm_call(model=model, prompt=..., response=..., tokens=...)

        if response.stop_reason == "tool_use":
            # Extract all tool_use blocks (Claude may request multiple tools at once)
            tool_blocks = [b for b in response.content if b.type == "tool_use"]
            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in tool_blocks:
                try:
                    result = execute_tool(block.name, block.input)
                except Exception as e:
                    result = {"error": str(e)}  # Graceful failure

                tracer.log_tool_call(span, block.name, block.input, result)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result, default=str)
                })

            messages.append({"role": "user", "content": tool_results})
            # Loop continues with tool results in context

        else:
            # stop_reason == "end_turn" — Claude has finished
            text = next((b.text for b in response.content if hasattr(b, "text")), "")
            output = self._try_parse_or_retry(client, settings, messages, tools, text, ticker)

            # Galileo validation
            galileo_result = log_agent_output(
                agent_name=self.AGENT_NAME,
                ticker=ticker,
                score=output.score,
                summary=output.summary,
                rag_context=rag_context,
            )
            if not galileo_result.get("is_grounded", True):
                logger.warning(f"Possible hallucination (score: {galileo_result['score']:.0%})")
            if galileo_result.get("pii_detected"):
                logger.warning(f"PII detected: {galileo_result['pii_detected']}")

            return output

    # Max iterations reached → force JSON output without tools
    return self._force_final_output(client, settings, messages, ticker)
```

**JSON parsing with fallback:**
```python
def _parse_output(self, text: str, ticker: str) -> AgentOutput:
    json_str = text.strip()

    # Handle code fences: ```json {...} ``` or ``` {...} ```
    if "```json" in json_str:
        json_str = json_str.split("```json")[1].split("```")[0]
    elif "```" in json_str:
        json_str = json_str.split("```")[1].split("```")[0]

    # Handle prose wrapping: "Based on my analysis: {...}"
    if not json_str.startswith("{"):
        start = json_str.find("{")
        end = json_str.rfind("}")
        if start != -1 and end > start:
            json_str = json_str[start:end + 1]

    data = json.loads(json_str)
    data["agent_name"] = self.AGENT_NAME   # Force correct values
    data["ticker"] = ticker
    return AgentOutput(**data)             # Pydantic validates all fields
```

**Retry on rate limit (exponential backoff):**
```python
def _call_with_retry(self, client, kwargs):
    for attempt in range(MAX_RETRIES + 1):  # 0, 1, 2, 3
        try:
            return client.messages.create(**kwargs)
        except anthropic.RateLimitError:
            if attempt < MAX_RETRIES:
                delay = 30 * (2 ** attempt)  # 30s, 60s, 120s
                time.sleep(delay)
            else:
                raise
```

### 6.3 Agent-to-Agent Communication via LangGraph State

Agents do **not** call each other directly. They communicate through shared LangGraph state. This is intentional — it prevents cascading failures and allows true parallelism.

```
State before agents start:
{
  "ticker": "AAPL",
  "rag_context": {
    "financial_analyst": "[Retrieval confidence: 76%]...",
    "technical_analyst": "[Retrieval confidence: 91%]...",
    ...
  },
  "agent_outputs": [],   ← starts empty
  ...
}

After financial_analyst completes (t=12s):
{
  "agent_outputs": [
    AgentOutput(agent_name="financial_analyst", score=0.72, ...)
  ]
}
                    ← operator.add reducer appended it

After technical_analyst completes (t=17s):
{
  "agent_outputs": [
    AgentOutput(agent_name="financial_analyst", score=0.72, ...),
    AgentOutput(agent_name="technical_analyst", score=0.45, ...)
  ]
}
                    ← operator.add appended again, no lock needed

After all 10 complete:
{
  "agent_outputs": [10 AgentOutput objects]  ← synthesis reads ALL of these
}
```

**Why operator.add reducer?**
Without it, parallel agents writing to the same list key would overwrite each other. The `Annotated[list, operator.add]` tells LangGraph: "when multiple parallel nodes write to this key, concatenate the lists instead of replacing."

```python
# operator.add behavior:
# Node A writes: [AgentOutput_A]
# Node B writes: [AgentOutput_B]
# State result:  [AgentOutput_A, AgentOutput_B]  ← both preserved
```

**What synthesis receives:**
```python
def synthesis_node(state: AnalysisState):
    outputs = state["agent_outputs"]  # All 10 AgentOutputs

    # No agent knows what other agents said — synthesis is the first place
    # where cross-agent information is combined
    recommendation = synthesize(outputs, state["ticker"], state["mode"])
    return {"recommendation": recommendation}
```

---

## 7. Tool Calling / Function Calling

### 7.1 Tool Schema (Claude format)

Each tool is defined with a JSON schema that Claude uses to decide what parameters to send:

```python
{
    "name": "calculate_financial_ratio",
    "description": "Calculate financial ratios like P/E, debt-to-equity, ROE, current ratio",
    "input_schema": {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "Stock ticker symbol (e.g., AAPL, MSFT)"
            },
            "ratio": {
                "type": "string",
                "enum": ["pe_ratio", "debt_to_equity", "roe", "current_ratio",
                         "price_to_book", "free_cash_flow_yield"],
                "description": "Which financial ratio to calculate"
            }
        },
        "required": ["ticker", "ratio"]
    }
}
```

**Example of Claude requesting tools:**
```json
{
  "type": "tool_use",
  "id": "toolu_01XWnm...",
  "name": "calculate_financial_ratio",
  "input": {
    "ticker": "AAPL",
    "ratio": "pe_ratio"
  }
}
```

**Example tool result injected back:**
```json
{
  "type": "tool_result",
  "tool_use_id": "toolu_01XWnm...",
  "content": "{\"pe_ratio\": 29.4, \"forward_pe\": 27.1, \"sector_avg_pe\": 24.3}"
}
```

### 7.2 Multi-Provider Fallback with Real Examples

```python
def get_stock_price(ticker: str, period: str = "1d") -> dict:
    # Try Finnhub (primary: 60 calls/min)
    if finnhub_client and finnhub_client.is_configured():
        try:
            data = finnhub_client.get_quote(ticker)
            # {"c": 189.30, "h": 191.20, "l": 187.50, "o": 188.00, "pc": 187.80}
            return {"price": data["c"], "change_pct": ..., "source": "finnhub"}
        except FinnhubRateLimitError:
            logger.warning("Finnhub rate limited, trying FMP")

    # Try FMP (secondary: 5 calls/min)
    if fmp_client and fmp_client.is_configured():
        try:
            data = fmp_client.get_quote(ticker)
            return {"price": data["price"], "source": "fmp"}
        except FMPRateLimitError:
            logger.warning("FMP rate limited, trying yfinance")

    # Try yfinance (fallback: unlimited, unofficial)
    try:
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.fast_info  # cached 30 min
        return {"price": info.last_price, "source": "yfinance"}
    except Exception as e:
        return {"error": f"All providers failed: {e}"}
```

**Caching strategy:**
```python
# yfinance: cache ticker object 30 minutes
_yfinance_cache = {}
_yfinance_cache_time = {}

def get_yf_ticker(ticker):
    if ticker in _yfinance_cache:
        age = time.time() - _yfinance_cache_time[ticker]
        if age < 1800:  # 30 minutes
            return _yfinance_cache[ticker]
    obj = yf.Ticker(ticker)
    _yfinance_cache[ticker] = obj
    _yfinance_cache_time[ticker] = time.time()
    return obj

# FMP client: 5-minute response cache
# Prevents identical calls within same analysis from counting against rate limit
```

---

## 8. LangGraph Orchestration

### 8.1 Graph Definition with Comments

```python
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

builder = StateGraph(AnalysisState)

# Add nodes
builder.add_node("route_query", route_query_node)
builder.add_node("gather_context", gather_context_node)
builder.add_node("fan_out", fan_out_node)
# Add one node per agent (they run in parallel via Send, not as separate nodes in the graph)
for name in AGENT_NAMES:
    builder.add_node(name, make_agent_node(name))  # Closure over agent instance
builder.add_node("synthesis", synthesis_node)

# Linear edges for sequential phases
builder.add_edge(START, "route_query")
builder.add_edge("route_query", "gather_context")
builder.add_edge("gather_context", "fan_out")

# Fan-out: conditional edges that create parallel sends
builder.add_conditional_edges(
    "fan_out",
    lambda state: [
        Send(agent_name, state)          # Each Send creates a parallel execution
        for agent_name in state["active_agents"]
    ]
)

# All parallel agents converge at synthesis
for name in AGENT_NAMES:
    builder.add_edge(name, "synthesis")   # Each agent → synthesis when done

builder.add_edge("synthesis", END)

graph = builder.compile()
```

**Why `Send()` instead of normal edges?**

Normal edge: `add_edge("fan_out", "agent")` → all agents run sequentially
`Send("agent", state)`: creates parallel async execution, all agents get their own copy of state, results merged via reducers.

### 8.2 State Reducer Deep Dive

```python
class AnalysisState(TypedDict):
    ticker: str
    query: Optional[str]
    rag_level: str
    active_agents: list[str]
    mode: str
    rag_context: dict[str, str]

    # Reducer fields: safe for parallel writes
    agent_outputs: Annotated[list[AgentOutput], operator.add]
    errors: Annotated[list[str], operator.add]

    # Normal fields: last write wins (only written by one node at a time)
    recommendation: Optional[InvestmentRecommendation]
    trace_id: Optional[str]
```

**What happens without a reducer:**
```
# Without Annotated[list, operator.add]:
# Agent A writes:  state["agent_outputs"] = [output_A]
# Agent B writes:  state["agent_outputs"] = [output_B]
# Final state:     [output_B]  ← output_A was overwritten!

# With reducer:
# Agent A writes:  +[output_A]
# Agent B writes:  +[output_B]
# Final state:     [output_A, output_B]  ← both preserved
```

---

## 9. Synthesis: Score Aggregation

### Real implementation from synthesis.py:

```python
AGENT_WEIGHTS = {
    "financial_analyst": 0.20,
    "news_sentiment":    0.12,
    "technical_analyst": 0.15,
    "risk_assessment":   0.10,
    "competitive_analysis": 0.10,
    "insider_activity":  0.08,
    "options_analysis":  0.05,
    "social_sentiment":  0.03,
    "earnings_analysis": 0.07,
    "analyst_ratings":   0.10,
}

# Overall score: confidence-adjusted weighted average
total_weighted = 0.0
total_weight = 0.0
for o in outputs:
    w = AGENT_WEIGHTS.get(o.agent_name, 0.05)
    total_weighted += o.score * o.confidence * w
    total_weight   += o.confidence * w
overall_score = total_weighted / total_weight  # Normalize by actual confidence

# Category scores: confidence-weighted within category
CATEGORIES = {
    "financial_score": ["financial_analyst", "earnings_analysis"],
    "technical_score": ["technical_analyst", "options_analysis"],
    "sentiment_score": ["news_sentiment", "social_sentiment", "analyst_ratings"],
    "risk_score":      ["risk_assessment", "competitive_analysis", "insider_activity"],
}

for category, agent_names in CATEGORIES.items():
    relevant = [o for o in outputs if o.agent_name in agent_names]
    weighted_sum = sum(o.score * o.confidence for o in relevant)
    confidence_sum = sum(o.confidence for o in relevant)
    category_scores[category] = weighted_sum / confidence_sum
```

**Example calculation:**
```
Scenario: All agents bullish but risk agent is very bearish

financial_analyst:   score=0.72, confidence=0.85, weight=0.20
                     contribution: 0.72 × 0.85 × 0.20 = 0.1224

risk_assessment:     score=-0.80, confidence=0.90, weight=0.10
                     contribution: -0.80 × 0.90 × 0.10 = -0.0720

Without confidence weighting:
  overall_score = Σ(score × weight) = ... → risk gets 10% weight regardless

With confidence weighting (actual implementation):
  High confidence risk warning (-0.80 × 0.90) weighs MORE than low confidence bullish
  signal (0.80 × 0.30) — correct behavior: certain bad news > uncertain good news
```

**Thesis generation prompt:**
```python
summaries = "\n".join(
    f"- {o.agent_name} (score={o.score}, confidence={o.confidence}): {o.summary}"
    for o in outputs
)

client.messages.create(
    model=settings.claude_model,
    messages=[{
        "role": "user",
        "content": (
            f"Based on these agent analyses for {ticker} (recommendation: {rec}):\n\n"
            f"{summaries}\n\n"
            "Generate a JSON with:\n"
            '- "thesis": 2-3 sentence investment thesis\n'
            '- "bullish_factors": 3 key bullish factors\n'
            '- "bearish_factors": 3 key bearish factors\n'
            '- "risks": 3 key risks\n\n'
            "Respond with ONLY valid JSON."
        )
    }]
)
```

---

## 10. Monitoring & Observability (Langfuse + Galileo)

### 10.1 Langfuse Trace Hierarchy

```
Langfuse UI shows this tree for every analysis:

Trace: AAPL | 2026-03-03 | 47.3s | $0.84 | 42,150 tokens
│
├── [Span] route_query                     12ms
│     Input:  { query: null, ticker: "AAPL" }
│     Output: { active_agents: [10 agents], mode: "full" }
│
├── [Span] gather_context                  3,240ms
│   ├── [Span] rag_financial_analyst_0     620ms
│   │     Input:  { query: "Revenue growth trends for AAPL", rag_level: "corrective" }
│   │     Output: { chunks: 5, confidence: 82%, corrections: 0 }
│   ├── [Span] rag_financial_analyst_1     580ms
│   ├── [Span] rag_technical_analyst_0    180ms
│   └── ... (2-4 spans per agent)
│
├── [Span] financial_analyst               8,420ms | $0.19
│   ├── [Span] llm_call_0                 2,100ms
│   │     Model: claude-opus-4-5-20251101
│   │     Input tokens: 3,420  Output tokens: 156
│   ├── [Span] tool: get_stock_price      340ms
│   │     Provider: finnhub
│   │     Input:  { ticker: "AAPL" }
│   │     Output: { price: 189.30, ... }
│   ├── [Span] tool: calculate_financial_ratio  280ms
│   ├── [Span] llm_call_1                3,800ms
│   │     Input tokens: 4,100  Output tokens: 312
│   └── [Score] agent_score
│         Value: 0.72 | Comment: "Strong FCF growth, improving margins"
│
├── [Span] technical_analyst              5,100ms | $0.11
├── [Span] news_sentiment                 4,200ms | $0.09
│   └── ... (other agents in parallel)
│
└── [Span] synthesis                      4,200ms | $0.09
      Input:  { num_agents: 10, agent_scores: {...} }
      Output: { recommendation: "BUY", overall_score: 0.565 }
      └── [Score] recommendation_score
            Value: 0.565 | Comment: "BUY based on weighted multi-agent analysis"
```

### 10.2 Galileo Evaluation Dashboard

```
Galileo shows per-output metrics:

Run: AAPL_analysis_2026-03-03

Agent Outputs:
  financial_analyst:
    Groundedness: 0.92 ✓     (summary matches SEC context)
    PII Detected: None ✓
    Context Relevance: 0.81

  risk_assessment:
    Groundedness: 0.71 ✓     (borderline — watch this agent)
    PII Detected: None ✓
    Flagged claims: ["China revenues represent 19% of total"]
    → Actual SEC text: "Greater China segment: $72.6B of $383.3B = 18.9%"
    → 19% ≈ 18.9% → acceptable rounding

  social_sentiment:
    Groundedness: 0.45 ✗     (LOW — social agent uses live Reddit data not in RAG)
    PII Detected: None ✓
    → Expected: social agent has no RAG context, lower groundedness is normal

Synthesis Thesis:
    Groundedness: 0.88 ✓
    PII Detected: None ✓
```

---

## 11. Guardrails

### Three active guardrails:

**1. Hallucination Detection (most important):**
```python
# Called after each agent produces its summary
result = check_hallucination(
    response=agent.summary,
    context=[rag_context],
    threshold=0.7
)

# Example where it triggers:
# Agent summary: "Apple's revenue grew 15% in FY2023"
# RAG context:   "Apple net sales of $383,285M, a 5.5% increase"
# result = { is_grounded: False, score: 0.31, flagged_claims: ["15% growth"] }
# → logger.warning(f"Possible hallucination (score: 31%)")
```

**2. PII Detection + Redaction:**
```python
result = check_pii(text)
# Example:
# Input:  "CEO Tim Cook's email tim@apple.com was mentioned in the filing"
# Output: { has_pii: True, pii_types: ["email", "name"],
#           redacted_text: "CEO [PERSON]'s email [EMAIL] was mentioned..." }
# → thesis_data["thesis"] = pii_result["redacted_text"]
```

**3. Context Relevance (RAG quality check):**
```python
result = check_context_relevance(query, chunks)
# Returns per-chunk relevance scores
# Used in evaluation runs to measure retrieval quality
```

---

## 12. Evaluations & Metrics

### 12.1 Eval Architecture Overview

The eval system is **pytest-based** with four evaluation dimensions, golden datasets, and persistent result storage. It benchmarks both the RAG retrieval pipeline and the LLM generation quality.

```
backend/evals/
├── conftest.py                        # Session fixtures: rag_index, golden_set, evaluators
├── test_baseline.py                   # IntermediateRetriever metrics (2 classes, 4 tests)
├── test_crag.py                       # CorrectiveRetriever + CRAG-specific metrics (2 classes, 4 tests)
├── test_galileo.py                    # Galileo guardrails: groundedness, PII, hallucination (4 classes, 10+ tests)
├── run_direct_comparison.py           # Standalone: side-by-side Baseline vs CRAG comparison
├── metrics/
│   ├── retrieval_metrics.py           # RetrievalEvaluator: P@K, R@K, F1, MRR, NDCG, MAP, Hit Rate
│   ├── ragas_metrics.py               # RAGASEvaluator: Faithfulness, Relevancy, Context P/R, Correctness
│   ├── cost_latency.py                # CostLatencyTracker: per-phase token/cost/latency tracking
│   └── galileo_metrics.py             # GalileoEvaluator: groundedness, PII, hallucination detection
├── datasets/
│   ├── golden_set.json                # 10 Apple queries with ground truth answers (chunk IDs NOT populated)
│   ├── galileo_eval_set.json          # 10 queries + 3 hallucination tests + 3 PII tests
│   └── generate_golden_set.py         # CLI utilities to generate/validate golden sets
└── results/
    └── comparison_20260219_*.json     # 7 baseline vs CRAG comparison runs from tuning sessions
```

### 12.2 Evaluation Dimensions

**Dimension 1: Retrieval Quality** — Custom metrics computed per-query and aggregated:

| Metric | Formula | Example |
|--------|---------|---------|
| **Precision@5** | relevant_in_top5 / 5 | Retrieved [doc1✓, doc2✗, doc3✓, doc4✗, doc5✓] → 3/5 = **0.60** |
| **Recall@5** | relevant_in_top5 / total_relevant | If total_relevant=4, found 3 → 3/4 = **0.75** |
| **F1@5** | 2×P×R / (P+R) | 2×0.60×0.75 / (0.60+0.75) = **0.667** |
| **MRR** | 1 / rank_first_relevant | First relevant at position 1 → **1.0**; at position 3 → **0.33** |
| **NDCG@5** | DCG / ideal_DCG | Measures ranking quality, penalizes relevant docs at lower positions |
| **MAP@5** | avg precision at each relevant rank | Average precision considering all relevant positions |
| **Hit Rate** | queries_with_≥1_hit / total | 18 of 20 queries found ≥1 relevant doc → **0.90** |
| **Avg Relevance Score** | mean(grader scores) | Optional: from CRAG relevance grader output |

**Dimension 2: Answer Quality (RAGAS)** — LLM-as-judge metrics via the RAGAS framework:

| Metric | What it checks | Example |
|--------|----------------|---------|
| **Faithfulness** | Is every claim in the answer supported by retrieved context? | Answer: "Revenue grew 15%" but context says "5.5%" → score=0.20 |
| **Answer Relevancy** | Does the answer actually address the question? | Q: "What are risk factors?" A: "Revenue was $383B" → low relevancy |
| **Context Precision** | Are the retrieved chunks actually useful for answering? | Retrieved chunk about Apple products but question is about risk → low precision |
| **Context Recall** | Did retrieved chunks contain all info needed for a complete answer? | Missing the most relevant risk chunk → low recall |
| **Answer Correctness** | Compared to ground truth, is the answer correct? | Ground truth: "$383.3B", answer: "$383.3 billion" → high score |

**Dimension 3: Hallucination & Safety (Galileo)** — Runtime guardrail quality:

| Test Class | Tests | What it validates |
|------------|-------|-------------------|
| TestGroundedness | 3 tests | Grounded answers score ≥0.5, hallucinated answers score <0.7, batch ≥50% pass |
| TestHallucinationDetection | 3 tests | Fabricated numbers, invented products, fabricated quotes detected |
| TestPIIDetection | 4 tests | Email, phone detected; clean financial text has no false positives |
| TestFullEvaluation | 2 tests | Full pipeline batch eval + file-based eval runner |

**Dimension 4: CRAG-Specific Metrics** — Self-healing RAG evaluation:

```python
# test_crag.py collects these per query:
{
    "initial_relevant_ratio": 0.40,   # Before any corrections
    "final_relevant_ratio": 0.80,     # After corrections
    "corrections_applied": ["rule_based: AAPL Apple revenue growth net sales..."],
    "num_corrections": 1,
    "improved": True,                 # final > initial
    "latency_overhead_ms": 820        # Extra time vs baseline
}

# Aggregate CRAG metrics across all queries:
# - Correction Rate: % of queries needing corrections
# - Improvement Rate: % of corrected queries that actually improved
# - Hybrid Skip Rate: % of queries where vector scores were high enough to skip LLM grading
# - Avg Corrections/Query
```

### 12.3 Cost & Latency Tracking

```python
# Pricing per token (from cost_latency.py):
PRICING = {
    "claude-opus-4-5-20251101": {
        "input":  0.000015,   # $15 per million input tokens
        "output": 0.000075,   # $75 per million output tokens
    },
    "claude-haiku-*": {
        "input":  0.00000025, # $0.25 per million (CRAG grading)
        "output": 0.00000125,
    },
    "text-embedding-3-small": {
        "per_1k_tokens": 0.00002,  # $0.02 per million
    }
}

# CostLatencyTracker breaks down per phase:
# - Embedding tokens + costs
# - Grading (Claude Haiku) tokens + costs
# - Correction (query transformation) tokens + costs
# - Generation (agent answer) tokens + costs
# - Web search count + costs (Tavily)
# - Latency: retrieval_ms, grading_ms, correction_ms, generation_ms
# - Percentiles: p50, p95, p99

# Typical cost breakdown per full AAPL analysis:
# 10 agents × ~6000 tokens avg = 60,000 agent tokens
# claude-opus-4-5: (50k input + 10k output) = $0.75 + $0.075 = $0.825
# synthesis: ~5k input + 0.5k output = $0.075 + $0.037 = $0.11
# CRAG grading: ~500 Haiku tokens × 10 agents = $0.001
# Total: ~$0.90 per full analysis
```

### 12.4 Running Evals

```bash
# Fast tests (retrieval metrics only, no LLM calls)
pytest backend/evals/ -v -m "not slow"

# Full suite including RAGAS (makes LLM calls, ~$0.50 per run)
pytest backend/evals/ -v -m slow

# Baseline vs CRAG comparison (standalone script)
python backend/evals/run_direct_comparison.py

# Galileo guardrails (skipped if GALILEO_API_KEY not set)
pytest backend/evals/test_galileo.py -v

# Generate/validate golden set
python backend/evals/datasets/generate_golden_set.py --validate
python backend/evals/datasets/generate_golden_set.py --ticker AAPL --find-chunks "query"
```

### 12.5 Current Eval Status & Gaps

| Component | Status | Notes |
|-----------|--------|-------|
| Retrieval Metrics (7 metrics) | ✅ Working | Production-ready, tested, aggregated |
| RAGAS Integration (5 metrics) | ⚠️ Partial | Works but marked `@pytest.mark.slow`, limited to 5 queries |
| Galileo Guardrails Evals | ⚠️ Partial | Skipped if no API key, can't run in CI |
| Cost/Latency Tracking | ✅ Working | Realistic pricing, all phases tracked |
| CRAG vs Baseline Comparison | ✅ Working | 7 comparison runs saved with timestamps |
| Golden Dataset | ❌ Broken | `ground_truth_chunks` empty for all 10 queries — tests skip them |
| Agent Scoring Evals | ❌ Missing | No per-agent accuracy, consistency, or calibration tests |
| End-to-End Pipeline Evals | ❌ Missing | No full analysis → recommendation quality testing |
| CI/CD Integration | ❌ Missing | No GitHub Actions workflow for automated eval runs |
| Eval Dashboard | ❌ Missing | Results stored as JSON files, no visualization |

**Critical issue:** The golden dataset (`golden_set.json`) has empty `ground_truth_chunks` arrays for all 10 queries. Since tests skip queries without ground truth (`if not ground_truth_ids: continue`), **0% of retrieval metrics actually compute against ground truth**. This must be fixed before retrieval evals are meaningful.

---

## 13. API Endpoints

```
Health
  GET  /health
       Response: { "status": "ok", "timestamp": "2026-03-03T12:00:00Z" }

Analysis
  POST /api/analysis/analyze
       Body:     { ticker, query?, rag_level?, mode? }
       Response: InvestmentRecommendation (full)
       Note:     Runs all 10 agents (~45-60 seconds)

  POST /api/analysis/quick
       Body:     { ticker }
       Response: AgentOutput (single agent, no RAG)
       Note:     ~5-10 seconds, less accurate

RAG
  POST /api/query
       Body:     { query, ticker? }
       Response: { "result": "retrieved context..." }

  POST /api/qa
       Body:     { question, ticker? }
       Response: { "answer": "...", "sources": ["SEC 10-K 2023", ...] }

SEC Filings
  GET  /api/sec/filings/{ticker}
       Response: [{ date, type, path }, ...]

  POST /api/sec/download
       Body:     { ticker, force: bool }
       Response: { "status": "ok", "num_files": 8 }

  POST /api/sec/index
       Body:     { force: bool }
       Response: { "status": "ok", "num_documents": 3420 }
```

---

## 14. Configuration Reference

```bash
# Required
OPENAI_API_KEY=sk-...           # Embeddings + GPT-4 fallback
ANTHROPIC_API_KEY=sk-ant-...    # Claude (all 10 agents + synthesis)
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_KEY=eyJ...
POSTGRES_CONNECTION_STRING=postgresql://user:pass@host:5432/db

# Financial Data (all optional, graceful degradation)
FINNHUB_API_KEY=xxx             # Primary: 60 calls/min (recommended)
FMP_API_KEY=xxx                 # Secondary: 5 calls/min
TRADIER_API_TOKEN=xxx           # Options: sandbox free

# Social Sentiment (optional)
REDDIT_CLIENT_ID=xxx
REDDIT_CLIENT_SECRET=xxx

# Observability (optional but highly recommended)
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
GALILEO_API_KEY=xxx

# RAG Configuration (optional, has defaults)
CHUNK_SIZE=512                  # Tokens per chunk
CHUNK_OVERLAP=50                # Token overlap between chunks
TOP_K=5                         # Documents retrieved per query
RAG_LEVEL=intermediate          # Default: intermediate

# Model Configuration (optional, has defaults)
CLAUDE_MODEL=claude-opus-4-5-20251101
OPENAI_MODEL=gpt-4-turbo-preview
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Server (optional)
API_PORT=8001                   # Overridden by Railway $PORT
SEC_USER_EMAIL=your@email.com   # Required for EDGAR download
LOG_LEVEL=INFO
```

---

## 15. How to Measure & Monitor: Metrics Guide

### 15.1 Retrieval Quality Targets

| Metric | Good | Acceptable | Action if below |
|--------|------|-----------|-----------------|
| Precision@5 (intermediate) | >0.70 | 0.50-0.70 | Add more SEC filings, improve metadata |
| Recall@5 (intermediate) | >0.60 | 0.45-0.60 | Increase top_k or improve synonyms |
| CRAG usable_ratio (final) | >0.65 | 0.50-0.65 | Review query transformer, add SEC sections |
| CRAG correction rate | <30% | 30-50% | Improve initial retrieval, tune threshold |
| Retrieval latency p95 | <500ms | 500ms-1s | Check HNSW index health, pgvector tuning |
| RAGAS Faithfulness | >0.85 | 0.70-0.85 | Investigate hallucinating agents |
| RAGAS Context Precision | >0.75 | 0.60-0.75 | Improve metadata filtering |

**Running retrieval evals:**
```bash
# Baseline
pytest backend/evals/test_baseline.py -v
# Output: precision=0.73, recall=0.65, mrr=0.84 → stored in evals/results/

# CRAG
pytest backend/evals/test_crag.py -v
# Output: comparison table showing CRAG vs baseline improvement

# RAGAS (slow, costs API $)
pytest backend/evals/ -v -m slow
```

### 15.2 Agent Quality Targets

| Metric | How to measure | Good | Action if bad |
|--------|----------------|------|---------------|
| Galileo groundedness | Langfuse + Galileo dashboard | >0.80 avg | Check agent system prompt, improve RAG quality |
| Score variance (same ticker, 2 runs) | Run analysis twice, compare scores | Δ < 0.15 | Lower temperature, improve few-shot examples |
| Tool success rate | Langfuse: count tool errors / total | >95% | Check API keys, rate limit handling |
| Max iterations hit | Langfuse: filter for forced_final_output spans | <5% | Increase TOP_K context, improve prompts |
| PII incidents | Galileo: has_pii=True count | 0 | Review agent prompts, add explicit PII avoidance |

### 15.3 End-to-End Performance

| Metric | Target | How to measure |
|--------|--------|----------------|
| Full analysis time | <60s | Langfuse trace duration, Prometheus histogram |
| Focused analysis time | <20s | Same |
| Cost per full analysis | <$1.00 | Langfuse token costs |
| API error rate | <1% | FastAPI middleware logs |
| Recommendation consistency | Δ < 0.15 | Run A/B test: same ticker, 2x analysis |

### 15.4 Prometheus Metrics to Scrape

```
# Expose at /metrics endpoint

airas_analysis_duration_seconds{rag_level, mode}    # Histogram
airas_agent_score{agent_name, ticker}               # Gauge
airas_tool_calls_total{tool_name, provider, status} # Counter
airas_tool_latency_seconds{tool_name, provider}     # Histogram
airas_rag_retrieval_latency_seconds{rag_level}      # Histogram
airas_rag_usable_ratio{agent_name}                  # Histogram
airas_crag_corrections_total{strategy}              # Counter
airas_llm_tokens_total{model, direction}            # Counter
airas_hallucination_warnings_total{agent_name}      # Counter
```

### 15.5 Key KPI Summary

```
Retrieval:
  Precision@5           target: >0.70
  CRAG usable_ratio     target: >0.65 after corrections
  RAGAS faithfulness    target: >0.85

Agent Quality:
  Groundedness (Galileo) target: >0.80 average
  Tool success rate      target: >95%
  Score variance (2 runs)target: Δ < 0.15

End-to-End:
  Full analysis time     target: <60 seconds
  Cost per analysis      target: <$1.00
  API error rate         target: <1%
```

### 15.6 Debugging Common Issues

| Symptom | Likely cause | How to diagnose | Fix |
|---------|-------------|-----------------|-----|
| Agents return score=0, confidence=0.1 | Tool calls all failing | Langfuse: check tool_call spans for errors | Verify API keys, check rate limit headers |
| High correction rate (>50%) in CRAG | Tickers not indexed in vector DB | Check airas_documents WHERE ticker=X count | Run SEC download + index for that ticker |
| Low groundedness in Galileo | Agent inventing facts | Galileo: see flagged_claims | Reduce temperature, improve system prompt |
| Analysis taking >90s | Claude rate limits causing retries | Langfuse: look for long gaps between spans | Reduce parallel agents or add API key capacity |
| Inconsistent recommendations | High temperature or low confidence context | Compare agent confidences across runs | Lower temperature, improve RAG quality |

---

## 16. Known Gaps & Improvement Roadmap

This section documents architectural limitations and concrete improvement opportunities, organized by priority and effort.

### 16.0 Concrete Example: Before & After Improvements 1-3

To understand the impact of improvements 1-3, walk through a single AAPL analysis end-to-end.

#### Current Flow (before improvements)

A request arrives: `POST /analyze { "ticker": "AAPL", "rag_level": "corrective" }`

**Context Building** — `ContextBuilder.build_all_contexts()` runs RAG queries for all 10 agents:

```
financial_analyst queries:
  Q1: "Revenue growth trends for AAPL"          → hits pgvector → 5 chunks
  Q2: "Operating margins and profitability AAPL" → hits pgvector → 5 chunks
  Q3: "Balance sheet debt and liquidity AAPL"    → hits pgvector → 5 chunks
  Q4: "Cash flow from operations AAPL"           → hits pgvector → 5 chunks

earnings_analysis queries:
  Q1: "Revenue growth trends for AAPL"           → hits pgvector → 5 chunks  ← DUPLICATE
  Q2: "EPS and earnings history for AAPL"        → hits pgvector → 5 chunks
  Q3: "Operating margins and profitability AAPL" → hits pgvector → 5 chunks  ← DUPLICATE

... 8 more agents, each with 2-4 queries
Total: ~30 pgvector queries. ~8-10 are duplicates across agents.
```

**Financial analyst receives fragmented context** — the income statement was split across 3 chunks during ingestion. The agent got chunks 1 and 3 but missed chunk 2:

```
Chunk 1: "Net sales: Products $298,085M, Services $85,200M, Total $383,285M..."
          ← chunk 2 (gross margin breakdown by segment) is MISSING
Chunk 3: "Cost of sales: Products $214,137M, Services $24,855M..."
```

The agent sees total revenue and cost, but not the per-segment gross margin that was in chunk 2.

**Evals produce no results** — `pytest evals/test_baseline.py`:
```python
for query in golden_set["queries"]:
    ground_truth_ids = query["ground_truth_chunks"]  # → []  EMPTY!
    if not ground_truth_ids:
        continue  # SKIPS EVERY QUERY — 0 queries evaluated
```

#### After Improvement 1: Golden Set Populated

Run `python populate_ground_truth.py`:

```
Processing q001: "What was Apple's total revenue in FY2023?"
  Retrieved 15 chunks, grading with Claude Haiku...
  ✓ Chunk abc123 [RELEVANT] score=0.91 "Net sales: Products $298,085M..."
  ✓ Chunk def456 [RELEVANT] score=0.87 "Total net sales of $383,285 million..."
  ✗ Chunk ghi789 [IRRELEVANT] "Apple Watch and wearables segment..."
  → Writing ground_truth_chunks: ["abc123", "def456", "jkl012"]

Processing q002: "What were Apple's operating margins in FY2023?"
  ...
```

`golden_set.json` now has real chunk IDs:
```json
{
  "id": "q001",
  "query": "What was Apple's total revenue in FY2023?",
  "ground_truth_chunks": ["abc123", "def456", "jkl012"]
}
```

Evals produce actual metrics:
```
pytest evals/test_baseline.py -v

BASELINE AGGREGATE METRICS
Queries evaluated: 10       ← was 0
Precision@5: 0.68 (±0.31)
Recall@5: 0.62 (±0.28)
MRR: 0.75
```

#### After Improvement 2: Cross-Agent Retrieval Cache

Same analysis request. `ContextBuilder` now has a dict cache:

```
financial_analyst queries:
  Q1: "Revenue growth trends for AAPL"          → MISS → pgvector → cached
  Q2: "Operating margins and profitability AAPL" → MISS → pgvector → cached
  Q3: "Balance sheet debt and liquidity AAPL"    → MISS → pgvector → cached
  Q4: "Cash flow from operations AAPL"           → MISS → pgvector → cached

earnings_analysis queries:
  Q1: "Revenue growth trends for AAPL"           → HIT ✓ (exact same query)
  Q2: "EPS and earnings history for AAPL"        → MISS → pgvector
  Q3: "Operating margins and profitability AAPL" → HIT ✓

risk_assessment queries:
  Q1: "Risk factors and disclosures for AAPL"    → MISS
  Q2: "Balance sheet debt and liquidity AAPL"    → HIT ✓ (financial_analyst ran this)
```

Before: ~30 pgvector queries, ~12s retrieval time.
After: ~20 pgvector queries (~10 cache hits), ~8s retrieval time.

Cache is a plain dict — created at start of `build_all_contexts()`, garbage collected when it returns. Zero infrastructure.

#### After Improvement 3: Full Section Reconstruction

Financial analyst's top-5 similarity search found a chunk from `(AAPL, income_statement, FY2023)`. Now `retrieve_full_section()` pulls ALL chunks for that section:

```python
# Before: top-5 similarity returns scattered chunks from different sections
retrieve("Revenue growth trends for AAPL", top_k=5)
# → [income_stmt chunk 1, risk_factors chunk 7, income_stmt chunk 3, ...]

# After: once a relevant section is identified, pull the complete section
retrieve_full_section(ticker="AAPL", section="income_statement", fiscal_period="FY2023")
# → [chunk_1, chunk_2, chunk_3, chunk_4]  ALL income statement chunks, in order
```

The agent now sees the complete income statement:
```
Chunk 1: "Net sales: Products $298,085M, Services $85,200M, Total $383,285M..."
Chunk 2: "Gross margin: Products 36.5%, Services 70.8%, Overall 44.1%..."  ← was MISSING
Chunk 3: "Cost of sales: Products $214,137M, Services $24,855M..."
Chunk 4: "Operating income: $114,301M, Operating margin: 29.8%..."
```

No more fragmented financial data.

#### Combined Impact

| Metric | Before | After |
|--------|--------|-------|
| Eval queries actually tested | 0 / 10 | 10 / 10 |
| pgvector calls per analysis | ~30 | ~20 (33% reduction) |
| Context retrieval time | ~12s | ~8s |
| Financial data completeness | Fragmented (top-k random chunks) | Full sections, ordered by chunk_index |

---

### 16.1 RAG Improvements (Inspired by Mintlify's ChromaFs Architecture)

Mintlify built a virtual filesystem over their Chroma vector database, letting AI agents explore documentation using familiar UNIX-style navigation (`ls`, `cat`, `grep`) instead of blind vector queries. Their approach achieved P90 boot times of ~100ms (down from ~46s with sandboxes) at zero marginal cost. Several of their techniques map directly to AIRAS improvements:

#### Improvement 1: Filing Structure Index ("Path Tree")
**Problem:** Agents fire blind vector queries without knowing what filings exist for a ticker. A query for TSLA options data hits Apple's filings because the agent doesn't know TSLA isn't indexed.

**Mintlify's approach:** They build a compressed directory tree (`__path_tree__`) of all docs, stored as gzipped JSON in Chroma. On init, it decompresses into an in-memory `Set<string>` of paths and a `Map<string, string[]>` of directory→children. Agents see the full structure before requesting content.

**AIRAS equivalent:** Build a filing tree index stored as a lightweight metadata table or cached JSON:
```
AAPL/
  10-K/
    FY2024/ [income_statement, balance_sheet, cash_flow, md_a, risk_factors]
    FY2023/ [...]
  10-Q/
    Q1-2024/ [...]

TSLA/
  10-K/
    FY2024/ [...]
```

**Benefit:** Retrieval becomes targeted ("get AAPL FY2023 balance_sheet") instead of hoping vector similarity finds the right filing. Agents can also see what's NOT available and skip RAG for missing data.

**Priority:** HIGH | **Effort:** Medium

---

#### Improvement 2: Full Section Reconstruction ("Chunk Reassembly")
**Problem:** AIRAS returns top-k chunks (512 tokens each) which often fragments financial data. A balance sheet split across 3 chunks might only return 2, missing critical line items like total assets or shareholder equity.

**Mintlify's approach:** When agents `cat /path/file.mdx`, ChromaFs fetches ALL chunks with matching path, sorts by `chunk_index`, and joins them into a complete document. Results are cached to prevent repeated DB hits.

**AIRAS equivalent:** Add a `retrieve_full_section()` method:
```python
def retrieve_full_section(self, ticker: str, section: str, fiscal_period: str) -> str:
    """Fetch ALL chunks for a given ticker+section+period, sorted by chunk_index."""
    filters = MetadataFilters(filters=[
        MetadataFilter(key="ticker", value=ticker),
        MetadataFilter(key="section", value=section),
        MetadataFilter(key="fiscal_period", value=fiscal_period),
    ])
    # Fetch all matching chunks (not just top-k)
    all_chunks = self.index.as_retriever(similarity_top_k=100, filters=filters).retrieve("")
    # Sort by chunk_index and concatenate
    sorted_chunks = sorted(all_chunks, key=lambda n: n.metadata.get("chunk_index", 0))
    return "\n".join(c.text for c in sorted_chunks)
```

**Benefit:** Agents see complete financial statements instead of fragments. Particularly impactful for financial_analyst and earnings_analysis agents.

**Priority:** HIGH | **Effort:** Low-Medium

---

#### Improvement 3: Cross-Agent Retrieval Cache
**Problem:** 10 parallel agents each run 2-4 RAG queries, many overlapping (multiple agents query the same 10-K income statement for the same ticker). Every query hits pgvector independently. A full AAPL analysis might make 30+ vector queries when 10-15 unique queries would suffice.

**Mintlify's approach:** They cache file contents after first fetch and use Redis for bulk prefetch, preventing repeated database hits for the same content.

**AIRAS equivalent:** Add a per-analysis cache in `ContextBuilder`:
```python
class ContextBuilder:
    def __init__(self):
        self._cache: dict[tuple[str, str, str], list[NodeWithScore]] = {}

    def _cached_retrieve(self, query, ticker, sections) -> list[NodeWithScore]:
        cache_key = (query, ticker, tuple(sorted(sections or [])))
        if cache_key not in self._cache:
            self._cache[cache_key] = self.retriever.retrieve(query, ticker=ticker, sections=sections)
        return self._cache[cache_key]
```

**Benefit:** Could cut pgvector calls by 40-60%, reducing latency and database load during peak analysis.

**Priority:** HIGH | **Effort:** Low

---

#### Improvement 4: Agent-Driven Exploration Tools
**Problem:** Agents use static `RAG_QUERIES` (predefined query templates) that can't adapt to what data is actually available. If a new filing has a novel section (e.g., "segment restructuring"), no agent's RAG_QUERIES will find it.

**Mintlify's core insight:** Let agents explore documentation like developers explore codebases — browse the structure, then drill into specific content — rather than relying on a single vector query to find everything.

**AIRAS equivalent:** Give agents two new tools alongside their existing financial API tools:
```python
{
    "name": "list_available_filings",
    "description": "List all SEC filings available for a ticker with their sections and dates",
    "input_schema": {
        "properties": {
            "ticker": {"type": "string"}
        }
    }
}

{
    "name": "retrieve_filing_section",
    "description": "Retrieve the full text of a specific SEC filing section",
    "input_schema": {
        "properties": {
            "ticker": {"type": "string"},
            "filing_type": {"type": "string", "enum": ["10-K", "10-Q"]},
            "fiscal_period": {"type": "string"},
            "section": {"type": "string"}
        }
    }
}
```

Agents could then:
1. Call `list_available_filings("AAPL")` → see what's indexed
2. Call `retrieve_filing_section("AAPL", "10-K", "FY2023", "income_statement")` → get full section
3. Make targeted follow-up requests based on what they find

**Benefit:** Most transformative improvement — replaces blind retrieval with intelligent navigation. Eliminates the "CRAG corrects bad queries" pattern by making queries precise from the start.

**Priority:** VERY HIGH | **Effort:** High (requires rethinking agent loop — shifts from eager context-building to lazy tool-driven retrieval)

---

#### Improvement 5: Two-Stage Strict-First Filtering
**Problem:** IntermediateRetriever's fallback chain (ticker+sections → ticker-only → no filters) falls back aggressively, which can return noisy cross-ticker results.

**Mintlify's approach:** Their grep uses a two-stage filter — coarse metadata query to narrow candidates, then fine in-memory matching on the narrowed set.

**AIRAS equivalent:** Invert the retrieval approach:
1. Always start with strict metadata filtering (ticker + filing_type + date_range)
2. Run vector similarity ONLY within that narrowed set
3. If narrowed set is empty, broaden ONE dimension at a time (remove date_range → remove filing_type → remove ticker)

This is the opposite of the current "try strict, fall back wide" pattern and would produce more precise results with less noise.

**Priority:** Medium | **Effort:** Medium

---

#### Improvement 6: Lazy Context Loading
**Problem:** `ContextBuilder.build_all_contexts()` eagerly fetches ALL RAG context for ALL agents upfront before any agent runs. If an agent has 4 RAG_QUERIES and only needs 2 (because the first 2 give high-confidence results), the other 2 queries are wasted.

**Mintlify's approach:** They use lazy file pointers for large files — content is only fetched when an agent actually reads it, not pre-loaded.

**AIRAS equivalent:** Make context retrieval lazy — agents request context during their tool-use loop, not all upfront. Combined with the exploration tools (Improvement 4), agents only pull data they actually need.

**Priority:** Medium | **Effort:** High (requires restructuring the gather_context → fan_out → agent pipeline)

---

### 16.2 Evaluation Improvements

#### Fix 1: Populate Golden Set Ground Truth (P0 — Unblocks All Retrieval Evals)
The `golden_set.json` has empty `ground_truth_chunks` arrays for all 10 queries. Without chunk IDs, all retrieval metric tests skip every query. Run `generate_golden_set.py --find-chunks` for each query to identify the correct chunk IDs from pgvector, then manually verify and populate them.

#### Fix 2: Expand Golden Dataset (P0)
- Current: 10 queries, 1 ticker (AAPL), 0 "hard" difficulty queries
- Target: 50+ queries, 5+ tickers, mix of easy/medium/hard
- Add adversarial queries: wrong ticker names, ambiguous fiscal periods, queries about data not in SEC filings
- Add multi-hop queries: "Compare AAPL and MSFT gross margins for FY2023" (requires cross-ticker retrieval)

#### Fix 3: Agent Scoring Consistency Evals (P1)
No tests evaluate individual agent quality. Need:
- **Consistency test:** Run same ticker through same agent 5x → score variance should be < 0.15
- **Known-answer test:** AAPL revenue = $383.3B → financial_analyst score should reflect this accurately
- **Confidence calibration:** High-confidence answers should be more accurate than low-confidence ones
- **Per-agent accuracy:** Each agent tested against agent-specific ground truth queries

#### Fix 4: End-to-End Recommendation Evals (P1)
No tests evaluate whether final BUY/SELL recommendations are any good. Need:
- **Historical backtesting:** Run analysis on past dates, compare recommendation to subsequent price movement
- **Consistency test:** Similar companies (e.g., AAPL vs MSFT) should get scores in similar ranges
- **Stress test:** Agent handles missing data gracefully (new IPO with limited filings)

#### Fix 5: CI/CD Integration (P2)
- GitHub Actions workflow running `pytest evals/ -m "not slow"` on every PR
- Weekly scheduled run of full eval suite including RAGAS
- Result tracking over time to detect regressions
- Alert on metric drops below thresholds

#### Fix 6: Eval Dashboard (P3)
- Script to read `results/*.json` and generate markdown/HTML report
- Trend charts: retrieval metrics over time
- Per-agent quality heatmap
- Cost tracking trends

---

### 16.3 Architecture Improvements

#### Arch 1: BM25 + Vector Hybrid Search
Currently retrieval is pure vector similarity. Adding BM25 keyword matching would improve recall for exact financial terms (ticker symbols, specific metric names, exact dollar amounts) that embeddings can miss. Libraries like `rank_bm25` or pgvector's upcoming full-text search integration could enable this without additional infrastructure.

#### Arch 2: Dedicated Reranking Model
AdvancedRetriever uses Claude as a cross-encoder reranker, which is expensive and slow. A dedicated reranking model (Cohere Rerank API at ~$0.001/query, or open-source BGE-Reranker) would be 10-100x cheaper and 2-5x faster while achieving similar quality.

#### Arch 3: Streaming Agent Responses
Currently the API blocks until all 10 agents complete (~45-60s). Streaming partial results (agent-by-agent as they finish) would improve perceived latency. FastAPI supports SSE (Server-Sent Events) natively.

#### Arch 4: Agent Memory / Cross-Analysis Learning
Each analysis is stateless — running AAPL twice doesn't benefit from the first run. A lightweight per-ticker cache of recent analysis results could:
- Skip re-analyzing tickers analyzed within the last N hours
- Track score trends over time ("AAPL score went from 0.72 → 0.65 over 2 weeks")
- Enable "what changed?" analysis

---

### 16.4 Priority Summary

| # | Improvement | Impact | Effort | Category |
|---|------------|--------|--------|----------|
| 1 | Populate golden set ground truth | Unblocks all evals | Low | Eval |
| 2 | Cross-agent retrieval cache | High (latency + cost) | Low | RAG |
| 3 | Full section reconstruction | High (data quality) | Low-Med | RAG |
| 4 | Expand golden dataset | High (eval coverage) | Medium | Eval |
| 5 | Filing structure index | High (precision) | Medium | RAG |
| 6 | Agent scoring consistency evals | High (quality signal) | Medium | Eval |
| 7 | Agent-driven exploration tools | Very High (architecture) | High | RAG |
| 8 | E2E recommendation evals | High (correctness signal) | Medium | Eval |
| 9 | Two-stage strict-first filtering | Medium (precision) | Medium | RAG |
| 10 | CI/CD eval integration | Medium (automation) | Medium | Eval |
| 11 | BM25 hybrid search | Medium (recall) | Medium | Architecture |
| 12 | Dedicated reranking model | Medium (cost + speed) | Low-Med | Architecture |
| 13 | Lazy context loading | Medium (efficiency) | High | RAG |
| 14 | Streaming agent responses | Medium (UX) | Medium | Architecture |
| 15 | Agent memory | Low-Med (quality over time) | High | Architecture |

Items 1-3 are quick wins that should be done first. Item 7 is the most transformative long-term improvement.
