# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AIRAS V3 (AI-powered Investment Research & Analysis System) backend. A Python financial analysis system that downloads SEC filings, chunks them with metadata, stores embeddings in Supabase (PostgreSQL + pgvector), and enables RAG queries over financial data using Claude and GPT-4. Features a 10-agent analysis system orchestrated via LangGraph with deep Langfuse tracing.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Download SEC filings for a company
python scripts/download_sec_filings.py --ticker AAPL
# Flags: --force (rebuild all), --clear (clear index), --file <path> (index specific file)

# Build vector index from downloaded filings
python scripts/smart_build_index.py
# Flags: --force (rebuild all), --clear (wipe index), --file <path> (index specific file)

# Test RAG queries
python scripts/test_rag.py

# Run full investment analysis
python scripts/run_analysis.py --ticker AAPL
# Flags: --query "question" (focused analysis), --full (force all agents with query),
#         --rag-level basic|intermediate|advanced, --sequential, --output results.json

# Run tests
pytest --asyncio-mode=auto
```

## Architecture

**Data pipeline:** SEC filings -> `IngestionPipeline` [`SentenceSplitter` (512-token chunks) -> `SECMetadataTransform` (metadata enrichment) -> OpenAI embeddings (text-embedding-3-small, 1536 dim)] -> Supabase pgvector (`data_airas_documents` table) -> LlamaIndex `VectorStoreIndex` -> RAG queries

**Analysis pipeline:** Query -> Router (classifies query, selects agents) -> Context Gathering (RAG queries per agent) -> 10 Parallel Agents (Claude tool-use loops) -> Synthesis (weighted scoring + thesis) -> InvestmentRecommendation

### Key Components

- **`config/settings.py`** — Pydantic BaseSettings loaded from `.env`. Singleton via `get_settings()`. All API keys, DB credentials, model names, and RAG parameters live here.
- **`src/rag/supabase_rag.py`** — `SupabaseRAG` class wrapping LlamaIndex's `IngestionPipeline`, `VectorStoreIndex`, and `PGVectorStore`. Pipeline runs `SentenceSplitter` -> `SECMetadataTransform` -> embedding model. Core methods: `setup_database()`, `build_index(documents)`, `load_index()`, `query(query_text)`.
- **`src/rag/metadata_extractor.py`** — `SECMetadataExtractor` parses filenames (`TICKER_TYPE_DATE.txt` pattern) and content to detect financial sections, metrics, fiscal periods, and content type. `SECMetadataTransform` extends LlamaIndex `BaseExtractor` to work as an `IngestionPipeline` transformation step.
- **`src/rag/retrieval.py`** — Progressive RAG: `BasicRetriever` (vector similarity), `IntermediateRetriever` (metadata filtering + query rewriting), `AdvancedRetriever` (HyDE + reranking + multi-query). Configurable via `--rag-level`.
- **`src/models/structured_outputs.py`** — Pydantic models: `AgentOutput` (agent analysis result with score -1 to 1), `FinancialMetrics`, `InvestmentRecommendation` (STRONG BUY through STRONG SELL), `RatioResult`, `CompanyComparison`.
- **`src/tools/fmp_client.py`** — FMP (Financial Modeling Prep) REST API client with 5-minute response caching and retry logic. Wraps endpoints: `get_quote`, `get_ratios`, `get_key_metrics`, `get_historical_prices`, `get_technical_indicator`, `get_insider_trades_fmp`, `get_insider_stats`, `get_price_target_summary`, `get_stock_grades`, `get_stock_news`. Returns `None` if `FMP_API_KEY` not configured, enabling graceful fallback.
- **`src/tools/finnhub_client.py`** — Finnhub API client for insider transactions and company news. Free tier: 60 calls/min. Returns `None` if `FINNHUB_API_KEY` not set.
- **`src/tools/tradier_client.py`** — Tradier API client for options chains and expirations. Free sandbox. Returns `None` if `TRADIER_API_TOKEN` not set.
- **`src/tools/financial_tools.py`** — Eight tools with Claude function-calling schemas. Multi-provider fallback chains: FMP -> Finnhub -> yfinance. Technical indicators use FMP historical prices + `ta` library (no Premium tier needed). Options use Tradier -> yfinance. Tools: `calculate_financial_ratio`, `compare_companies`, `get_stock_price`, `get_technical_indicators`, `get_insider_trades`, `get_options_data`, `get_analyst_ratings`, `get_social_sentiment`. Dispatched via `execute_tool()`.
- **`src/utils/langfuse_setup.py`** — Optional Langfuse tracing. `@trace_agent(agent_name)` decorator for agent execution monitoring.
- **`src/utils/llama_setup.py`** — `configure_llama_index()` sets global LlamaIndex settings (embedding model, LLM, chunk params).

### Agent System (`src/agents/`)

- **`state.py`** — `AnalysisState` TypedDict for LangGraph. Uses `Annotated[list, operator.add]` reducers for parallel agent output merging.
- **`graph.py`** — LangGraph `StateGraph` with nodes: route_query -> gather_context -> fan-out to N agents (parallel via `Send`) -> synthesis -> END. Built via `build_analysis_graph()`.
- **`router.py`** — Smart query router: classifies query into categories (financial, technical, risk, sentiment, insider, competitive) and selects relevant agents. Full analysis if no query or `--full`.
- **`base_agent.py`** — `BaseAgent` class with Claude tool-use loop. Each agent subclass defines 5 class attributes: `AGENT_NAME`, `SYSTEM_PROMPT`, `RAG_QUERIES`, `TOOLS`, `SECTIONS`. The `analyze()` method runs the full loop: build user message with RAG context -> Claude API call -> handle tool_use blocks -> parse JSON output -> return `AgentOutput`.
- **`context.py`** — `ContextBuilder` gathers RAG context for all active agents. Uses the configured retrieval level (basic/intermediate/advanced). Runs all RAG queries upfront before agents execute.
- **`prompts.py`** — System prompts for all 10 agents. Each prompt has: expert persona, scoring rubric, analysis method, output schema, few-shot example.
- **`tracing.py`** — `TracingManager` for deep Langfuse spans (RAG queries, LLM calls, tool calls, agent scores, recommendation scores).
- **`synthesis.py`** — Weighted scoring: 10 agents with fixed weights (totaling 100%). Category scores (financial, technical, sentiment, risk). Thesis generation via Claude.

### The 10 Agents

| # | Agent | Weight | Data Source | File |
|---|-------|--------|-------------|------|
| 1 | Financial Analyst | 20% | RAG + FMP/yfinance | `financial_analyst.py` |
| 2 | News Sentiment | 12% | RAG | `news_sentiment.py` |
| 3 | Technical Analyst | 15% | FMP prices + ta lib | `technical_analyst.py` |
| 4 | Risk Assessment | 10% | FMP/yfinance | `risk_assessment.py` |
| 5 | Competitive Analysis | 10% | RAG | `competitive_analysis.py` |
| 6 | Insider Activity | 8% | FMP/Finnhub/yfinance | `insider_activity.py` |
| 7 | Options Analysis | 5% | Tradier/yfinance | `options_analysis.py` |
| 8 | Social Sentiment | 3% | StockTwits+Reddit+Finnhub/FMP News | `social_sentiment.py` |
| 9 | Earnings Analysis | 7% | RAG | `earnings_analysis.py` |
| 10 | Analyst Ratings | 10% | FMP/yfinance | `analyst_ratings.py` |

### Not Yet Implemented

- `src/api/` — FastAPI REST endpoints
- No linter configuration
- Evals / test suite for agent scoring consistency

## Environment

Requires `.env` with: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `SUPABASE_URL`, `SUPABASE_KEY`, `POSTGRES_CONNECTION_STRING`, and optionally `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `FMP_API_KEY` (Financial Modeling Prep), `FINNHUB_API_KEY` (free at finnhub.io), `TRADIER_API_TOKEN` (free sandbox at developer.tradier.com). Python 3.11.

## Conventions

- Configuration via Pydantic BaseSettings with env vars; never hardcode secrets
- Scripts use `sys.path.insert(0, str(Path(__file__).parent.parent))` for imports from project root
- Logging via `config/logging_config.setup_logging()` with rotating file handler (10MB, 5 backups) to `logs/airas.log`
- Metadata stored as JSONB in PostgreSQL for flexible filtering
- All structured data uses Pydantic models for validation
- Database table for vectors: `data_airas_documents` with HNSW index on embeddings and GIN index on metadata
- Agents are subclasses of `BaseAgent` with 5 class attributes; no method overrides needed
- Agent orchestration via LangGraph `StateGraph` with `Send()` for parallel fan-out
