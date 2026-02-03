# AIRAS V3 - Architecture Reference

Detailed documentation for every module in the backend.

---

## Overview

```
SEC EDGAR  -->  download_sec_filings.py  -->  data/raw/*.txt
                                                   |
                                            SmartChunker
                                          (512 tokens, metadata)
                                                   |
                                         OpenAI Embeddings
                                       (text-embedding-3-small)
                                                   |
                                     Supabase pgvector (data_airas_documents)
                                                   |
                                      LlamaIndex VectorStoreIndex
                                                   |
                                          RAG Query Engine
                                                   |
                                          Natural Language Q&A
```

---

## config/settings.py

**Purpose:** Central configuration loaded from `.env` via Pydantic `BaseSettings`.

**Key class:** `Settings`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `openai_api_key` | str | required | OpenAI API key |
| `anthropic_api_key` | str | required | Anthropic API key |
| `supabase_url` | str | required | Supabase project URL |
| `supabase_key` | str | required | Supabase anon key |
| `postgres_connection_string` | str | required | PostgreSQL URI |
| `langfuse_public_key` | str | None | Langfuse public key (optional) |
| `langfuse_secret_key` | str | None | Langfuse secret key (optional) |
| `langfuse_host` | str | `https://cloud.langfuse.com` | Langfuse host |
| `environment` | str | `development` | `development` or `production` |
| `log_level` | str | `INFO` | Python logging level |
| `api_port` | int | `8000` | FastAPI server port |
| `sec_user_email` | str | - | Email for SEC EDGAR downloads |
| `chunk_size` | int | `512` | Token count per chunk |
| `chunk_overlap` | int | `50` | Token overlap between chunks |
| `top_k` | int | `5` | Number of chunks retrieved per query |
| `openai_model` | str | `gpt-4-turbo-preview` | LLM for RAG responses |
| `openai_embedding_model` | str | `text-embedding-3-small` | Embedding model (1536 dim) |
| `claude_model` | str | `claude-3-5-sonnet-20241022` | Claude model for agents |
| `data_dir` | Path | `data` | Base data directory |
| `raw_dir` | Path | `data/raw` | Raw filing text files |
| `logs_dir` | Path | `logs` | Log file directory |

**Usage:**

```python
from config.settings import get_settings

settings = get_settings()  # Singleton - cached after first call
print(settings.supabase_url)
print(settings.chunk_size)
```

---

## config/logging_config.py

**Purpose:** Configure Python logging with console + rotating file output.

**Key function:** `setup_logging(log_level, logs_dir)`

**Behavior:**
- Console output to stdout with timestamps
- File output to `logs/airas.log` with rotation (10MB max, 5 backups)
- Suppresses noisy HTTP libraries (httpx, httpcore, openai, anthropic, urllib3)

**Usage:**

```python
from config.logging_config import setup_logging

setup_logging("INFO")  # or "DEBUG", "WARNING", etc.
```

---

## src/rag/metadata_extractor.py

**Purpose:** Extract structured metadata from SEC filings for filtering.

**Key class:** `SECMetadataExtractor`

### Filename Parsing

Parses filenames in the format `TICKER_TYPE_DATE.txt`:

```python
extractor = SECMetadataExtractor()
meta = extractor.extract_from_filename("AAPL_10K_2024-11-01.txt")
# {'source_file': 'AAPL_10K_2024-11-01.txt', 'ticker': 'AAPL',
#  'document_type': '10-K', 'filing_date': '2024-11-01'}
```

### Content Analysis

Detects from chunk text:

| Detection | Values |
|-----------|--------|
| **Section** | `income_statement`, `balance_sheet`, `cash_flow`, `shareholders_equity` |
| **Metrics** | `revenue`, `income`, `assets`, `liabilities`, `equity`, `cash_flow`, `debt`, `margin` |
| **Fiscal period** | Patterns like "fiscal year 2024", "FY2023", "Q1 2024" |
| **Content type** | `financial_data` (has numbers + metrics) or `narrative` |

```python
enhanced = extractor.extract_from_content(chunk_text, base_metadata)
# Adds: section, metric_types, fiscal_period, content_type, has_numbers, word_count
```

---

## src/rag/smart_chunker.py

**Purpose:** Split documents into chunks with rich metadata.

**Key class:** `SmartChunker`

**Parameters:**
- `chunk_size`: 512 tokens (default)
- `chunk_overlap`: 50 tokens (default)

**How it works:**

1. Extract base metadata from filename (ticker, type, date)
2. Split text using LlamaIndex `SentenceSplitter`
3. For each chunk, extract content metadata (section, metrics, fiscal period)
4. Add position metadata (chunk_index, total_chunks, chunk_size)
5. Return list of `Document` objects

**Metadata attached to each chunk:**

```json
{
  "source_file": "AAPL_10K_2024-11-01.txt",
  "ticker": "AAPL",
  "document_type": "10-K",
  "filing_date": "2024-11-01",
  "section": "income_statement",
  "metric_types": ["revenue", "margin"],
  "fiscal_period": "fiscal year 2024",
  "content_type": "financial_data",
  "has_numbers": true,
  "word_count": 142,
  "chunk_index": 12,
  "total_chunks": 215,
  "chunk_size": 487
}
```

**Usage:**

```python
from src.rag.smart_chunker import SmartChunker

chunker = SmartChunker(chunk_size=512, chunk_overlap=50)

# Single file
chunks = chunker.chunk_document(filepath, content)

# Entire directory
all_chunks = chunker.process_directory(Path("data/raw"))
```

---

## src/rag/supabase_rag.py

**Purpose:** Complete RAG pipeline using Supabase pgvector as the vector store.

**Key class:** `SupabaseRAG`

### Methods

| Method | Description |
|--------|-------------|
| `setup_database()` | Connect to Supabase via `PGVectorStore` |
| `build_index(documents)` | Generate embeddings and store chunks |
| `load_index()` | Load existing index from Supabase |
| `create_query_engine(top_k)` | Create a retrieval + generation engine |
| `query(query_text)` | Run a natural language query, returns string answer |

### Data Flow

```
setup_database()  -->  PGVectorStore (Supabase connection)
                              |
build_index(docs)  -->  VectorStoreIndex.from_documents()
                         (embeds + stores)
                              |
load_index()       -->  VectorStoreIndex.from_vector_store()
                         (connects to existing data)
                              |
create_query_engine() --> VectorIndexRetriever + RetrieverQueryEngine
                              |
query("question")  -->  Retrieves top-k chunks --> LLM generates answer
```

### Database Table

The `data_airas_documents` table:

| Column | Type | Description |
|--------|------|-------------|
| `id` | TEXT | Primary key |
| `text` | TEXT | Chunk text content |
| `metadata` | JSONB | All metadata fields (filterable) |
| `embedding` | vector(1536) | OpenAI embedding |
| `created_at` | TIMESTAMP | Insertion time |

Indexes:
- **HNSW** on `embedding` for fast vector similarity search
- **GIN** on `metadata` for fast JSONB filtering
- B-tree indexes on `metadata->>'ticker'`, `metadata->>'section'`, `metadata->>'document_type'`, `metadata->>'fiscal_period'`

---

## src/utils/llama_setup.py

**Purpose:** Configure global LlamaIndex settings (embedding model, LLM, chunk params).

**Key function:** `configure_llama_index()`

Must be called once before using any RAG functionality. Sets:

- `Settings.embed_model` = OpenAI `text-embedding-3-small` (1536 dimensions)
- `Settings.llm` = OpenAI `gpt-4-turbo-preview` (temperature 0.1)
- `Settings.chunk_size` = from settings (default 512)
- `Settings.chunk_overlap` = from settings (default 50)

```python
from src.utils.llama_setup import configure_llama_index
configure_llama_index()  # Call once at startup
```

---

## src/utils/langfuse_setup.py

**Purpose:** Optional monitoring/tracing via Langfuse.

### Functions

**`setup_langfuse()`** - Initialize the Langfuse client. Returns `None` if keys are not configured (graceful degradation).

**`@trace_agent(agent_name)`** - Decorator for tracing agent execution. Automatically creates traces with agent name and ticker metadata.

```python
from src.utils.langfuse_setup import setup_langfuse, trace_agent

# Initialize (call once at startup)
client = setup_langfuse()  # Returns None if not configured

# Decorate agent functions
@trace_agent("financial_health")
def analyze(self, ticker: str):
    # ... agent logic ...
    pass
```

---

## src/models/structured_outputs.py

**Purpose:** Pydantic models ensuring all data is typed and validated.

### Models

**`AgentOutput`** - Standard return type from any analysis agent.

| Field | Type | Description |
|-------|------|-------------|
| `agent_name` | str | Agent identifier |
| `ticker` | str | Company analyzed |
| `score` | float [-1, 1] | Bearish (-1) to bullish (+1) |
| `confidence` | float [0, 1] | How confident the agent is |
| `metrics` | Dict | Key metrics analyzed |
| `strengths` | List[str] | Top strengths (max 3) |
| `weaknesses` | List[str] | Top weaknesses (max 3) |
| `summary` | str | One-sentence summary |
| `sources` | List[str] | Data sources used |
| `timestamp` | str | ISO timestamp |

**`FinancialMetrics`** - Structured financial data (income statement, balance sheet, cash flow, ratios).

**`InvestmentRecommendation`** - Complete synthesis output with recommendation (`STRONG BUY` through `STRONG SELL`), category scores, bullish/bearish factors, risks, and thesis.

**`RatioResult`** - Output from `calculate_financial_ratio` tool.

**`CompanyComparison`** - Output from `compare_companies` tool.

---

## src/tools/financial_tools.py

**Purpose:** Three yfinance-backed tools that can be called by agents via Claude function calling.

### Tools

**`calculate_financial_ratio(ratio_type, ticker, period)`**

Supported ratios: `pe_ratio`, `roe`, `debt_to_equity`, `profit_margin`, `current_ratio`

```python
result = calculate_financial_ratio("pe_ratio", "AAPL")
# {'ratio_name': 'pe_ratio', 'ticker': 'AAPL', 'value': 28.5,
#  'interpretation': 'Fairly valued'}
```

**`compare_companies(tickers, metric, period)`**

Supported metrics: `revenue`, `profit_margin`, `roe`, `pe_ratio`, `market_cap`

```python
result = compare_companies(["AAPL", "MSFT"], "revenue")
# {'metric': 'revenue', 'winner': 'AAPL', 'values': [394.33, 211.92]}
```

**`get_stock_price(ticker, date=None)`**

Current or historical price data.

```python
result = get_stock_price("NVDA")          # Current
result = get_stock_price("AAPL", "2024-06-15")  # Historical
```

### Tool Registry

`FINANCIAL_TOOLS` is a list of OpenAI-compatible function schemas for Claude tool_use integration.

`execute_tool(tool_name, tool_input)` dispatches to the correct function by name.

---

## scripts/

### download_sec_filings.py

Downloads SEC filings from EDGAR, cleans HTML to text, saves to `data/raw/`.

**Class:** `SECDownloader`
- `download_filings(ticker, filing_types, num_filings)` - Download and process
- `_process_downloads(ticker, filing_type)` - Convert HTML to clean text
- `_clean_html(html_content)` - Strip tags via BeautifulSoup

### build_index.py

Basic index builder. Reads all `.txt` files, chunks them, generates embeddings, stores in Supabase. Re-indexes everything each run.

### smart_build_index.py

Incremental index builder. Tracks which files are already indexed in Supabase and only processes new files. Supports `--force`, `--clear`, and `--file` flags.

### test_rag.py

Runs predefined test queries against the RAG system to verify the pipeline.

### list_indexed_10k.py

Lists all 10-K filings currently stored in the vector index with metadata.

---

## Database Schema

### Tables

| Table | Description |
|-------|-------------|
| `companies` | 10 pre-loaded companies with ticker, name, sector, industry, market_cap |
| `sec_filings` | Filing metadata (ticker, type, period, date, accession number) |
| `data_airas_documents` | Vector store: text chunks + JSONB metadata + 1536-dim embeddings |
| `financial_metrics` | Structured financials (income, balance sheet, cash flow, ratios) |
| `analysis_results` | Cached analysis with scores, recommendation, thesis |
| `agent_executions` | Per-agent execution logs with timing and error tracking |

### Views

| View | Description |
|------|-------------|
| `latest_analyses` | Most recent analysis per ticker |
| `company_overview` | Company info + latest metrics + latest analysis |
| `agent_performance` | Execution counts, avg time, error rates per agent |

### Helper Functions

| Function | Description |
|----------|-------------|
| `search_documents(ticker, section, limit)` | Search chunks by ticker and section |
| `get_latest_metrics(ticker)` | Get most recent financial metrics |
| `clean_expired_analyses()` | Delete expired cache entries |

---

## Not Yet Implemented

These directories have `__init__.py` but no implementation yet:

- **`src/agents/`** - 11 specialized analysis agents (financial health, valuation, growth, etc.)
- **`src/api/`** - FastAPI REST endpoints
- No main application entry point (`main.py` / `app.py`)

See `docs/Airas_v3_docs_2_agents_tools.md` for the agent implementation guide.
