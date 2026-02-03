# AIRAS V3 - Usage Examples

Complete reference for all CLI scripts and Python API usage.

---

## 1. Download SEC Filings

**Script:** `scripts/download_sec_filings.py`

Downloads SEC filings from EDGAR, converts HTML to clean text, and saves to `data/raw/`.

### Examples

```bash
# Single company, default 3 annual filings
uv run uv run python scripts/download_sec_filings.py --ticker AAPL

# Single company, quarterly filings
uv run uv run python scripts/download_sec_filings.py --ticker AAPL --type 10-Q

# Single company, 5 filings
uv run uv run python scripts/download_sec_filings.py --ticker MSFT --type 10-K --amount 5

# Multiple companies at once
uv run uv run python scripts/download_sec_filings.py --tickers AAPL MSFT GOOGL TSLA --type 10-K --amount 2

# 8-K filings (current reports)
uv run uv run python scripts/download_sec_filings.py --ticker NVDA --type 8-K --amount 3
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--ticker` | - | Single ticker symbol |
| `--tickers` | - | Multiple tickers (space-separated) |
| `--type` | `10-K` | Filing type: `10-K`, `10-Q`, `8-K` |
| `--amount` | `3` | Number of filings to download per ticker |

### Output

Files are saved as `TICKER_TYPE_DATE.txt` in `data/raw/`:

```
data/raw/
  AAPL_10K_2024-11-01.txt
  AAPL_10K_2023-11-03.txt
  MSFT_10K_2024-07-30.txt
```

---

## 2. Build Vector Index

**Script:** `scripts/build_index.py`

Reads all `.txt` files from `data/raw/`, chunks them, generates embeddings, and stores in Supabase.

### Example

```bash
uv run python scripts/build_index.py
```

### What It Does

1. Finds all `.txt` files in `data/raw/`
2. Chunks each file using `SmartChunker` (512 tokens, 50 overlap)
3. Attaches metadata to each chunk (ticker, section, metrics, fiscal period)
4. Generates embeddings via OpenAI `text-embedding-3-small`
5. Stores everything in the `data_airas_documents` table

### Sample Output

```
======================================================================
BUILDING AIRAS INDEX
======================================================================

Found 4 files:
   - AAPL_10K_2024-11-01.txt
   - AAPL_10K_2023-11-03.txt
   - MSFT_10K_2024-07-30.txt
   - TSLA_10K_2024-01-29.txt

Processing documents into chunks...
Created 892 chunks

Building index (generating embeddings)...

======================================================================
INDEX BUILT SUCCESSFULLY!
======================================================================
```

---

## 3. Smart Build Index (Incremental)

**Script:** `scripts/smart_build_index.py`

Like `build_index.py` but tracks which files are already indexed and only processes new ones.

### Examples

```bash
# Only index new files (skip already-indexed)
uv run python scripts/smart_build_index.py

# Force re-index everything
uv run python scripts/smart_build_index.py --force

# Clear entire index and rebuild
uv run python scripts/smart_build_index.py --clear

# Index one specific file
uv run python scripts/smart_build_index.py --file data/raw/NVDA_10K_2024-02-21.txt

# Force re-index a specific file
uv run python scripts/smart_build_index.py --force --file AAPL_10K_2024-11-01.txt
```

### Arguments

| Argument | Description |
|----------|-------------|
| `--force` | Re-index all files, even already-indexed ones |
| `--clear` | Delete all documents from index before building |
| `--file` | Index a specific file (path or filename in `data/raw/`) |

---

## 4. Test RAG Queries

**Script:** `scripts/test_rag.py`

Runs predefined queries against the indexed data to verify the RAG pipeline works.

### Example

```bash
uv run python scripts/test_rag.py
```

### Sample Output

```
======================================================================
TESTING RAG SYSTEM
======================================================================

Query: What was Apple's revenue in 2023?
----------------------------------------------------------------------
Response: Apple reported total net revenue of approximately $383.3 billion
for fiscal year 2023...

Query: What is Apple's gross margin?
----------------------------------------------------------------------
Response: Apple's gross margin for fiscal year 2023 was approximately
44.1%...

Query: How much cash does Apple have?
----------------------------------------------------------------------
Response: As of September 30, 2023, Apple had approximately $29.97
billion in cash and cash equivalents...

======================================================================
RAG TEST COMPLETE
======================================================================
```

---

## 5. List Indexed Documents

**Script:** `scripts/list_indexed_10k.py`

Shows all 10-K filings currently in the vector index.

### Example

```bash
uv run python scripts/list_indexed_10k.py
```

### Sample Output

```
Indexed 10-K filings:
============================================================
File: AAPL_10K_2024-11-01.txt
Ticker: AAPL
Filing Date: 2024-11-01
Doc Type: 10-K
------------------------------------------------------------
File: MSFT_10K_2024-07-30.txt
Ticker: MSFT
Filing Date: 2024-07-30
Doc Type: 10-K
------------------------------------------------------------
```

---

## 6. Python API Usage

### Query the RAG System Programmatically

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path("backend")))

from config.settings import get_settings
from config.logging_config import setup_logging
from src.utils.llama_setup import configure_llama_index
from src.rag.supabase_rag import SupabaseRAG

# Initialize
settings = get_settings()
setup_logging(settings.log_level)
configure_llama_index()

# Load existing index
rag = SupabaseRAG()
rag.setup_database()
rag.load_index()
rag.create_query_engine(top_k=5)

# Query
answer = rag.query("What are Tesla's main revenue sources?")
print(answer)
```

### Use Financial Tools

```python
from src.tools.financial_tools import (
    calculate_financial_ratio,
    compare_companies,
    get_stock_price,
    execute_tool,
)

# Calculate P/E ratio
result = calculate_financial_ratio("pe_ratio", "AAPL")
print(result)
# {'ratio_name': 'pe_ratio', 'ticker': 'AAPL', 'value': 28.5,
#  'components': {'period': 'FY2023'}, 'interpretation': 'Fairly valued'}

# Compare companies on revenue
result = compare_companies(["AAPL", "MSFT", "GOOGL"], "revenue")
print(result)
# {'metric': 'revenue', 'companies': ['AAPL', 'MSFT', 'GOOGL'],
#  'values': [394.33, 211.92, 307.39], 'winner': 'AAPL', ...}

# Get stock price
result = get_stock_price("NVDA")
print(result)
# {'ticker': 'NVDA', 'date': '2026-01-31', 'current_price': 132.65, ...}

# Historical price
result = get_stock_price("AAPL", "2024-06-15")
print(result)
# {'ticker': 'AAPL', 'date': '2024-06-15', 'open': 213.37, 'close': 214.29, ...}

# Execute tool by name (for agent integration)
result = execute_tool("calculate_financial_ratio", {
    "ratio_type": "roe",
    "ticker": "MSFT"
})
print(result)
```

### Use Structured Output Models

```python
from src.models.structured_outputs import (
    AgentOutput,
    FinancialMetrics,
    InvestmentRecommendation,
)

# Create an agent output
output = AgentOutput(
    agent_name="financial_health",
    ticker="AAPL",
    score=0.75,
    confidence=0.85,
    metrics={"revenue_growth": 2.1, "gross_margin": 44.1},
    strengths=["Strong cash position", "High margins"],
    weaknesses=["Revenue growth slowing"],
    summary="Apple maintains strong financial health with industry-leading margins."
)

print(output.model_dump_json(indent=2))
```

### Use Metadata Extractor Directly

```python
from src.rag.metadata_extractor import SECMetadataExtractor

extractor = SECMetadataExtractor()

# Extract from filename
meta = extractor.extract_from_filename("AAPL_10K_2024-11-01.txt")
print(meta)
# {'source_file': 'AAPL_10K_2024-11-01.txt', 'ticker': 'AAPL',
#  'document_type': '10-K', 'filing_date': '2024-11-01'}

# Extract from content
content = "Total revenue for fiscal year 2024 was $391.0 billion, with gross margin of 46.2%."
enhanced = extractor.extract_from_content(content, meta)
print(enhanced['metric_types'])    # ['revenue', 'margin']
print(enhanced['content_type'])    # 'financial_data'
print(enhanced['fiscal_period'])   # 'fiscal year 2024'
```

---

## 7. Common Workflows

### Add a New Company

```bash
# 1. Download filings
uv run python scripts/download_sec_filings.py --ticker JPM --type 10-K --amount 3

# 2. Index new files (smart - skips already indexed)
uv run python scripts/smart_build_index.py

# 3. Query
uv run python scripts/test_rag.py
```

### Rebuild Entire Index

```bash
# Clear and rebuild everything
uv run python scripts/smart_build_index.py --clear
```

### Bulk Download for Multiple Companies

```bash
# Annual reports
uv run python scripts/download_sec_filings.py --tickers AAPL MSFT GOOGL AMZN NVDA TSLA META JPM V JNJ --type 10-K --amount 2

# Then index
uv run python scripts/smart_build_index.py
```

---

## 8. Database Queries (SQL)

Run these in the Supabase SQL Editor.

```sql
-- Count chunks per company
SELECT
    metadata->>'ticker' as ticker,
    COUNT(*) as total_chunks
FROM data_airas_documents
GROUP BY metadata->>'ticker'
ORDER BY total_chunks DESC;

-- Find chunks about revenue for a specific company
SELECT
    id,
    LEFT(text, 200) as preview,
    metadata->>'section' as section,
    metadata->>'fiscal_period' as period
FROM data_airas_documents
WHERE metadata->>'ticker' = 'AAPL'
  AND metadata->'metric_types' ? 'revenue'
LIMIT 10;

-- Check section distribution
SELECT
    metadata->>'ticker' as ticker,
    metadata->>'section' as section,
    COUNT(*) as chunks
FROM data_airas_documents
WHERE metadata->>'section' IS NOT NULL
GROUP BY metadata->>'ticker', metadata->>'section'
ORDER BY ticker, chunks DESC;

-- Use the helper function
SELECT * FROM search_documents('AAPL', 'income_statement', 5);

-- Company overview
SELECT * FROM company_overview;
```
