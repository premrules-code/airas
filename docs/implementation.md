# AIRAS V3 Implementation

## 1. Technology Stack & Selection Rationale

### Claude (Anthropic) for Agent LLM

**Selected:** Claude 3.5 Sonnet via the Anthropic SDK
**Alternatives considered:** GPT-4 (OpenAI), Gemini (Google)

Claude was selected for the agent reasoning loop because of its native tool-use API. Each agent call uses `tools=` in the `messages.create()` request, and Claude returns `tool_use` content blocks that the `BaseAgent` loop dispatches directly. This avoids prompt-engineering workarounds for function calling. Claude's structured JSON output is reliable enough that a simple `json.loads()` parse (with markdown fence stripping) handles the response without a retry-based JSON repair loop.

The agent loop is in `src/agents/base_agent.py`:

```python
response = client.messages.create(
    model=settings.claude_model,
    max_tokens=2048,
    temperature=self.TEMPERATURE,
    system=self.SYSTEM_PROMPT,
    messages=messages,
    tools=tools,  # Claude-native tool definitions
)

if response.stop_reason == "tool_use":
    # Dispatch tool calls, append results, loop
    ...
else:
    # End turn -- parse JSON output
    text = next((b.text for b in response.content if hasattr(b, "text")), "")
    output = self._parse_output(text, ticker)
```

### LangGraph for Orchestration

**Selected:** LangGraph `StateGraph` with `Send()` API
**Alternatives considered:** raw `asyncio.gather()`, `ThreadPoolExecutor`, Prefect

LangGraph was chosen because it provides a graph-based execution model with built-in state management. The `Send()` API enables parallel fan-out to N agents from a single conditional edge, and `Annotated[list, operator.add]` reducers automatically merge outputs from parallel nodes back into the shared state. This is cleaner than manual thread pool coordination and provides a visual graph structure that maps directly to the pipeline stages.

A raw asyncio approach would require manually managing concurrent tasks, collecting results, and handling partial failures. LangGraph handles all of this through its state reducer pattern.

### LlamaIndex for RAG

**Selected:** LlamaIndex `IngestionPipeline` + `VectorStoreIndex` + `PGVectorStore`
**Alternatives considered:** LangChain, custom embedding pipeline

LlamaIndex was selected for its `IngestionPipeline` abstraction, which chains transformations (chunking, metadata extraction, embedding) into a single pipeline that writes directly to the vector store. The `BaseExtractor` interface allowed `SECMetadataTransform` to be plugged in as a pipeline step without custom orchestration code.

LangChain's document loading and splitting would require more manual wiring to achieve the same metadata enrichment and direct-to-pgvector pipeline.

### Supabase + pgvector for Vector Storage

**Selected:** Supabase PostgreSQL with pgvector extension
**Alternatives considered:** Pinecone, Qdrant, Weaviate, ChromaDB

Supabase was chosen because it provides PostgreSQL with pgvector in a managed service. Key advantages:

- **JSONB metadata**: Metadata filters use native PostgreSQL JSONB indexing (GIN index), which supports arbitrary key filtering without schema changes. Pinecone requires pre-declared metadata fields.
- **HNSW index**: pgvector supports HNSW indexing for approximate nearest neighbor search, competitive with dedicated vector databases for datasets under ~1M vectors.
- **Single database**: Both vector embeddings and application data live in the same PostgreSQL instance, simplifying infrastructure.
- **SQL access**: Complex queries (join vectors with metadata, aggregation) use standard SQL.

The vector store configuration in `src/rag/supabase_rag.py`:

```python
self.vector_store = PGVectorStore.from_params(
    database=url.database,
    host=url.host,
    password=url.password,
    port=url.port or 5432,
    user=url.username,
    table_name="airas_documents",
    embed_dim=1536,  # text-embedding-3-small
)
```

### FMP + Finnhub as Primary Data Providers

**Selected:** Financial Modeling Prep (FMP) as primary, Finnhub as secondary, yfinance as fallback
**Alternatives considered:** yfinance only, Alpha Vantage, Polygon.io

yfinance alone was initially used but proved unreliable due to aggressive rate limiting (HTTP 429) when 10 agents run concurrently. FMP provides a stable REST API with the Starter tier covering quotes, ratios, key metrics, historical prices, insider trades, analyst data, and news. Finnhub supplements with insider transactions and company news on a free tier (60 calls/min).

The multi-provider fallback pattern ensures data availability:

```
Tool request → FMP → Finnhub → yfinance
```

If `FMP_API_KEY` is not set, `fmp_client` functions return `None` and the tool falls through to the next provider.

### ta Library for Technical Indicators

**Selected:** `ta` (Technical Analysis Library in Python)
**Alternatives considered:** `pandas-ta`, `talib`

`ta` was chosen over `pandas-ta` because `pandas-ta` has installation issues on Python 3.11+ with certain NumPy versions. The `ta` library provides SMA, RSI, MACD, and Bollinger Bands with a straightforward API that takes a pandas Series as input. `talib` (TA-Lib) requires a C library installation that complicates deployment.

### Langfuse for Tracing

**Selected:** Langfuse
**Alternatives considered:** LangSmith, Phoenix (Arize), custom logging

Langfuse was selected because it provides free self-hosted or cloud tracing with a Python SDK that supports nested spans and LLM-specific generation logging. The `TracingManager` creates a trace hierarchy that maps directly to the pipeline stages: trace -> context gathering span -> agent spans -> LLM generation + tool spans. Langfuse's score API is used to log agent scores and recommendation scores for analysis across runs.

LangSmith was considered but ties more tightly to the LangChain ecosystem.

### Pydantic for Structured Outputs

All agent outputs and the final recommendation use Pydantic `BaseModel` subclasses with constrained fields (`ge=-1, le=1` for scores). This provides:

- Validation at parse time (invalid scores, missing fields raise errors)
- Type hints for IDE support
- `.model_dump()` for JSON serialization
- `Literal` types for recommendation values (`"STRONG BUY"` through `"STRONG SELL"`)

---

## 2. RAG Pipeline

### Data Flow

```
SEC filings (.txt)
      |
      v
LlamaIndex IngestionPipeline
  |-- SentenceSplitter (512-token chunks, 50-token overlap)
  |-- SECMetadataTransform (metadata enrichment)
  |-- OpenAI text-embedding-3-small (1536 dimensions)
      |
      v
Supabase pgvector (data_airas_documents table)
      |
      v
VectorStoreIndex
      |
      v
RAG queries (3 retrieval levels)
```

### Ingestion Pipeline

The pipeline is configured in `SupabaseRAG.setup_database()` (`src/rag/supabase_rag.py`):

```python
self.pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(
            chunk_size=self.settings.chunk_size,    # 512
            chunk_overlap=self.settings.chunk_overlap,  # 50
        ),
        SECMetadataTransform(),
        embed_model,  # text-embedding-3-small
    ],
    vector_store=self.vector_store,
)
```

Documents are loaded and indexed via:

```python
nodes = self.pipeline.run(documents=documents, show_progress=True)
self.index = VectorStoreIndex.from_vector_store(self.vector_store)
```

### Metadata Extraction

`SECMetadataExtractor` (`src/rag/metadata_extractor.py`) enriches each chunk with structured metadata from two sources:

**1. Filename parsing** (pattern: `TICKER_TYPE_DATE.txt`):

```python
def extract_from_filename(self, filename: str) -> Dict:
    parts = Path(filename).stem.split('_')
    metadata = {
        'source_file': filename,
        'ticker': parts[0].upper() if len(parts) >= 1 else None,
        'document_type': None,
        'filing_date': parts[2] if len(parts) >= 3 else None,
    }
    if len(parts) >= 2:
        doc_type = parts[1].upper().replace('-', '')
        metadata['document_type'] = '10-K' if 'K' in doc_type else '10-Q' if 'Q' in doc_type else doc_type
    return metadata
```

**2. Content analysis** (section detection, metric detection, fiscal period extraction):

```python
def extract_from_content(self, content: str, base_metadata: Dict) -> Dict:
    content_lower = content.lower()
    section = self._detect_section(content_lower)      # income_statement, balance_sheet, etc.
    metrics = self._detect_metrics(content_lower)       # [revenue, income, debt, ...]
    fiscal_period = self._detect_fiscal_period(content)  # "fiscal year 2023", "Q3 2024"
    has_numbers = bool(re.search(r'\$?\d{1,3}(,\d{3})*(\.\d+)?\s*(million|billion)?', content))
    content_type = 'financial_data' if (has_numbers and metrics) else 'narrative'
    return {**base_metadata, 'section': section, 'metric_types': metrics,
            'fiscal_period': fiscal_period, 'content_type': content_type, ...}
```

`SECMetadataTransform` wraps this as a LlamaIndex `BaseExtractor` for pipeline integration. It groups nodes by source file to compute correct `chunk_index` and `total_chunks` per document.

### Retrieval Level 1: Basic

Pure vector similarity search. No metadata filtering, no query rewriting.

```python
class BasicRetriever:
    def __init__(self, index):
        self.index = index

    def retrieve(self, query: str, top_k: int = 5, **kwargs):
        retriever = VectorIndexRetriever(index=self.index, similarity_top_k=top_k)
        return retriever.retrieve(query)
```

Used when `--rag-level basic` is specified. Returns the top-k chunks by cosine similarity to the query embedding.

### Retrieval Level 2: Intermediate

Adds metadata filtering and query expansion. This is the default level.

**Query expansion** rewrites queries using a synonym dictionary:

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
    # ... more expansions
}
```

**Metadata filter fallback chain** tries progressively less specific filters:

```python
def _filter_fallback_chain(self, ticker, sections):
    chain = []
    # Most specific: ticker + sections
    if ticker and sections:
        chain.append(MetadataFilters(filters=[
            MetadataFilter(key="ticker", value=ticker.upper()),
            MetadataFilter(key="section", value=sections, operator=FilterOperator.IN),
        ]))
    # Ticker only
    if ticker:
        chain.append(MetadataFilters(filters=[
            MetadataFilter(key="ticker", value=ticker.upper()),
        ]))
    return chain
```

The retriever tries each filter set in order. If the most specific filter returns no results, it falls back to ticker-only, then to no filters.

### Retrieval Level 3: Advanced

Adds HyDE (Hypothetical Document Embedding), multi-query generation, and Claude-based reranking on top of Intermediate retrieval.

**Step 1 -- Multi-query generation** creates 3 query variations:

```python
def _multi_query(self, query: str) -> list[str]:
    response = self._claude.messages.create(
        model=self._model,
        max_tokens=200,
        temperature=0.3,
        messages=[{"role": "user", "content": (
            "Generate 3 different ways to search for the answer to this financial question. "
            "Each should emphasize different keywords.\n"
            f"Question: {query}\n"
            "Return as a JSON array of 3 strings."
        )}],
    )
    return json.loads(response.content[0].text.strip())
```

**Step 2 -- HyDE transform** generates a hypothetical answer to improve embedding similarity:

```python
def _hyde_transform(self, query: str) -> str:
    response = self._claude.messages.create(
        model=self._model,
        max_tokens=200,
        temperature=0.0,
        messages=[{"role": "user", "content": (
            "Write a brief, factual paragraph that would answer this financial question. "
            "Use realistic but hypothetical numbers.\n"
            f"Question: {query}\nAnswer:"
        )}],
    )
    return response.content[0].text
```

**Step 3 -- Retrieve** runs all queries (original + 3 variations + HyDE) through the Intermediate retriever.

**Step 4 -- Deduplicate** by node ID.

**Step 5 -- Rerank** using Claude as a cross-encoder:

```python
def _rerank(self, query: str, nodes, top_n: int = 5):
    chunks_text = "\n---\n".join(f"[{i}] {node.text[:500]}" for i, node in enumerate(nodes))
    response = self._claude.messages.create(
        model=self._model,
        max_tokens=100,
        temperature=0.0,
        messages=[{"role": "user", "content": (
            f'Given this query: "{query}"\n'
            "Rank these chunks by relevance (most relevant first). "
            "Return ONLY the indices as a JSON array.\n"
            f"{chunks_text}"
        )}],
    )
    ranked_indices = json.loads(response.content[0].text.strip())
    return [nodes[i] for i in ranked_indices[:top_n] if i < len(nodes)]
```

---

## 3. The 10-Agent System

All agents inherit from `BaseAgent` and define 5 class attributes. No method overrides are needed. The base class provides the full Claude tool-use loop, retry logic, JSON parsing, and fallback output.

### BaseAgent Architecture

```python
class BaseAgent:
    AGENT_NAME: str = ""
    SYSTEM_PROMPT: str = ""
    RAG_QUERIES: list[str] = []
    TOOLS: list[str] = []
    SECTIONS: Optional[list[str]] = None
    WEIGHT: float = 0.0
    TEMPERATURE: float = 0.2
    MAX_TOOL_ITERATIONS: int = 5

    def analyze(self, ticker, rag_context, tracer) -> AgentOutput:
        # 1. Build user message with RAG context
        # 2. Loop: call Claude -> handle tool_use blocks -> continue
        # 3. Parse final JSON response into AgentOutput
        ...
```

The tool-use loop runs up to `MAX_TOOL_ITERATIONS` (5) rounds. Each round:
1. Calls Claude with the system prompt, conversation history, and available tools
2. If `stop_reason == "tool_use"`, executes each tool and appends results
3. If `stop_reason == "end_turn"`, parses the JSON response into `AgentOutput`

### Agent 1: Financial Analyst (20%)

```python
class FinancialAnalystAgent(BaseAgent):
    AGENT_NAME = "financial_analyst"
    WEIGHT = 0.20
    TEMPERATURE = 0.2
    SECTIONS = ["balance_sheet", "income_statement", "cash_flow"]
    SYSTEM_PROMPT = FINANCIAL_ANALYST_PROMPT
    RAG_QUERIES = [
        "{ticker} revenue gross profit operating income net income margins",
        "{ticker} balance sheet total assets liabilities shareholders equity",
        "{ticker} cash flow from operations free cash flow capital expenditures",
        "{ticker} debt structure interest expense debt maturity",
    ]
    TOOLS = ["calculate_financial_ratio", "compare_companies", "get_stock_price"]
```

**Prompt persona:** Senior financial analyst at a top-tier investment bank with 15+ years of experience evaluating corporate fundamentals.

**Scoring guide:**
- `> +0.5`: Strong financials -- healthy balance sheet, growing revenue, solid margins
- `+0.2 to +0.5`: Adequate financials with some concerns
- `-0.2 to +0.2`: Mixed signals
- `< -0.2`: Weak financials -- declining revenue, thin margins, or excessive leverage

**Sample output:**

```json
{
  "agent_name": "financial_analyst",
  "ticker": "AAPL",
  "score": 0.55,
  "confidence": 0.85,
  "metrics": {"pe_ratio": 38.2, "roe": 157.4, "profit_margin": 25.3, "current_ratio": 0.99},
  "strengths": ["Industry-leading margins (25%)", "Exceptional ROE (157%)", "Strong cash generation"],
  "weaknesses": ["High debt-to-equity (4.67)", "Negative working capital"],
  "summary": "Apple shows strong profitability and cash generation despite elevated leverage.",
  "sources": ["10-K FY2023", "FMP live data"]
}
```

### Agent 2: News Sentiment (12%)

```python
class NewsSentimentAgent(BaseAgent):
    AGENT_NAME = "news_sentiment"
    WEIGHT = 0.12
    TEMPERATURE = 0.4
    SECTIONS = None  # All sections -- needs MD&A narrative
    SYSTEM_PROMPT = NEWS_SENTIMENT_PROMPT
    RAG_QUERIES = [
        "{ticker} management discussion and analysis business outlook",
        "{ticker} forward looking statements guidance expectations",
        "{ticker} risk factors significant risks uncertainties",
        "{ticker} results of operations compared to prior year performance",
    ]
    TOOLS = ["get_stock_price"]
```

**Prompt persona:** Senior sentiment analyst specializing in SEC filing language and management communication patterns.

**Data sources:** RAG (SEC filing MD&A sections, risk factors, forward-looking statements)

**Sample output:**

```json
{
  "agent_name": "news_sentiment",
  "ticker": "AAPL",
  "score": 0.30,
  "confidence": 0.65,
  "metrics": {"tone": "moderately_optimistic", "guidance_specificity": "high", "risk_severity": "moderate"},
  "strengths": ["Confident revenue guidance", "Positive services growth narrative"],
  "weaknesses": ["Expanded China risk disclosure", "Cautious macro commentary"],
  "summary": "Apple management shows moderate optimism with specific guidance but expanded geopolitical risk language.",
  "sources": ["10-K MD&A section"]
}
```

### Agent 3: Technical Analyst (15%)

```python
class TechnicalAnalystAgent(BaseAgent):
    AGENT_NAME = "technical_analyst"
    WEIGHT = 0.15
    TEMPERATURE = 0.2
    SECTIONS = None
    SYSTEM_PROMPT = TECHNICAL_ANALYST_PROMPT
    RAG_QUERIES = []  # Pure tools, no RAG
    TOOLS = ["get_technical_indicators", "get_stock_price"]
```

**Prompt persona:** Senior technical analyst and quantitative trader with expertise in chart patterns, momentum indicators, and price action analysis.

**Technical rules encoded in prompt:**
- Price > SMA 50 > SMA 200 = bullish "golden cross" setup
- RSI > 70 = overbought (bearish), RSI < 30 = oversold (bullish)
- MACD line > signal line = bullish momentum
- Price near upper Bollinger Band = potentially overextended

**Sample output:**

```json
{
  "agent_name": "technical_analyst",
  "ticker": "AAPL",
  "score": 0.20,
  "confidence": 0.70,
  "metrics": {"rsi_14": 62, "sma_50": 230.5, "sma_200": 195.8, "macd_signal": "bullish", "trend": "bullish"},
  "strengths": ["Price above all major SMAs", "MACD bullish crossover", "Volume trending up"],
  "weaknesses": ["RSI approaching overbought territory", "Near upper Bollinger Band"],
  "summary": "AAPL shows a bullish technical setup but RSI suggests limited near-term upside.",
  "sources": ["FMP price data"]
}
```

### Agent 4: Risk Assessment (10%)

```python
class RiskAssessmentAgent(BaseAgent):
    AGENT_NAME = "risk_assessment"
    WEIGHT = 0.10
    TEMPERATURE = 0.3
    SECTIONS = None
    SYSTEM_PROMPT = RISK_ASSESSMENT_PROMPT
    RAG_QUERIES = []  # Uses tools only
    TOOLS = ["calculate_financial_ratio", "get_stock_price", "get_technical_indicators"]
```

**Prompt persona:** Senior risk analyst specializing in investment risk quantification.

**Scoring is inverted:** `+1 = very low risk (bullish)`, `-1 = very high risk (bearish)`. This means a high score from this agent contributes positively to the overall bullish signal.

**Sample output:**

```json
{
  "agent_name": "risk_assessment",
  "ticker": "AAPL",
  "score": 0.40,
  "confidence": 0.75,
  "metrics": {"beta": 1.24, "debt_to_equity": 4.67, "volatility": "moderate", "52wk_range_pct": 0.85},
  "strengths": ["Moderate beta (1.24)", "Strong cash reserves", "Blue-chip stability"],
  "weaknesses": ["High debt-to-equity ratio", "Trading near 52-week high"],
  "summary": "Apple presents moderate investment risk with manageable volatility but elevated leverage.",
  "sources": ["FMP live data"]
}
```

### Agent 5: Competitive Analysis (10%)

```python
class CompetitiveAnalysisAgent(BaseAgent):
    AGENT_NAME = "competitive_analysis"
    WEIGHT = 0.10
    TEMPERATURE = 0.4
    SECTIONS = None
    SYSTEM_PROMPT = COMPETITIVE_ANALYSIS_PROMPT
    RAG_QUERIES = [
        "{ticker} competitive landscape market position market share",
        "{ticker} competitive advantages intellectual property patents",
        "{ticker} barriers to entry brand recognition customer loyalty",
        "{ticker} industry competition risk factors competitive threats",
    ]
    TOOLS = ["compare_companies", "get_stock_price"]
```

**Prompt persona:** Strategy consultant specializing in competitive analysis and economic moats, with expertise in Porter's Five Forces and sustainable competitive advantages.

**Evaluates:** Moat durability, market share, barriers to entry, brand strength, IP, network effects, switching costs.

**Sample output:**

```json
{
  "agent_name": "competitive_analysis",
  "ticker": "AAPL",
  "score": 0.70,
  "confidence": 0.80,
  "metrics": {"moat_type": "wide", "market_position": "leader", "moat_sources": "ecosystem,brand,switching_costs"},
  "strengths": ["1.2B+ active device ecosystem", "Premium brand positioning", "High switching costs"],
  "weaknesses": ["Smartphone market maturing", "Regulatory pressure on App Store"],
  "summary": "Apple has a wide economic moat driven by ecosystem lock-in, brand, and switching costs.",
  "sources": ["10-K Business section", "10-K Risk Factors"]
}
```

### Agent 6: Insider Activity (8%)

```python
class InsiderActivityAgent(BaseAgent):
    AGENT_NAME = "insider_activity"
    WEIGHT = 0.08
    TEMPERATURE = 0.2
    SECTIONS = None
    SYSTEM_PROMPT = INSIDER_ACTIVITY_PROMPT
    RAG_QUERIES = []  # Uses tools only
    TOOLS = ["get_insider_trades", "get_stock_price"]
```

**Prompt persona:** Specialist in insider trading analysis who interprets executive buy/sell patterns as investment signals.

**Key principles encoded in prompt:**
- Insiders buy for one reason (they expect price increase) but sell for many (tax, diversification)
- Weight buys more heavily than sells
- Cluster buying (multiple insiders at similar time) is the strongest signal

### Agent 7: Options Analysis (5%)

```python
class OptionsAnalysisAgent(BaseAgent):
    AGENT_NAME = "options_analysis"
    WEIGHT = 0.05
    TEMPERATURE = 0.2
    SECTIONS = None
    SYSTEM_PROMPT = OPTIONS_ANALYSIS_PROMPT
    RAG_QUERIES = []  # Uses tools only
    TOOLS = ["get_options_data", "get_stock_price"]
```

**Prompt persona:** Options market analyst who reads market expectations from derivatives data and implied volatility patterns.

**Key indicators:** Put/call ratio (>1.0 = bearish, <0.7 = bullish), implied volatility levels, unusual volume.

**Note:** Options data is currently returning a neutral placeholder due to provider reliability issues. The agent scores with low confidence when options data is unavailable.

### Agent 8: Social Sentiment (3%)

```python
class SocialSentimentAgent(BaseAgent):
    AGENT_NAME = "social_sentiment"
    WEIGHT = 0.03
    TEMPERATURE = 0.3
    SECTIONS = None
    SYSTEM_PROMPT = SOCIAL_SENTIMENT_PROMPT
    RAG_QUERIES = []  # Uses tools only
    TOOLS = ["get_social_sentiment", "get_stock_price"]
```

**Prompt persona:** Social media sentiment analyst measuring retail investor mood across platforms.

**Data sources:** StockTwits (bull/bear labels), Reddit (VADER sentiment on r/wallstreetbets, r/stocks, r/investing), news headlines (FMP/Finnhub/yfinance).

**Weight justification:** Lowest weight (3%) because social sentiment is the noisiest signal. News sentiment is weighted highest within the tool, StockTwits medium, Reddit lowest.

### Agent 9: Earnings Analysis (7%)

```python
class EarningsAnalysisAgent(BaseAgent):
    AGENT_NAME = "earnings_analysis"
    WEIGHT = 0.07
    TEMPERATURE = 0.2
    SECTIONS = ["income_statement"]
    SYSTEM_PROMPT = EARNINGS_ANALYSIS_PROMPT
    RAG_QUERIES = [
        "{ticker} earnings per share diluted EPS trend",
        "{ticker} revenue growth year over year quarterly",
        "{ticker} segment revenue product services breakdown",
        "{ticker} operating expenses cost structure efficiency",
    ]
    TOOLS = ["calculate_financial_ratio", "compare_companies"]
```

**Prompt persona:** Earnings quality analyst who evaluates EPS trends, revenue composition, and earnings sustainability from SEC filings.

**Sample output:**

```json
{
  "agent_name": "earnings_analysis",
  "ticker": "AAPL",
  "score": 0.45,
  "confidence": 0.80,
  "metrics": {"eps_growth_yoy": -2.8, "services_growth": 9.1, "margin_trend": "stable"},
  "strengths": ["Services segment growing at 9%", "Stable gross margins", "Strong EPS base"],
  "weaknesses": ["Slight EPS decline YoY", "Product revenue flat"],
  "summary": "Apple earnings show stable quality with services growth offsetting flat product revenue.",
  "sources": ["10-K Income Statement", "FMP live data"]
}
```

### Agent 10: Analyst Ratings (10%)

```python
class AnalystRatingsAgent(BaseAgent):
    AGENT_NAME = "analyst_ratings"
    WEIGHT = 0.10
    TEMPERATURE = 0.2
    SECTIONS = None
    SYSTEM_PROMPT = ANALYST_RATINGS_PROMPT
    RAG_QUERIES = []  # Uses tools only
    TOOLS = ["get_analyst_ratings", "get_stock_price"]
```

**Prompt persona:** Analyst consensus interpreter who evaluates Wall Street sentiment from analyst ratings, price targets, and recommendation trends.

**Key factors:** Consensus rating, upside to mean price target (>15% = bullish, <0% = bearish), target range width, recent upgrades vs downgrades.

**Sample output:**

```json
{
  "agent_name": "analyst_ratings",
  "ticker": "AAPL",
  "score": 0.35,
  "confidence": 0.80,
  "metrics": {"consensus": "buy", "target_upside_pct": 12.5, "num_analysts": 38, "target_mean": 250},
  "strengths": ["Buy consensus from 38 analysts", "12.5% upside to mean target"],
  "weaknesses": ["Wide target range ($180-$300) shows disagreement"],
  "summary": "Wall Street consensus is Buy with 12.5% upside, supported by 38 analysts.",
  "sources": ["FMP analyst data"]
}
```

### Agent Summary Table

| # | Agent | Class | Weight | Temp | RAG | Tools | Sections |
|---|-------|-------|--------|------|-----|-------|----------|
| 1 | Financial Analyst | `FinancialAnalystAgent` | 20% | 0.2 | 4 queries | `calculate_financial_ratio`, `compare_companies`, `get_stock_price` | balance_sheet, income_statement, cash_flow |
| 2 | News Sentiment | `NewsSentimentAgent` | 12% | 0.4 | 4 queries | `get_stock_price` | All |
| 3 | Technical Analyst | `TechnicalAnalystAgent` | 15% | 0.2 | None | `get_technical_indicators`, `get_stock_price` | -- |
| 4 | Risk Assessment | `RiskAssessmentAgent` | 10% | 0.3 | None | `calculate_financial_ratio`, `get_stock_price`, `get_technical_indicators` | -- |
| 5 | Competitive Analysis | `CompetitiveAnalysisAgent` | 10% | 0.4 | 4 queries | `compare_companies`, `get_stock_price` | All |
| 6 | Insider Activity | `InsiderActivityAgent` | 8% | 0.2 | None | `get_insider_trades`, `get_stock_price` | -- |
| 7 | Options Analysis | `OptionsAnalysisAgent` | 5% | 0.2 | None | `get_options_data`, `get_stock_price` | -- |
| 8 | Social Sentiment | `SocialSentimentAgent` | 3% | 0.3 | None | `get_social_sentiment`, `get_stock_price` | -- |
| 9 | Earnings Analysis | `EarningsAnalysisAgent` | 7% | 0.2 | 4 queries | `calculate_financial_ratio`, `compare_companies` | income_statement |
| 10 | Analyst Ratings | `AnalystRatingsAgent` | 10% | 0.2 | None | `get_analyst_ratings`, `get_stock_price` | -- |

Weights sum to 100%.

---

## 4. Tool Fallback Chains

### Overview

Each tool implements a multi-provider fallback chain. The `execute_tool()` function in `src/tools/financial_tools.py` dispatches by name:

```python
def execute_tool(tool_name: str, tool_input: Dict) -> Dict:
    tools = {
        "calculate_financial_ratio": calculate_financial_ratio,
        "compare_companies": compare_companies,
        "get_stock_price": get_stock_price,
        "get_technical_indicators": get_technical_indicators,
        "get_insider_trades": get_insider_trades,
        "get_options_data": get_options_data,
        "get_analyst_ratings": get_analyst_ratings,
        "get_social_sentiment": get_social_sentiment,
    }
    return tools[tool_name](**tool_input)
```

### Fallback Chain Diagram

```
get_stock_price          FMP quote ---------> yfinance info
                                               (+ history for date)

calculate_financial_ratio FMP ratios --------> yfinance info
                          FMP key-metrics
                          (for ROE)

compare_companies        FMP quote/ratios ---> yfinance info
                         (per ticker)

get_technical_indicators FMP historical ------> yfinance history
                         prices + ta lib        + manual calc

get_insider_trades       FMP insider ---------> Finnhub insider ----> yfinance insider
                         /latest                transactions          transactions

get_options_data         (currently disabled -- returns neutral placeholder)
                         Tradier chain -------> yfinance options
                         (code exists)

get_analyst_ratings      FMP price targets ---> yfinance info
                         + FMP grades

get_social_sentiment     StockTwits API (direct, no auth)
                         + Reddit/PRAW (if credentials set)
                         + News: FMP news ---> Finnhub news ---> yfinance news
```

### Technical Indicators: FMP + ta Library

The `_fmp_get_technical_indicators` function downloads FMP historical prices and computes indicators locally using the `ta` library. This avoids needing the FMP Premium tier for pre-computed indicators:

```python
def _fmp_get_technical_indicators(ticker: str) -> Optional[Dict]:
    hist = fmp_client.get_historical_prices(ticker)
    if not hist or len(hist) < 26:
        return None

    prices = list(reversed(hist))  # FMP returns newest first
    close = pd.Series([float(p.get("price", p.get("close", 0))) for p in prices])

    from ta.trend import SMAIndicator, MACD as TAmacd
    from ta.momentum import RSIIndicator
    from ta.volatility import BollingerBands

    sma_20_ind = SMAIndicator(close, window=20)
    sma_50_ind = SMAIndicator(close, window=50)
    rsi_ind = RSIIndicator(close, window=14)
    macd_ind = TAmacd(close)
    bb_ind = BollingerBands(close, window=20)

    return {
        "ticker": ticker,
        "current_price": round(float(close.iloc[-1]), 2),
        "sma_20": round(float(sma_20_ind.sma_indicator().iloc[-1]), 2),
        "sma_50": round(float(sma_50_ind.sma_indicator().iloc[-1]), 2),
        "rsi_14": round(float(rsi_ind.rsi().iloc[-1]), 2),
        "macd": {
            "macd": round(float(macd_ind.macd().iloc[-1]), 4),
            "signal": round(float(macd_ind.macd_signal().iloc[-1]), 4),
            "histogram": round(float(macd_ind.macd_diff().iloc[-1]), 4),
        },
        "source": "fmp+ta",
    }
```

If `ta` is not installed, it falls back to manual pandas calculations for RSI, MACD, and Bollinger Bands.

### Insider Trades: FMP -> Finnhub -> yfinance

The `get_insider_trades` function chains three providers:

```python
def get_insider_trades(ticker: str) -> Dict:
    # Try FMP first
    result = _fmp_get_insider_trades(ticker)
    if result:
        return result

    # Try Finnhub
    result = _finnhub_get_insider_trades(ticker)
    if result:
        return result

    # Fallback to yfinance
    return _yfinance_get_insider_trades(ticker)
```

The Finnhub implementation parses transaction codes:

```python
def _finnhub_get_insider_trades(ticker: str) -> Optional[Dict]:
    transactions = finnhub_client.get_insider_transactions(ticker)
    if not transactions:
        return None

    for t in transactions[:20]:
        code = (t.get("transactionCode") or "").upper()
        if code == "P":    # Purchase
            buys += 1
        elif code == "S":  # Sale
            sells += 1
    ...
```

### API Client Architecture

All three API clients (`fmp_client`, `finnhub_client`, `tradier_client`) share the same pattern:

1. **Guard clause**: Return `None` if API key is not configured
2. **Response caching**: 5-minute TTL dictionary cache
3. **Retry logic**: Exponential backoff on HTTP 429 (rate limit)
4. **Graceful degradation**: Return `None` on failure, allowing the next provider in the chain to try

FMP client example (`src/tools/fmp_client.py`):

```python
def _fmp_get(path: str, params: dict = None, max_retries: int = 2) -> Optional[dict | list]:
    settings = get_settings()
    if not settings.fmp_api_key:
        return None  # Graceful fallback

    # Check cache (5-min TTL)
    cache_key = f"{url}?{sorted_params}"
    if cache_key in _cache and now - cached_time < CACHE_TTL:
        return cached_data

    # Request with retry
    for attempt in range(max_retries + 1):
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code == 200:
            _cache[cache_key] = (now, resp.json())
            return resp.json()
        elif resp.status_code == 429:
            time.sleep(5 * (2 ** attempt))
    return None
```

### Tool Status Table

| Tool | FMP | Finnhub | Tradier | yfinance | Status |
|------|-----|---------|---------|----------|--------|
| `get_stock_price` | Primary | -- | -- | Fallback | Active |
| `calculate_financial_ratio` | Primary | -- | -- | Fallback | Active |
| `compare_companies` | Primary | -- | -- | Fallback | Active |
| `get_technical_indicators` | Primary (prices) + ta lib | -- | -- | Fallback | Active |
| `get_insider_trades` | Primary | Secondary | -- | Fallback | Active |
| `get_options_data` | -- | -- | Exists (disabled) | Exists (disabled) | Disabled |
| `get_analyst_ratings` | Primary | -- | -- | Fallback | Active |
| `get_social_sentiment` | News provider | News fallback | -- | News fallback | Active |

---

## 5. LangGraph Orchestration

### Pipeline Diagram

```
                    +--------------+
                    | Entry Point  |
                    +--------------+
                           |
                           v
                    +--------------+
                    | route_query  |  Classify query, select agents,
                    |              |  start Langfuse trace
                    +--------------+
                           |
                           v
                    +----------------+
                    | gather_context |  Run RAG queries for all
                    |                |  active agents upfront
                    +----------------+
                           |
              fan_out_to_agents (Send API)
                    /   |   |    \
                   v    v   v     v
              +------+------+------+------+
              |Agent1|Agent2| ...  |AgentN|  Parallel execution
              |      |      |      |      |  (with 5s stagger)
              +------+------+------+------+
                   \    |   |    /
                    v   v   v   v
                    +-----------+
                    | synthesis |  Weighted scoring,
                    |           |  thesis generation
                    +-----------+
                         |
                         v
                      +-----+
                      | END |
                      +-----+
```

### AnalysisState TypedDict

The shared state uses LangGraph's `Annotated` reducer pattern for parallel output merging:

```python
class AnalysisState(TypedDict):
    ticker: str
    query: Optional[str]                                       # None = full analysis
    active_agents: list[str]                                   # Selected by router
    mode: str                                                  # "full" | "focused"
    rag_level: str                                             # "basic" | "intermediate" | "advanced"
    rag_context: dict[str, str]                                # agent_name -> context string
    agent_outputs: Annotated[list[AgentOutput], operator.add]  # Auto-merges parallel outputs
    recommendation: Optional[InvestmentRecommendation]
    trace_id: Optional[str]                                    # Langfuse trace ID
    errors: Annotated[list[str], operator.add]                 # Collects errors from all nodes
```

The `Annotated[list, operator.add]` reducer is the key mechanism for parallel execution. When multiple agent nodes run concurrently, each returns `{"agent_outputs": [output]}`. LangGraph uses `operator.add` to concatenate all the lists into the shared `agent_outputs` field. The same reducer is used for `errors`.

### Fan-Out via Send API

The `fan_out_to_agents` function generates a `Send` message for each active agent:

```python
def fan_out_to_agents(state: AnalysisState) -> list[Send]:
    return [
        Send(name, state)
        for name in state["active_agents"]
        if name in AGENT_MAP
    ]
```

This is registered as a conditional edge from `gather_context`:

```python
graph.add_conditional_edges("gather_context", fan_out_to_agents)
```

Each agent node is created with a staggered start delay to avoid rate limit bursts:

```python
def make_agent_node(agent_cls, index: int = 0):
    def node_fn(state: AnalysisState) -> dict:
        if index > 0:
            time.sleep(index * 5)  # 5s stagger between agents
        agent = agent_cls()
        tracer = TracingManager.from_trace_id(state.get("trace_id"))
        rag_context = state.get("rag_context", {}).get(agent.AGENT_NAME, "")
        output = agent.analyze(state["ticker"], rag_context, tracer)
        return {"agent_outputs": [output]}
    return node_fn
```

### Graph Construction

```python
def build_analysis_graph():
    graph = StateGraph(AnalysisState)

    # Nodes
    graph.add_node("route_query", route_query_node)
    graph.add_node("gather_context", gather_context_node)
    for i, agent_cls in enumerate(ALL_AGENTS):
        graph.add_node(agent_cls.AGENT_NAME, make_agent_node(agent_cls, index=i))
    graph.add_node("synthesis", synthesis_node)

    # Edges
    graph.set_entry_point("route_query")
    graph.add_edge("route_query", "gather_context")
    graph.add_conditional_edges("gather_context", fan_out_to_agents)
    for agent_cls in ALL_AGENTS:
        graph.add_edge(agent_cls.AGENT_NAME, "synthesis")
    graph.add_edge("synthesis", END)

    return graph.compile()
```

### Router Query Classification

The router uses Claude to classify queries into categories, then maps categories to agent sets:

```python
CATEGORY_AGENTS = {
    "full_analysis": None,  # All agents
    "financial": ["financial_analyst", "earnings_analysis"],
    "technical": ["technical_analyst", "options_analysis"],
    "risk": ["risk_assessment", "competitive_analysis"],
    "sentiment": ["news_sentiment", "social_sentiment", "analyst_ratings"],
    "insider": ["insider_activity"],
    "competitive": ["competitive_analysis"],
}
```

The router prompt asks Claude to classify into categories:

```
Classify this financial analysis query into relevant categories.

Categories:
- "full_analysis": Comprehensive investment analysis, "should I buy/invest"
- "financial": Balance sheet, revenue, margins, debt, earnings, cash flow
- "technical": Stock price trends, charts, moving averages, RSI, momentum
- "risk": Risk factors, volatility, downside, debt risk
- "sentiment": News tone, analyst ratings, social media mood
- "insider": Insider trading, executive buying/selling
- "competitive": Competitive advantages, moat, market position
```

If the query implies a full recommendation ("should I buy AAPL?"), all 10 agents run. A focused query like "What's AAPL's technical setup?" routes only to `technical_analyst` and `options_analysis`.

---

## 6. Synthesis & Scoring

### Agent Weights

```python
AGENT_WEIGHTS = {
    "financial_analyst": 0.20,   # Heaviest -- fundamentals matter most
    "technical_analyst": 0.15,   # Price action is a strong signal
    "news_sentiment":    0.12,   # Filing tone provides context
    "risk_assessment":   0.10,   # Risk quantification
    "competitive_analysis": 0.10,# Moat durability
    "analyst_ratings":   0.10,   # Wall Street consensus
    "insider_activity":  0.08,   # Insider signals
    "earnings_analysis": 0.07,   # Earnings quality
    "options_analysis":  0.05,   # Derivatives sentiment
    "social_sentiment":  0.03,   # Noisiest signal
}
# Total: 1.00
```

### Scoring Algorithm

The synthesis function in `src/agents/synthesis.py` computes the overall score using confidence-weighted averaging:

```python
# Overall score: weight-adjusted, confidence-weighted
total_weighted = 0.0
total_weight = 0.0
for output in outputs:
    w = AGENT_WEIGHTS.get(output.agent_name, 0.05)
    total_weighted += output.score * output.confidence * w
    total_weight += output.confidence * w
overall_score = total_weighted / total_weight if total_weight > 0 else 0.0
overall_score = max(-1.0, min(1.0, overall_score))  # Clamp
```

This means an agent with high confidence has more influence than one with low confidence, even at the same weight. If the financial analyst scores +0.6 with 0.9 confidence and the social sentiment agent scores -0.3 with 0.2 confidence, the financial analyst's contribution dominates.

### Category Scores

Agents are grouped into four categories for the category breakdown:

```python
CATEGORIES = {
    "financial_score": ["financial_analyst", "earnings_analysis"],
    "technical_score": ["technical_analyst", "options_analysis"],
    "sentiment_score": ["news_sentiment", "social_sentiment", "analyst_ratings"],
    "risk_score":      ["risk_assessment", "competitive_analysis", "insider_activity"],
}
```

Each category score is computed as a confidence-weighted average of its member agents:

```python
for category, agent_names in CATEGORIES.items():
    relevant = [o for o in outputs if o.agent_name in agent_names]
    if relevant:
        weighted_sum = sum(o.score * o.confidence for o in relevant)
        confidence_sum = sum(o.confidence for o in relevant)
        category_scores[category] = weighted_sum / confidence_sum
```

### Score to Recommendation Mapping

```
Overall Score        Recommendation
>= +0.6             STRONG BUY
>= +0.2             BUY
>= -0.2             HOLD
>= -0.6             SELL
<  -0.6             STRONG SELL
```

### Thesis Generation

After computing scores, Claude generates a natural language thesis:

```python
def _generate_thesis(outputs, ticker, rec, settings):
    summaries = "\n".join(
        f"- {o.agent_name} (score={o.score}, confidence={o.confidence}): {o.summary}"
        for o in outputs
    )
    response = client.messages.create(
        model=settings.claude_model,
        max_tokens=500,
        temperature=0.3,
        messages=[{
            "role": "user",
            "content": (
                f"Based on these agent analyses for {ticker} (recommendation: {rec}):\n\n"
                f"{summaries}\n\n"
                "Generate a JSON with:\n"
                '- "thesis": 2-3 sentence investment thesis\n'
                '- "bullish_factors": 3 key bullish factors\n'
                '- "bearish_factors": 3 key bearish factors\n'
                '- "risks": 3 key risks'
            ),
        }],
    )
    return json.loads(response.content[0].text.strip())
```

### InvestmentRecommendation Model

The final output is validated by Pydantic:

```python
class InvestmentRecommendation(BaseModel):
    ticker: str
    company_name: Optional[str] = None
    recommendation: Literal["STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"]
    confidence: float = Field(ge=0, le=1)
    overall_score: float = Field(ge=-1, le=1)
    financial_score: float = Field(ge=-1, le=1)
    technical_score: float = Field(ge=-1, le=1)
    sentiment_score: float = Field(ge=-1, le=1)
    risk_score: float = Field(ge=-1, le=1)
    agent_scores: Dict[str, float]
    bullish_factors: List[str]
    bearish_factors: List[str]
    risks: List[str]
    thesis: str
    num_agents: int
```

---

## 7. CLI Usage

### Entry Point

```bash
python scripts/run_analysis.py --ticker AAPL [options]
```

### Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--ticker` | Stock ticker symbol (required) | -- |
| `--query` | Focused question (routes to relevant agents) | None (full analysis) |
| `--full` | Force all 10 agents even with `--query` | False |
| `--rag-level` | RAG retrieval level: `basic`, `intermediate`, `advanced` | `intermediate` |
| `--sequential` | Run agents sequentially (for debugging) | False |
| `--output` | Save results to JSON file | None |

### Full Analysis

```bash
python scripts/run_analysis.py --ticker AAPL
```

Output:

```
======================================================================
AIRAS V3 -- Investment Analysis for AAPL
======================================================================
RAG Level: intermediate

======================================================================
  RECOMMENDATION: BUY
======================================================================
  Overall Score:  +0.374
  Confidence:     72.1%
  Agents Run:     10
  Time:           187.3s

  Category Scores:
    Financial:    +0.512
    Technical:    +0.201
    Sentiment:    +0.328
    Risk:         +0.401

  Agent Scores:
    financial_analyst           +0.550
    news_sentiment              +0.300
    technical_analyst           +0.200
    risk_assessment             +0.400
    competitive_analysis        +0.700
    insider_activity            -0.100
    options_analysis            +0.000
    social_sentiment            +0.150
    earnings_analysis           +0.450
    analyst_ratings             +0.350

  Thesis:
    Apple presents a compelling investment case driven by industry-leading
    profitability and a wide competitive moat from its ecosystem. While
    elevated leverage and near-overbought technical conditions warrant
    monitoring, strong fundamentals and analyst consensus support a BUY.

  Bullish Factors:
    + Industry-leading profit margins (25%) and exceptional ROE
    + Wide economic moat from ecosystem lock-in and brand strength
    + Buy consensus from 38 analysts with 12.5% upside to targets

  Bearish Factors:
    - High debt-to-equity ratio (4.67) and negative working capital
    - RSI approaching overbought territory near 62
    - Smartphone market maturation limiting product revenue growth

  Risks:
    ! Regulatory pressure on App Store revenue model
    ! China geopolitical risk affecting supply chain
    ! Interest rate sensitivity on $100B+ debt load
======================================================================
```

### Focused Query

```bash
python scripts/run_analysis.py --ticker NVDA --query "What are the technical signals?"
```

This routes only to `technical_analyst` and `options_analysis` agents (the "technical" category).

### Advanced RAG

```bash
python scripts/run_analysis.py --ticker AAPL --rag-level advanced
```

Uses HyDE + multi-query + reranking for higher quality RAG context. Costs more in LLM calls due to the additional Claude requests for query generation, hypothetical answers, and reranking.

### JSON Export

```bash
python scripts/run_analysis.py --ticker AAPL --output results.json
```

Saves the `InvestmentRecommendation` model as JSON:

```json
{
  "ticker": "AAPL",
  "company_name": "AAPL",
  "recommendation": "BUY",
  "confidence": 0.721,
  "overall_score": 0.374,
  "financial_score": 0.512,
  "technical_score": 0.201,
  "sentiment_score": 0.328,
  "risk_score": 0.401,
  "agent_scores": {
    "financial_analyst": 0.55,
    "news_sentiment": 0.30,
    "technical_analyst": 0.20,
    "risk_assessment": 0.40,
    "competitive_analysis": 0.70,
    "insider_activity": -0.10,
    "options_analysis": 0.00,
    "social_sentiment": 0.15,
    "earnings_analysis": 0.45,
    "analyst_ratings": 0.35
  },
  "bullish_factors": ["..."],
  "bearish_factors": ["..."],
  "risks": ["..."],
  "thesis": "...",
  "analysis_date": "2025-01-15T14:32:01.123456",
  "analysis_time_seconds": 187.34,
  "num_agents": 10
}
```

### Data Pipeline Commands

```bash
# Download SEC filings
python scripts/download_sec_filings.py --ticker AAPL

# Build vector index
python scripts/smart_build_index.py

# Rebuild specific file
python scripts/smart_build_index.py --file data/sec_filings/AAPL_10K_2023-11-03.txt

# Wipe and rebuild entire index
python scripts/smart_build_index.py --clear --force

# Test RAG queries
python scripts/test_rag.py
```

---

## 8. Monitoring (Langfuse)

### Trace Hierarchy

```
Trace: {ticker}_analysis
│
├── Span: gather_context
│   ├── Generation: rag_query_financial_analyst_0
│   ├── Generation: rag_query_financial_analyst_1
│   ├── Generation: rag_query_financial_analyst_2
│   ├── Generation: rag_query_financial_analyst_3
│   ├── Generation: rag_query_news_sentiment_0
│   ├── ...
│   └── Generation: rag_query_earnings_analysis_3
│
├── Span: agent_financial_analyst
│   ├── Generation: llm_call  (Claude API, with usage tokens)
│   ├── Span: tool_calculate_financial_ratio
│   ├── Span: tool_compare_companies
│   ├── Generation: llm_call  (second round)
│   └── Score: financial_analyst_score = 0.55
│       Score: financial_analyst_confidence = 0.85
│
├── Span: agent_news_sentiment
│   ├── Generation: llm_call
│   └── Score: news_sentiment_score = 0.30
│
├── ... (8 more agent spans)
│
├── Span: agent_synthesis
│
└── Scores:
    ├── overall_score = 0.374
    ├── financial_score = 0.512
    ├── technical_score = 0.201
    ├── sentiment_score = 0.328
    └── risk_score = 0.401
```

### TracingManager Methods

The `TracingManager` class (`src/agents/tracing.py`) provides the following methods:

| Method | Creates | Logged Data |
|--------|---------|-------------|
| `start_trace(ticker)` | Top-level trace | Trace ID, ticker |
| `span_context_gathering()` | Span | RAG phase start/end |
| `log_rag_query(span, agent, query, response, idx)` | Generation | Query text, response (truncated to 2000 chars) |
| `span_agent(agent_name)` | Span | Agent name in metadata |
| `log_llm_call(span, messages, response, model, usage)` | Generation | Input/output (truncated to 5000 chars), model name, token usage |
| `log_tool_call(span, tool_name, input, output)` | Span | Tool name, input/output (truncated to 2000 chars) |
| `log_agent_score(span, agent_output)` | 2 Scores | `{agent}_score`, `{agent}_confidence` |
| `log_recommendation(recommendation)` | 5 Scores | overall, financial, technical, sentiment, risk |
| `end_trace()` | -- | Flushes Langfuse buffer |

### NullSpan Pattern

When Langfuse is not configured (no `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` in `.env`), the `TracingManager` returns `_NullSpan` objects that accept all method calls as no-ops:

```python
class _NullSpan:
    def start_span(self, **kwargs):   return _NullSpan()
    def start_generation(self, **kwargs): return _NullSpan()
    def update(self, **kwargs):       return self
    def end(self):                    pass
    def score(self, **kwargs):        pass
```

This means the entire analysis pipeline runs identically with or without Langfuse configured. No conditional checks are needed in the agent code.

### TracingManager Reconstruction

The `TracingManager` is created in `route_query_node`, and its trace ID is stored in the `AnalysisState`. Downstream nodes reconstruct the manager from the trace ID:

```python
# In route_query_node:
tracer = TracingManager()
trace_id = tracer.start_trace(state["ticker"])
return {"trace_id": trace_id}

# In any downstream node:
tracer = TracingManager.from_trace_id(state.get("trace_id"))
```

This pattern works because LangGraph may serialize state between nodes. The trace ID (a string) is serializable, while the full `TracingManager` (with its Langfuse client reference) is not.

### Langfuse Setup

Langfuse is initialized in `src/utils/langfuse_setup.py` and requires these environment variables:

```
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com  # or self-hosted URL
```

If not configured, all tracing is silently disabled via the `_NullSpan` pattern.
