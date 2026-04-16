"""Query correction strategies for Corrective RAG.

Implements strategies to improve retrieval when initial results are poor:
1. QueryTransformer: Rewrite queries with better terminology
2. QueryDecomposer: Break complex queries into sub-questions
3. WebSearchFallback: Search external sources (optional)
"""

import json
import logging
from dataclasses import dataclass
from typing import List, Optional

import anthropic

from config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class TransformResult:
    """Result of query transformation."""

    original_query: str
    transformed_query: str
    strategy: str  # e.g., "keyword_expansion", "entity_focus", "temporal_grounding"
    explanation: str


class QueryTransformer:
    """Transform failed queries into better search terms.

    Strategies:
    1. Keyword expansion: Add financial synonyms and SEC-specific terms
    2. Entity focus: Extract and emphasize ticker, metrics, time periods
    3. Temporal grounding: Convert vague time references to specific periods
    4. Section targeting: Add SEC filing section keywords (Item 1A, Item 7, etc.)

    Usage:
        transformer = QueryTransformer()
        result = transformer.transform(
            query="What is Apple's financial health?",
            failed_docs=["doc1 about products...", "doc2 about marketing..."]
        )
        print(result.transformed_query)
        # "Apple AAPL financial health debt-to-equity current ratio liquidity 10-K Item 7 FY2023"
    """

    # SEC filing section keywords
    SEC_SECTIONS = {
        "risk": "Item 1A Risk Factors",
        "business": "Item 1 Business Description",
        "financial": "Item 7 MD&A Management Discussion Analysis",
        "legal": "Item 3 Legal Proceedings",
        "properties": "Item 2 Properties",
        "market": "Item 7A Market Risk",
        "controls": "Item 9A Controls Procedures",
        "executive": "Item 11 Executive Compensation",
    }

    # Financial term expansions (supplements IntermediateRetriever.EXPANSIONS)
    FINANCIAL_EXPANSIONS = {
        "revenue": ["net sales", "total revenue", "sales revenue", "top line"],
        "profit": ["net income", "earnings", "bottom line", "net profit"],
        "debt": ["total debt", "long-term debt", "borrowings", "obligations"],
        "cash": ["cash equivalents", "liquidity", "cash position", "cash flow"],
        "margin": ["gross margin", "operating margin", "profit margin", "EBITDA margin"],
        "growth": ["year-over-year", "YoY", "growth rate", "increase", "trend"],
        "valuation": ["market cap", "enterprise value", "P/E ratio", "price earnings"],
        "dividend": ["dividend yield", "payout ratio", "distributions", "shareholder return"],
        "expense": ["operating expenses", "SG&A", "R&D", "cost of sales"],
        "health": ["financial condition", "solvency", "liquidity ratios", "working capital"],
    }

    # Model for smart transformation - uses settings.claude_model
    DEFAULT_MODEL = None

    TRANSFORM_PROMPT = """You are improving a search query that failed to retrieve relevant financial documents.

Original query: {query}

The retrieved documents were about:
{failed_summary}

The query needs to find information from SEC 10-K filings. Improve the query by:
1. Adding specific financial terms and synonyms
2. Including SEC filing section references if relevant (e.g., "Item 1A" for risks, "Item 7" for MD&A)
3. Adding the ticker symbol explicitly
4. Specifying time periods if the query is about recent data (e.g., "FY2023", "fiscal year 2023")
5. Using more specific financial terminology

Return ONLY a JSON object (no markdown):
{{"transformed_query": "the improved query string", "strategy": "brief name of main improvement", "explanation": "1 sentence why this should work better"}}"""

    def __init__(self, model: Optional[str] = None):
        """Initialize transformer.

        Args:
            model: Claude model for smart transformation (default: settings.claude_model)
        """
        settings = get_settings()
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self._model = model or settings.claude_model

    def transform(
        self,
        query: str,
        failed_docs: List[str],
        ticker: Optional[str] = None,
    ) -> TransformResult:
        """Transform a query that produced poor retrieval results.

        Args:
            query: Original query
            failed_docs: Documents that were retrieved but marked irrelevant
            ticker: Stock ticker (if known)

        Returns:
            TransformResult with improved query
        """
        # First try rule-based transformation (fast, no LLM)
        rule_based = self._rule_based_transform(query, ticker)

        # If rule-based made significant changes, use that
        if len(rule_based) > len(query) * 1.5:
            return TransformResult(
                original_query=query,
                transformed_query=rule_based,
                strategy="rule_based_expansion",
                explanation="Added financial keywords and SEC section terms",
            )

        # Otherwise, use LLM for smart transformation
        return self._llm_transform(query, failed_docs, ticker)

    def _rule_based_transform(
        self, query: str, ticker: Optional[str] = None
    ) -> str:
        """Apply rule-based query expansion."""
        parts = [query]

        # Add ticker if provided and not already in query
        if ticker and ticker.upper() not in query.upper():
            parts.append(ticker.upper())

        # Detect topic and add SEC section keywords
        query_lower = query.lower()
        for topic, section in self.SEC_SECTIONS.items():
            if topic in query_lower:
                parts.append(section)
                break

        # Add financial term expansions
        for term, expansions in self.FINANCIAL_EXPANSIONS.items():
            if term in query_lower:
                parts.extend(expansions[:2])  # Add top 2 expansions

        # Add temporal hints for recent data queries
        if any(word in query_lower for word in ["recent", "current", "latest", "now"]):
            parts.append("FY2023 FY2024 fiscal year")

        return " ".join(parts)

    def _llm_transform(
        self,
        query: str,
        failed_docs: List[str],
        ticker: Optional[str] = None,
    ) -> TransformResult:
        """Use LLM for intelligent query transformation."""
        # Summarize what failed docs were about
        failed_summary = "\n".join(
            f"- {doc[:200]}..." for doc in failed_docs[:3]
        )

        prompt = self.TRANSFORM_PROMPT.format(
            query=query, failed_summary=failed_summary
        )

        if ticker:
            prompt += f"\n\nNote: The query is about {ticker}."

        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=200,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )

            text = response.content[0].text.strip()

            # Parse JSON response
            if "```" in text:
                text = text.split("```")[1].split("```")[0]
                if text.startswith("json"):
                    text = text[4:]

            data = json.loads(text)

            return TransformResult(
                original_query=query,
                transformed_query=data.get("transformed_query", query),
                strategy=data.get("strategy", "llm_transformation"),
                explanation=data.get("explanation", ""),
            )

        except Exception as e:
            logger.warning(f"LLM transform failed: {e}")
            # Fall back to rule-based
            return TransformResult(
                original_query=query,
                transformed_query=self._rule_based_transform(query, ticker),
                strategy="rule_based_fallback",
                explanation=f"LLM failed: {e}",
            )


class QueryDecomposer:
    """Decompose complex queries into simpler sub-questions.

    Useful for multi-part questions that need information from
    different sections of the filing.

    Usage:
        decomposer = QueryDecomposer()
        sub_queries = decomposer.decompose(
            "Compare Apple's revenue growth to Microsoft and analyze their debt levels"
        )
        # ["Apple AAPL revenue growth YoY FY2023",
        #  "Microsoft MSFT revenue growth YoY FY2023",
        #  "Apple AAPL debt long-term debt total debt",
        #  "Microsoft MSFT debt long-term debt total debt"]
    """

    DEFAULT_MODEL = None

    DECOMPOSE_PROMPT = """Break down this complex financial query into simpler sub-questions that can each be answered from SEC filings.

Query: {query}

Rules:
1. Each sub-question should focus on ONE metric or fact
2. Include ticker symbols explicitly in each sub-question
3. Add relevant financial keywords
4. Keep each sub-question focused and searchable

Return ONLY a JSON array of strings (no markdown, no explanation):
["sub-question 1", "sub-question 2", ...]

Return 2-4 sub-questions."""

    def __init__(self, model: Optional[str] = None):
        settings = get_settings()
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self._model = model or settings.claude_model

    def decompose(self, query: str) -> List[str]:
        """Decompose a complex query into sub-questions.

        Args:
            query: Complex multi-part query

        Returns:
            List of simpler sub-queries
        """
        # Check if decomposition is needed (simple heuristic)
        if not self._needs_decomposition(query):
            return [query]

        prompt = self.DECOMPOSE_PROMPT.format(query=query)

        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=300,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )

            text = response.content[0].text.strip()

            if "```" in text:
                text = text.split("```")[1].split("```")[0]
                if text.startswith("json"):
                    text = text[4:]

            sub_queries = json.loads(text)

            if isinstance(sub_queries, list) and len(sub_queries) > 1:
                logger.info(f"Decomposed query into {len(sub_queries)} sub-queries")
                return sub_queries

            return [query]

        except Exception as e:
            logger.warning(f"Query decomposition failed: {e}")
            return [query]

    def _needs_decomposition(self, query: str) -> bool:
        """Heuristic to check if query needs decomposition."""
        # Look for comparison keywords
        comparison_words = ["compare", "versus", "vs", "and", "both", "difference"]
        has_comparison = any(word in query.lower() for word in comparison_words)

        # Look for multiple metrics
        metric_words = ["revenue", "profit", "debt", "cash", "margin", "growth", "risk"]
        metric_count = sum(1 for word in metric_words if word in query.lower())

        # Look for multiple tickers (simplified check)
        import re
        ticker_pattern = r'\b[A-Z]{2,5}\b'
        potential_tickers = re.findall(ticker_pattern, query)
        has_multiple_tickers = len(potential_tickers) > 1

        return has_comparison or metric_count > 2 or has_multiple_tickers


class WebSearchFallback:
    """Search external sources when internal RAG fails.

    This is optional and requires a web search API (e.g., Tavily).

    Usage:
        fallback = WebSearchFallback()
        results = fallback.search("Apple FY2023 revenue", ticker="AAPL")
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize web search fallback.

        Args:
            api_key: Tavily API key (or from env TAVILY_API_KEY)
        """
        self._api_key = api_key
        self._enabled = False

        # Check if Tavily is available
        try:
            import os
            self._api_key = api_key or os.environ.get("TAVILY_API_KEY")
            if self._api_key:
                from tavily import TavilyClient
                self._client = TavilyClient(api_key=self._api_key)
                self._enabled = True
                logger.info("Web search fallback enabled (Tavily)")
        except ImportError:
            logger.debug("Tavily not installed, web search disabled")
        except Exception as e:
            logger.debug(f"Web search not available: {e}")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def search(
        self,
        query: str,
        ticker: Optional[str] = None,
        max_results: int = 3,
    ) -> List[dict]:
        """Search the web for financial information.

        Args:
            query: Search query
            ticker: Stock ticker for filtering
            max_results: Maximum results to return

        Returns:
            List of dicts with 'title', 'url', 'content'
        """
        if not self._enabled:
            logger.debug("Web search not enabled")
            return []

        # Add ticker and financial context to query
        search_query = f"{ticker} {query}" if ticker else query
        search_query += " SEC filing financial"

        try:
            response = self._client.search(
                query=search_query,
                max_results=max_results,
                include_domains=[
                    "sec.gov",
                    "finance.yahoo.com",
                    "bloomberg.com",
                    "reuters.com",
                    "wsj.com",
                ],
            )

            results = []
            for item in response.get("results", []):
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "content": item.get("content", "")[:1000],
                })

            logger.info(f"Web search returned {len(results)} results")
            return results

        except Exception as e:
            logger.warning(f"Web search failed: {e}")
            return []
