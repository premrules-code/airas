"""Q&A route — RAG-powered question answering."""

import re
import logging

import anthropic
from fastapi import APIRouter

from src.api.schemas import QARequest, QAResponse
from src.rag.supabase_rag import SupabaseRAG
from src.rag.retrieval import BasicRetriever, IntermediateRetriever, AdvancedRetriever
from config.settings import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()

# Patterns that need data from multiple financial concepts
_RATIO_PATTERNS = [
    (r'debt.{0,10}equity', ['total debt long-term debt borrowings', 'shareholders equity stockholders equity total equity book value']),
    (r'price.{0,10}earnings|p/?e\s', ['earnings per share net income EPS', 'stock price share price market cap']),
    (r'price.{0,10}book|p/?b\s', ['book value shareholders equity total equity', 'stock price share price market cap']),
    (r'current\s+ratio', ['current assets cash receivables', 'current liabilities accounts payable short-term debt']),
    (r'quick\s+ratio', ['cash cash equivalents receivables liquid assets', 'current liabilities short-term obligations']),
    (r'return\s+on\s+(equity|assets|invested)', ['net income earnings profit', 'total assets shareholders equity invested capital']),
    (r'profit\s+margin|operating\s+margin|gross\s+margin', ['revenue net sales total revenue', 'cost of goods sold operating expenses net income gross profit']),
]


def _get_sub_queries(question: str) -> list[str]:
    """Detect ratio/formula queries and return sub-queries for each component."""
    q_lower = question.lower()
    for pattern, expansions in _RATIO_PATTERNS:
        if re.search(pattern, q_lower):
            return expansions
    return []


def _run_qa(tickers, question: str) -> dict:
    """Run a RAG query and return answer + sources with cited text.

    Args:
        tickers: a single ticker string or a list of ticker strings.
        question: the user's question.
    """
    # Normalise to a list
    if isinstance(tickers, str):
        tickers = [tickers]
    tickers = [t.upper() for t in tickers]

    settings = get_settings()
    rag = SupabaseRAG()
    rag.load_index()

    level = settings.rag_level
    if level == "advanced":
        retriever = AdvancedRetriever(rag.index)
    elif level == "intermediate":
        retriever = IntermediateRetriever(rag.index)
    else:
        retriever = BasicRetriever(rag.index)

    # Check if this is a ratio/formula query that needs multiple retrieval passes
    sub_queries = _get_sub_queries(question)

    # Retrieve nodes for each ticker and merge
    all_nodes = []
    per_ticker = max(settings.top_k // len(tickers), 3) if len(tickers) > 1 else settings.top_k

    for t in tickers:
        if sub_queries:
            # Ratio query: retrieve for each component to ensure both sides are covered
            per_sub = max(per_ticker // len(sub_queries), 2)
            seen_ids = set()
            for sq in sub_queries:
                full_q = f"{t} {sq}"
                nodes = retriever.retrieve(full_q, top_k=per_sub, ticker=t)
                for n in nodes:
                    nid = getattr(n, 'node_id', None) or (getattr(n.node, 'node_id', None) if hasattr(n, 'node') else id(n))
                    if nid not in seen_ids:
                        seen_ids.add(nid)
                        all_nodes.append(n)
            # Also retrieve with the original question for overall context
            nodes = retriever.retrieve(question, top_k=per_sub, ticker=t)
            for n in nodes:
                nid = getattr(n, 'node_id', None) or (getattr(n.node, 'node_id', None) if hasattr(n, 'node') else id(n))
                if nid not in seen_ids:
                    seen_ids.add(nid)
                    all_nodes.append(n)
        else:
            nodes = retriever.retrieve(question, top_k=per_ticker, ticker=t)
            all_nodes.extend(nodes)

    ticker_label = " vs ".join(tickers)

    if not all_nodes:
        return {
            "answer": f"No relevant data found for {ticker_label}. You may need to download and index SEC filings first.",
            "sources": [],
        }

    # Build numbered sources with the actual chunk text
    sources = []
    context_parts = []
    for i, node in enumerate(all_nodes):
        text = node.text if hasattr(node, "text") else ""
        if not text and hasattr(node, "node"):
            text = node.node.text if hasattr(node.node, "text") else ""

        meta = node.metadata if hasattr(node, "metadata") else {}
        if not meta and hasattr(node, "node"):
            meta = node.node.metadata if hasattr(node.node, "metadata") else {}

        source_file = meta.get("source_file", "Unknown")
        filing_type = meta.get("filing_type", "")
        filing_date = meta.get("filing_date", "")
        section = meta.get("section", "")

        # Label for the citation, e.g. "AAPL_10K_2024-09-28.txt > Financial Statements"
        label = source_file
        if section:
            label += f" > {section}"

        sources.append({
            "id": i + 1,
            "file": source_file,
            "ticker": meta.get("ticker", tickers[0]),
            "filing_type": filing_type,
            "filing_date": filing_date,
            "section": section,
            "label": label,
            "text": text.strip(),
        })

        context_parts.append(f"[Source {i + 1}: {label}]\n{text.strip()}")

    context = "\n\n---\n\n".join(context_parts)

    # Synthesize answer with citation references
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    multi = len(tickers) > 1
    prompt_intro = (
        f"Based on the following SEC filing excerpts for {ticker_label}, "
        f"answer this question: {question}"
    )
    comparison_hint = (
        "\n- This is a comparison query — compare the companies side by side with specific numbers from each."
        if multi else ""
    )

    try:
        response = client.messages.create(
            model=settings.claude_model,
            max_tokens=2000 if multi else 1500,
            temperature=0.2,
            messages=[{
                "role": "user",
                "content": (
                    f"{prompt_intro}\n\n"
                    f"{context}\n\n"
                    "Instructions:\n"
                    "- Provide a clear, detailed answer with specific numbers and dates.\n"
                    "- Reference sources using [Source N] notation inline in your answer.\n"
                    "- If asked about a financial ratio, calculate it from the component data in the sources "
                    "(e.g., for debt-to-equity, find total debt and total equity, then divide).\n"
                    "- If the context doesn't contain enough information, say so.\n"
                    f"- Use paragraphs to organize your answer.{comparison_hint}"
                ),
            }],
        )
        answer = response.content[0].text
    except Exception as e:
        logger.error(f"QA synthesis failed: {e}")
        answer = f"Error generating answer: {e}"

    return {"answer": answer, "sources": sources}


@router.post("/api/qa")
def qa_query(req: QARequest) -> QAResponse:
    """Direct Q&A endpoint."""
    result = _run_qa(req.ticker, req.question)  # single ticker string is fine
    return QAResponse(
        answer=result["answer"],
        sources=result["sources"],
        ticker=req.ticker.upper(),
    )
