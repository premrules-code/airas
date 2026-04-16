"""Galileo AI observability and evaluation setup."""

import logging
from functools import wraps
from typing import Callable, Any, Optional

from config.settings import get_settings

logger = logging.getLogger(__name__)

# Galileo client (lazy initialization)
_galileo_initialized = False
_galileo_available = False


def init_galileo() -> bool:
    """Initialize Galileo AI client."""
    global _galileo_initialized, _galileo_available

    if _galileo_initialized:
        return _galileo_available

    settings = get_settings()

    if not settings.galileo_api_key:
        logger.info("Galileo API key not set, skipping initialization")
        _galileo_initialized = True
        _galileo_available = False
        return False

    try:
        import galileo_observe as galileo

        galileo.init(
            project=settings.galileo_project,
            log_stream=settings.galileo_log_stream,
        )
        _galileo_initialized = True
        _galileo_available = True
        logger.info(f"Galileo initialized for project: {settings.galileo_project}")
        return True
    except ImportError:
        logger.warning("galileo-observe not installed, skipping initialization")
        _galileo_initialized = True
        _galileo_available = False
        return False
    except Exception as e:
        logger.warning(f"Failed to initialize Galileo: {e}")
        _galileo_initialized = True
        _galileo_available = False
        return False


def is_galileo_available() -> bool:
    """Check if Galileo is available and initialized."""
    global _galileo_initialized, _galileo_available
    if not _galileo_initialized:
        init_galileo()
    return _galileo_available


def log_llm_call(
    model: str,
    prompt: str,
    response: str,
    tokens_in: int = 0,
    tokens_out: int = 0,
    latency_ms: float = 0,
    metadata: Optional[dict] = None,
) -> None:
    """Log an LLM call to Galileo with automatic evaluation."""
    if not is_galileo_available():
        return

    try:
        import galileo_observe as galileo

        galileo.log_llm(
            model=model,
            input=prompt[:10000],  # Truncate very long prompts
            output=response[:5000],  # Truncate very long responses
            num_input_tokens=tokens_in,
            num_output_tokens=tokens_out,
            latency_ms=latency_ms,
            metadata=metadata or {},
        )
    except Exception as e:
        logger.debug(f"Failed to log LLM call to Galileo: {e}")


def log_rag_query(
    query: str,
    retrieved_chunks: list[str],
    response: str,
    chunk_scores: Optional[list[float]] = None,
    metadata: Optional[dict] = None,
) -> None:
    """Log a RAG query to Galileo with context relevance evaluation."""
    if not is_galileo_available():
        return

    try:
        import galileo_observe as galileo

        galileo.log_rag(
            query=query,
            contexts=retrieved_chunks[:10],  # Limit to 10 chunks
            response=response[:5000],
            context_scores=chunk_scores,
            metadata=metadata or {},
        )
    except Exception as e:
        logger.debug(f"Failed to log RAG query to Galileo: {e}")


def log_agent_output(
    agent_name: str,
    ticker: str,
    score: float,
    confidence: float,
    summary: str,
    sources: list[str],
    rag_context: str = "",
) -> dict:
    """
    Log agent output and run evaluations.

    Returns evaluation results:
        {
            "hallucination_score": float,
            "is_grounded": bool,
            "pii_detected": list,
            "flagged_claims": list,
        }
    """
    results = {
        "hallucination_score": 0.0,
        "is_grounded": True,
        "pii_detected": [],
        "flagged_claims": [],
    }

    if not is_galileo_available():
        return results

    try:
        import galileo_observe as galileo

        # Log the output
        galileo.log(
            name=f"agent-{agent_name}",
            output=summary,
            metadata={
                "agent_name": agent_name,
                "ticker": ticker,
                "score": score,
                "confidence": confidence,
                "sources": sources,
            },
        )

        # Run groundedness evaluation if we have RAG context
        if rag_context:
            try:
                groundedness = galileo.evaluate.groundedness(
                    response=summary,
                    contexts=[rag_context[:10000]],  # Truncate context
                )
                results["hallucination_score"] = 1.0 - groundedness.score
                results["is_grounded"] = groundedness.score >= 0.7
                results["flagged_claims"] = groundedness.flagged_spans or []
            except Exception as e:
                logger.debug(f"Groundedness evaluation failed: {e}")

        # Run PII detection
        try:
            pii_result = galileo.evaluate.pii(text=summary)
            results["pii_detected"] = [p.type for p in pii_result.detected] if pii_result.detected else []
        except Exception as e:
            logger.debug(f"PII detection failed: {e}")

        return results

    except Exception as e:
        logger.debug(f"Failed to log agent output to Galileo: {e}")
        return results


def trace_with_galileo(name: str):
    """
    Decorator to trace function calls with Galileo.

    Args:
        name: Name of the trace (e.g., "financial-analyst")
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not is_galileo_available():
                return func(*args, **kwargs)

            try:
                import galileo_observe as galileo

                with galileo.trace(name=name):
                    return func(*args, **kwargs)
            except Exception:
                return func(*args, **kwargs)

        return wrapper
    return decorator


def flush_galileo() -> None:
    """Flush Galileo logs before shutdown."""
    if not is_galileo_available():
        return

    try:
        import galileo_observe as galileo
        galileo.flush()
        logger.info("Galileo logs flushed")
    except Exception as e:
        logger.debug(f"Failed to flush Galileo: {e}")
