"""Galileo-powered guardrails for input/output validation."""

import logging
from typing import Optional
from config.settings import get_settings
from src.utils.galileo_setup import is_galileo_available

logger = logging.getLogger(__name__)


def check_hallucination(
    response: str,
    context: list[str],
    threshold: float = 0.7,
) -> dict:
    """
    Check if response is grounded in context using Galileo.

    Args:
        response: The LLM output to check
        context: List of source documents/chunks
        threshold: Minimum groundedness score (0-1) to pass

    Returns:
        {
            "is_grounded": bool,
            "score": float,
            "flagged_claims": list[str],
        }
    """
    default_result = {"is_grounded": True, "score": 1.0, "flagged_claims": []}

    if not is_galileo_available():
        return default_result

    if not response or not context:
        return default_result

    try:
        import galileo_observe as galileo

        result = galileo.evaluate.groundedness(
            response=response[:5000],
            contexts=[c[:5000] for c in context[:5]],  # Limit size
        )

        return {
            "is_grounded": result.score >= threshold,
            "score": result.score,
            "flagged_claims": result.flagged_spans or [],
        }
    except Exception as e:
        logger.debug(f"Galileo groundedness check failed: {e}")
        return default_result


def check_pii(text: str) -> dict:
    """
    Detect PII in text using Galileo.

    Args:
        text: Text to scan for PII

    Returns:
        {
            "has_pii": bool,
            "pii_types": list[str],
            "redacted_text": str,
        }
    """
    default_result = {"has_pii": False, "pii_types": [], "redacted_text": text}

    if not is_galileo_available():
        return default_result

    if not text:
        return default_result

    try:
        import galileo_observe as galileo

        result = galileo.evaluate.pii(text=text[:10000])

        return {
            "has_pii": len(result.detected) > 0 if result.detected else False,
            "pii_types": [p.type for p in result.detected] if result.detected else [],
            "redacted_text": result.redacted_text if hasattr(result, 'redacted_text') else text,
        }
    except Exception as e:
        logger.debug(f"Galileo PII check failed: {e}")
        return default_result


def check_toxicity(text: str, threshold: float = 0.7) -> dict:
    """
    Check text for toxicity using Galileo.

    Args:
        text: Text to check
        threshold: Score above which content is considered toxic

    Returns:
        {
            "is_toxic": bool,
            "score": float,
            "categories": list[str],
        }
    """
    default_result = {"is_toxic": False, "score": 0.0, "categories": []}

    if not is_galileo_available():
        return default_result

    if not text:
        return default_result

    try:
        import galileo_observe as galileo

        result = galileo.evaluate.toxicity(text=text[:5000])

        return {
            "is_toxic": result.score >= threshold,
            "score": result.score,
            "categories": result.categories if hasattr(result, 'categories') else [],
        }
    except Exception as e:
        logger.debug(f"Galileo toxicity check failed: {e}")
        return default_result


def check_context_relevance(query: str, chunks: list[str]) -> dict:
    """
    Score relevance of retrieved chunks to query.

    Args:
        query: The user's query
        chunks: Retrieved document chunks

    Returns:
        {
            "scores": list[float],
            "avg_score": float,
            "relevant_indices": list[int],
        }
    """
    default_scores = [1.0] * len(chunks) if chunks else []
    default_result = {
        "scores": default_scores,
        "avg_score": 1.0,
        "relevant_indices": list(range(len(chunks))),
    }

    if not is_galileo_available():
        return default_result

    if not query or not chunks:
        return default_result

    try:
        import galileo_observe as galileo

        result = galileo.evaluate.context_relevance(
            query=query[:1000],
            contexts=[c[:2000] for c in chunks[:10]],
        )

        scores = result.scores if hasattr(result, 'scores') else default_scores
        avg_score = sum(scores) / len(scores) if scores else 0.0
        relevant = [i for i, s in enumerate(scores) if s >= 0.5]

        return {
            "scores": scores,
            "avg_score": avg_score,
            "relevant_indices": relevant,
        }
    except Exception as e:
        logger.debug(f"Galileo context relevance check failed: {e}")
        return default_result


def validate_agent_output(
    summary: str,
    rag_context: str,
    check_hallucination_flag: bool = True,
    check_pii_flag: bool = True,
) -> dict:
    """
    Run all relevant validations on agent output.

    Args:
        summary: Agent's summary output
        rag_context: RAG context used for the analysis
        check_hallucination_flag: Whether to check for hallucination
        check_pii_flag: Whether to check for PII

    Returns:
        {
            "is_valid": bool,
            "hallucination": {...},
            "pii": {...},
            "warnings": list[str],
        }
    """
    warnings = []
    result = {
        "is_valid": True,
        "hallucination": {"is_grounded": True, "score": 1.0, "flagged_claims": []},
        "pii": {"has_pii": False, "pii_types": [], "redacted_text": summary},
        "warnings": warnings,
    }

    # Check hallucination
    if check_hallucination_flag and rag_context:
        hallucination_result = check_hallucination(summary, [rag_context])
        result["hallucination"] = hallucination_result

        if not hallucination_result["is_grounded"]:
            warnings.append(
                f"Low groundedness ({hallucination_result['score']:.0%}): "
                f"output may contain hallucinated claims"
            )
            if hallucination_result["score"] < 0.5:
                result["is_valid"] = False

    # Check PII
    if check_pii_flag:
        pii_result = check_pii(summary)
        result["pii"] = pii_result

        if pii_result["has_pii"]:
            warnings.append(f"PII detected: {', '.join(pii_result['pii_types'])}")

    result["warnings"] = warnings
    return result
