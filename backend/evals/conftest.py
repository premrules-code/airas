"""Pytest configuration and shared fixtures for RAG evaluation."""

import json
import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import get_settings
from src.utils.llama_setup import configure_llama_index
from src.rag.supabase_rag import SupabaseRAG


@pytest.fixture(scope="session")
def settings():
    """Get application settings."""
    return get_settings()


@pytest.fixture(scope="session")
def rag_index(settings):
    """Load RAG index (shared across all tests in session)."""
    configure_llama_index()
    rag = SupabaseRAG()
    rag.load_index()
    return rag.index


@pytest.fixture(scope="session")
def golden_set():
    """Load golden dataset."""
    golden_path = Path(__file__).parent / "datasets" / "golden_set.json"
    with open(golden_path) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def langfuse_client(settings):
    """Get Langfuse client for score storage."""
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        return None

    try:
        from langfuse import Langfuse
        return Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
    except Exception as e:
        print(f"Warning: Could not initialize Langfuse: {e}")
        return None


@pytest.fixture
def retrieval_evaluator():
    """Get retrieval metrics evaluator."""
    from evals.metrics.retrieval_metrics import RetrievalEvaluator
    return RetrievalEvaluator()


@pytest.fixture
def ragas_evaluator(langfuse_client):
    """Get RAGAS evaluator with Langfuse integration."""
    from evals.metrics.ragas_metrics import RAGASEvaluator
    return RAGASEvaluator(langfuse_client=langfuse_client)


@pytest.fixture
def cost_tracker():
    """Get fresh cost/latency tracker."""
    from evals.metrics.cost_latency import CostLatencyTracker
    return CostLatencyTracker()


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "requires_api: marks tests that require API calls"
    )


def pytest_collection_modifyitems(config, items):
    """Add markers based on test characteristics."""
    for item in items:
        # Mark RAGAS tests as slow (they make LLM calls)
        if "ragas" in item.nodeid.lower():
            item.add_marker(pytest.mark.slow)
            item.add_marker(pytest.mark.requires_api)
