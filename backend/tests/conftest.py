"""Shared fixtures and pytest configuration."""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Ensure project root is on sys.path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.structured_outputs import AgentOutput
from src.agents.tracing import TracingManager
from tests.fixtures.agent_outputs import make_agent_output, ALL_AGENT_NAMES
from tests.fixtures.claude_responses import make_claude_response, THESIS_RESPONSE


@pytest.fixture
def mock_settings():
    """Patch get_settings() with test env vars (no real API keys)."""
    mock = MagicMock()
    mock.openai_api_key = "test-openai-key"
    mock.anthropic_api_key = "test-anthropic-key"
    mock.supabase_url = "https://test.supabase.co"
    mock.supabase_key = "test-supabase-key"
    mock.postgres_connection_string = "postgresql://test:test@localhost/test"
    mock.langfuse_public_key = None
    mock.langfuse_secret_key = None
    mock.reddit_client_id = None
    mock.reddit_client_secret = None
    mock.fmp_api_key = None
    mock.finnhub_api_key = None
    mock.tradier_api_token = None
    mock.environment = "test"
    mock.claude_model = "claude-sonnet-4-20250514"
    mock.openai_embedding_model = "text-embedding-3-small"
    mock.chunk_size = 512
    mock.chunk_overlap = 50
    mock.top_k = 5
    mock.rag_level = "basic"
    with patch("config.settings.get_settings", return_value=mock), \
         patch("config.settings._settings", mock):
        yield mock


@pytest.fixture
def mock_anthropic():
    """Patch anthropic.Anthropic to return configurable responses.

    Usage in tests:
        def test_something(mock_anthropic):
            mock_anthropic.return_value.messages.create.return_value = some_response
    """
    with patch("anthropic.Anthropic") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        yield mock_cls


@pytest.fixture
def sample_agent_outputs() -> list[AgentOutput]:
    """Return a list of 10 AgentOutput objects (one per agent)."""
    scores = [0.5, 0.3, 0.4, -0.1, 0.2, 0.1, -0.05, 0.15, 0.35, 0.25]
    confidences = [0.85, 0.7, 0.8, 0.6, 0.7, 0.55, 0.4, 0.35, 0.75, 0.8]
    return [
        make_agent_output(name, score=s, confidence=c)
        for name, s, c in zip(ALL_AGENT_NAMES, scores, confidences)
    ]


@pytest.fixture
def null_tracer() -> TracingManager:
    """Return a TracingManager with langfuse=None."""
    tracer = TracingManager()
    tracer.langfuse = None
    return tracer


@pytest.fixture
def mock_thesis_generation():
    """Patch anthropic in synthesis to return a thesis response."""
    with patch("src.agents.synthesis.anthropic.Anthropic") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create.return_value = THESIS_RESPONSE
        yield mock_cls
