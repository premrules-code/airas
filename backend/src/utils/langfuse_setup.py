"""Langfuse integration for tracing."""

import logging
from typing import Optional
from functools import wraps
from langfuse import Langfuse
from config.settings import get_settings

logger = logging.getLogger(__name__)
_langfuse_client: Optional[Langfuse] = None

def setup_langfuse() -> Optional[Langfuse]:
    """Initialize Langfuse client."""
    global _langfuse_client
    settings = get_settings()
    
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        logger.warning("⚠️  Langfuse not configured")
        return None
    
    _langfuse_client = Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host
    )
    
    logger.info("✅ Langfuse connected")
    return _langfuse_client

def trace_agent(agent_name: str):
    """Decorator to trace agent execution."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if _langfuse_client is None:
                return func(*args, **kwargs)

            ticker = kwargs.get('ticker') or (args[1] if len(args) > 1 else 'UNKNOWN')

            span = _langfuse_client.start_span(
                name=f"{agent_name}_analysis",
                metadata={"agent": agent_name, "ticker": ticker},
                input={"ticker": ticker},
            )

            try:
                result = func(*args, **kwargs)
                span.update(output={"status": "success"})
                return result
            except Exception:
                span.update(output={"status": "error"})
                raise
            finally:
                span.end()

        return wrapper
    return decorator