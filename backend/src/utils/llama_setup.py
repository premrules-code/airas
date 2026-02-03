"""Configure LlamaIndex."""

import logging
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from config.settings import get_settings

logger = logging.getLogger(__name__)


def configure_llama_index():
    """Configure global LlamaIndex settings."""
    settings = get_settings()
    
    logger.info("⚙️  Configuring LlamaIndex...")
    
    # Embeddings
    Settings.embed_model = OpenAIEmbedding(
        model=settings.openai_embedding_model,
        api_key=settings.openai_api_key
    )
    
    # LLM - Fix: explicitly set default_headers to empty dict
    Settings.llm = OpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=0.1,
        default_headers={}
    )
    
    Settings.chunk_size = settings.chunk_size
    Settings.chunk_overlap = settings.chunk_overlap
    
    logger.info(f"   Embedding: {settings.openai_embedding_model}")
    logger.info(f"   LLM: {settings.openai_model}")
    logger.info("✅ LlamaIndex configured")
