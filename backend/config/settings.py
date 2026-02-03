"""Application settings using Pydantic."""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
from pathlib import Path


class Settings(BaseSettings):
    """AIRAS V3 Configuration."""
    
    # API Keys
    openai_api_key: str = Field(..., description="OpenAI API key")
    anthropic_api_key: str = Field(..., description="Anthropic API key")
    
    # Supabase
    supabase_url: str = Field(..., description="Supabase URL")
    supabase_key: str = Field(..., description="Supabase key")
    postgres_connection_string: str = Field(..., description="PostgreSQL connection")
    
    # Langfuse (optional)
    langfuse_public_key: Optional[str] = None
    langfuse_secret_key: Optional[str] = None
    langfuse_host: str = "https://cloud.langfuse.com"
    
    # Reddit (optional — for social sentiment)
    reddit_client_id: Optional[str] = None
    reddit_client_secret: Optional[str] = None

    # Financial Modeling Prep (optional — replaces yfinance for market data)
    fmp_api_key: Optional[str] = None

    # Finnhub (optional — insider trades + news)
    finnhub_api_key: Optional[str] = None

    # Tradier (optional — options data, sandbox is free)
    tradier_api_token: Optional[str] = None
    tradier_sandbox: bool = True  # Use sandbox endpoint by default

    # App
    environment: str = "development"
    log_level: str = "INFO"
    api_port: int = 8001
    port: Optional[int] = None  # Railway injects PORT; takes priority over api_port
    
    # SEC
    sec_user_email: str = "premrules@gmail.com"
    
    # RAG
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k: int = 5
    rag_level: str = "intermediate"  # "basic" | "intermediate" | "advanced"
    
    # Models
    openai_model: str = "gpt-4-turbo-preview"
    openai_embedding_model: str = "text-embedding-3-small"
    claude_model: str = "claude-sonnet-4-20250514"
    
    # Paths
    data_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    logs_dir: Path = Path("logs")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get cached settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


if __name__ == "__main__":
    s = get_settings()
    print(f"Environment: {s.environment}")
    print(f"Supabase URL: {s.supabase_url}")