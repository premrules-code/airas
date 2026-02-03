"""Health check and company listing routes."""

import logging
from fastapi import APIRouter
from sqlalchemy import create_engine, text

from config.settings import get_settings
from src.api.schemas import CompanyInfo

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/api/health")
def health():
    return {"status": "online"}


@router.get("/api/companies")
def list_companies() -> dict:
    """List companies that have indexed data in Supabase."""
    settings = get_settings()
    try:
        engine = create_engine(settings.postgres_connection_string)
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT
                    metadata_->>'ticker' AS ticker,
                    COUNT(DISTINCT metadata_->>'source_file') AS files_count,
                    MAX(metadata_->>'created_at') AS last_indexed
                FROM data_airas_documents
                WHERE metadata_->>'ticker' IS NOT NULL
                GROUP BY metadata_->>'ticker'
                ORDER BY metadata_->>'ticker'
            """))
            companies = [
                CompanyInfo(
                    ticker=row[0],
                    files_count=row[1],
                    last_indexed=row[2],
                ).model_dump()
                for row in result.fetchall()
            ]
        return {"companies": companies}
    except Exception as e:
        logger.warning(f"Could not list companies: {e}")
        return {"companies": []}
