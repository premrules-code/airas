"""SEC filing download and indexing routes."""

import logging
import threading
from pathlib import Path

from fastapi import APIRouter

from config.settings import get_settings
from src.api.schemas import SECDownloadRequest, SECStatusResponse

logger = logging.getLogger(__name__)
router = APIRouter()

# Track background SEC tasks: ticker -> {download_status, index_status, error}
_sec_status: dict[str, dict] = {}
_sec_lock = threading.Lock()


def _get_ticker_status(ticker: str) -> dict:
    with _sec_lock:
        return _sec_status.get(ticker.upper(), {
            "download_status": "idle",
            "index_status": "idle",
            "files_count": 0,
            "error": None,
        })


def _set_ticker_status(ticker: str, **kwargs):
    with _sec_lock:
        t = ticker.upper()
        if t not in _sec_status:
            _sec_status[t] = {
                "download_status": "idle",
                "index_status": "idle",
                "files_count": 0,
                "error": None,
            }
        _sec_status[t].update(kwargs)


def _download_worker(ticker: str, filing_types: list[str], num_filings: int):
    """Background thread for SEC download."""
    try:
        _set_ticker_status(ticker, download_status="running", error=None)

        from scripts.download_sec_filings import SECDownloader
        downloader = SECDownloader()
        downloader.download_filings(ticker.upper(), filing_types, num_filings)

        # Count downloaded files
        settings = get_settings()
        raw_dir = settings.raw_dir
        files = list(raw_dir.glob(f"{ticker.upper()}_*.txt"))
        _set_ticker_status(ticker, download_status="done", files_count=len(files))

    except Exception as e:
        logger.exception(f"SEC download failed for {ticker}")
        _set_ticker_status(ticker, download_status="error", error=str(e))


def _index_worker(ticker: str):
    """Background thread for building vector index."""
    try:
        _set_ticker_status(ticker, index_status="running", error=None)

        from src.utils.llama_setup import configure_llama_index
        from src.rag.supabase_rag import SupabaseRAG
        from scripts.smart_build_index import get_indexed_files, make_document

        settings = get_settings()
        configure_llama_index()

        raw_dir = settings.raw_dir
        all_files = sorted(raw_dir.glob(f"{ticker.upper()}_*.txt"))

        if not all_files:
            _set_ticker_status(ticker, index_status="error", error="No files found to index")
            return

        indexed = get_indexed_files(settings.postgres_connection_string)
        new_files = [f for f in all_files if f.name not in indexed]

        if not new_files:
            _set_ticker_status(ticker, index_status="done")
            return

        documents = [make_document(f) for f in new_files]

        rag = SupabaseRAG()
        rag.setup_database()
        rag.build_index(documents)

        _set_ticker_status(ticker, index_status="done", files_count=len(all_files))

    except Exception as e:
        logger.exception(f"SEC indexing failed for {ticker}")
        _set_ticker_status(ticker, index_status="error", error=str(e))


@router.post("/api/sec/download")
def download_sec(req: SECDownloadRequest) -> dict:
    """Start SEC filing download in background."""
    ticker = req.ticker.upper()
    t = threading.Thread(
        target=_download_worker,
        args=(ticker, req.filing_types, req.num_filings),
        daemon=True,
    )
    t.start()
    return {"ticker": ticker, "status": "download_started"}


@router.post("/api/sec/index")
def index_sec(req: SECDownloadRequest) -> dict:
    """Build vector index for a ticker's downloaded filings."""
    ticker = req.ticker.upper()
    t = threading.Thread(target=_index_worker, args=(ticker,), daemon=True)
    t.start()
    return {"ticker": ticker, "status": "index_started"}


@router.get("/api/sec/status/{ticker}")
def sec_status(ticker: str) -> SECStatusResponse:
    """Check download/index status for a ticker."""
    status = _get_ticker_status(ticker)
    return SECStatusResponse(ticker=ticker.upper(), **status)
