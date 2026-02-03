"""In-memory job store and background analysis runner."""

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from config.settings import get_settings
from src.agents.graph import build_analysis_graph
from src.agents.state import AnalysisState
from src.models.structured_outputs import AgentOutput, InvestmentRecommendation
from src.api.score_utils import (
    to_display_score,
    score_to_color,
    score_to_signal,
    AGENT_DISPLAY_META,
)

logger = logging.getLogger(__name__)


@dataclass
class Job:
    job_id: str
    ticker: str
    status: str = "pending"  # pending | running | done | error
    events: list = field(default_factory=list)
    agent_outputs: list = field(default_factory=list)
    recommendation: Optional[dict] = None
    errors: list = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def push_event(self, event: dict):
        with self._lock:
            self.events.append(event)

    def snapshot_events(self, after: int = 0) -> list:
        with self._lock:
            return list(self.events[after:])


_jobs: dict[str, Job] = {}
_jobs_lock = threading.Lock()


def get_job(job_id: str) -> Optional[Job]:
    return _jobs.get(job_id)


def _format_agent_output(o: AgentOutput) -> dict:
    meta = AGENT_DISPLAY_META.get(o.agent_name, {})
    ds = to_display_score(o.score)
    return {
        "agent_name": o.agent_name,
        "label": meta.get("label", o.agent_name),
        "icon": meta.get("icon", "circle"),
        "description": meta.get("description", ""),
        "internal_score": o.score,
        "display_score": ds,
        "color": score_to_color(ds),
        "signal": score_to_signal(ds),
        "summary": o.summary,
        "confidence": o.confidence,
        "strengths": o.strengths,
        "weaknesses": o.weaknesses,
        "sources": o.sources,
    }


def _format_recommendation(rec: InvestmentRecommendation) -> dict:
    ds = to_display_score(rec.overall_score)
    return {
        "ticker": rec.ticker,
        "company_name": rec.company_name,
        "recommendation": rec.recommendation,
        "confidence": rec.confidence,
        "overall_score": ds,
        "overall_color": score_to_color(ds),
        "financial_score": to_display_score(rec.financial_score),
        "technical_score": to_display_score(rec.technical_score),
        "sentiment_score": to_display_score(rec.sentiment_score),
        "risk_score": to_display_score(rec.risk_score),
        "bullish_factors": rec.bullish_factors,
        "bearish_factors": rec.bearish_factors,
        "risks": rec.risks,
        "thesis": rec.thesis,
        "num_agents": rec.num_agents,
    }


def _ticker_has_index(ticker: str) -> bool:
    """Check if a ticker has any indexed documents in Supabase."""
    try:
        from sqlalchemy import create_engine, text
        settings = get_settings()
        engine = create_engine(settings.postgres_connection_string)
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT COUNT(*) FROM data_airas_documents WHERE metadata_->>'ticker' = :ticker"),
                {"ticker": ticker.upper()},
            )
            count = result.scalar()
            return count > 0
    except Exception as e:
        logger.warning(f"Could not check index for {ticker}: {e}")
        return False


def _ensure_sec_data(job: Job):
    """Download 10-K filings and build vector index if ticker has no indexed data."""
    ticker = job.ticker

    if _ticker_has_index(ticker):
        logger.info(f"{ticker} already has indexed data, skipping SEC download")
        return

    settings = get_settings()

    # --- Download ---
    job.push_event({
        "event": "phase",
        "data": {"phase": "downloading", "message": f"Downloading SEC 10-K filings for {ticker}..."},
    })

    try:
        from scripts.download_sec_filings import SECDownloader
        downloader = SECDownloader()
        downloader.download_filings(ticker, ["10-K"], 3)
    except Exception as e:
        logger.error(f"SEC download failed for {ticker}: {e}")
        job.push_event({
            "event": "phase",
            "data": {"phase": "download_error", "message": f"SEC download failed: {e}. Analysis will proceed without filing data."},
        })
        return

    raw_dir = settings.raw_dir
    files = sorted(raw_dir.glob(f"{ticker}_*.txt"))
    if not files:
        job.push_event({
            "event": "phase",
            "data": {"phase": "download_error", "message": "No filing files found after download. Analysis will proceed without filing data."},
        })
        return

    job.push_event({
        "event": "phase",
        "data": {"phase": "indexing", "message": f"Building vector index for {len(files)} files..."},
    })

    # --- Index ---
    try:
        from src.utils.llama_setup import configure_llama_index
        from src.rag.supabase_rag import SupabaseRAG
        from scripts.smart_build_index import get_indexed_files, make_document

        configure_llama_index()
        indexed = get_indexed_files(settings.postgres_connection_string)
        new_files = [f for f in files if f.name not in indexed]

        if new_files:
            documents = [make_document(f) for f in new_files]
            rag = SupabaseRAG()
            rag.setup_database()
            rag.build_index(documents)

        job.push_event({
            "event": "phase",
            "data": {"phase": "index_done", "message": f"Index built: {len(new_files)} files indexed."},
        })
    except Exception as e:
        logger.error(f"SEC indexing failed for {ticker}: {e}")
        job.push_event({
            "event": "phase",
            "data": {"phase": "index_error", "message": f"Indexing failed: {e}. Analysis will proceed without filing data."},
        })


def _run_analysis(job: Job, query: Optional[str], rag_level: str):
    """Background thread target — streams LangGraph execution."""
    try:
        job.status = "running"

        # Auto-download and index SEC filings if needed
        _ensure_sec_data(job)

        graph = build_analysis_graph()

        initial_state: AnalysisState = {
            "ticker": job.ticker,
            "query": query,
            "active_agents": [],
            "mode": "full",
            "rag_level": rag_level,
            "rag_context": {},
            "agent_outputs": [],
            "recommendation": None,
            "trace_id": None,
            "errors": [],
        }

        # Use graph.stream() for per-node updates
        for step in graph.stream(initial_state):
            for node_name, state_update in step.items():
                if node_name == "route_query":
                    active = state_update.get("active_agents", [])
                    job.push_event({
                        "event": "phase",
                        "data": {"phase": "routing", "agents": active},
                    })

                elif node_name == "gather_context":
                    job.push_event({
                        "event": "phase",
                        "data": {"phase": "context_gathered"},
                    })

                elif node_name == "synthesis":
                    rec = state_update.get("recommendation")
                    if rec:
                        job.recommendation = _format_recommendation(rec)

                        # Also add synthesis as an 11th "agent" card
                        synth_ds = to_display_score(rec.overall_score)
                        synth_meta = AGENT_DISPLAY_META.get("synthesis", {})
                        synth_agent = {
                            "agent_name": "synthesis",
                            "label": synth_meta.get("label", "Enhanced Synthesis"),
                            "icon": synth_meta.get("icon", "zap"),
                            "description": synth_meta.get("description", ""),
                            "internal_score": rec.overall_score,
                            "display_score": synth_ds,
                            "color": score_to_color(synth_ds),
                            "signal": score_to_signal(synth_ds),
                            "summary": rec.thesis[:200] if rec.thesis else "",
                            "confidence": rec.confidence,
                            "strengths": rec.bullish_factors[:3],
                            "weaknesses": rec.bearish_factors[:3],
                        }
                        job.agent_outputs.append(synth_agent)
                        job.push_event({
                            "event": "agent_completed",
                            "data": synth_agent,
                        })

                        job.push_event({"event": "done", "data": job.recommendation})

                else:
                    # Agent node completed
                    outputs = state_update.get("agent_outputs", [])
                    for o in outputs:
                        formatted = _format_agent_output(o)
                        job.agent_outputs.append(formatted)
                        job.push_event({
                            "event": "agent_completed",
                            "data": formatted,
                        })

                    errors = state_update.get("errors", [])
                    job.errors.extend(errors)

        job.status = "done"

    except Exception as e:
        logger.exception(f"Analysis job {job.job_id} failed")
        job.status = "error"
        job.errors.append(str(e))
        job.push_event({"event": "error", "data": {"message": str(e)}})


def start_analysis(ticker: str, query: Optional[str] = None, rag_level: str = "intermediate") -> Job:
    """Create a job and start analysis in a background daemon thread."""
    job_id = str(uuid.uuid4())
    job = Job(job_id=job_id, ticker=ticker.upper())

    with _jobs_lock:
        _jobs[job_id] = job

    t = threading.Thread(target=_run_analysis, args=(job, query, rag_level), daemon=True)
    t.start()

    return job
