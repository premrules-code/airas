"""Analysis routes — start, poll, and stream analysis jobs."""

import json
import asyncio
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from src.api.schemas import AnalysisRequest, AnalysisJobResponse, AnalysisResultResponse
from src.api.jobs import start_analysis, get_job

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/api/analysis")
def create_analysis(req: AnalysisRequest) -> AnalysisJobResponse:
    """Start a new analysis job. Returns immediately with job_id."""
    job = start_analysis(req.ticker, req.query, req.rag_level)
    return AnalysisJobResponse(job_id=job.job_id, status=job.status, ticker=job.ticker)


@router.get("/api/analysis/{job_id}")
def get_analysis(job_id: str) -> AnalysisResultResponse:
    """Poll for analysis results."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return AnalysisResultResponse(
        job_id=job.job_id,
        status=job.status,
        ticker=job.ticker,
        agents=job.agent_outputs,
        recommendation=job.recommendation,
        errors=job.errors,
    )


@router.get("/api/analysis/{job_id}/stream")
async def stream_analysis(job_id: str):
    """SSE stream of analysis progress events."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_generator():
        cursor = 0
        heartbeat_counter = 0
        while True:
            new_events = job.snapshot_events(after=cursor)
            for evt in new_events:
                event_type = evt.get("event", "message")
                data = json.dumps(evt.get("data", evt))
                yield f"event: {event_type}\ndata: {data}\n\n"
                cursor += 1
                heartbeat_counter = 0

                if event_type in ("done", "error"):
                    return

            if job.status in ("done", "error") and not new_events:
                yield f"event: done\ndata: {{}}\n\n"
                return

            # Send heartbeat comment every ~15 seconds to keep connection alive
            heartbeat_counter += 1
            if heartbeat_counter >= 15:
                yield ": heartbeat\n\n"
                heartbeat_counter = 0

            await asyncio.sleep(1)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
