"""FastAPI application factory."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from config.settings import get_settings
from config.logging_config import setup_logging
from src.utils.llama_setup import configure_llama_index
from src.utils.langfuse_setup import setup_langfuse

from src.api.routes import health, analysis, query, qa, sec

logger = logging.getLogger(__name__)

# Built frontend directory (populated by Docker build or manual `npm run build`)
STATIC_DIR = Path(__file__).resolve().parent.parent.parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown hooks."""
    settings = get_settings()
    setup_logging(settings.log_level)
    configure_llama_index()
    setup_langfuse()
    logger.info("AIRAS API started")
    if STATIC_DIR.exists():
        logger.info(f"Serving frontend from {STATIC_DIR}")
    else:
        logger.info("No static dir found — frontend served by Vite dev server")
    yield
    logger.info("AIRAS API shutting down")


def create_app() -> FastAPI:
    app = FastAPI(
        title="AIRAS V3 API",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API routes first (take priority over static files)
    app.include_router(health.router)
    app.include_router(analysis.router)
    app.include_router(query.router)
    app.include_router(qa.router)
    app.include_router(sec.router)

    # Serve built frontend assets if the static dir exists (Docker / production)
    if STATIC_DIR.exists():
        # Mount Vite's asset files (JS, CSS, images)
        assets_dir = STATIC_DIR / "assets"
        if assets_dir.exists():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

        # SPA catch-all: any non-API route returns index.html
        @app.get("/{path:path}")
        async def spa_fallback(request: Request, path: str):
            # Serve actual static files (favicon, etc.)
            file = STATIC_DIR / path
            if path and file.exists() and file.is_file():
                return FileResponse(str(file))
            # Everything else → index.html (React Router handles it)
            return FileResponse(str(STATIC_DIR / "index.html"))

    return app


app = create_app()
