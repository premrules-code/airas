#!/usr/bin/env python3
"""Uvicorn launcher for the AIRAS API."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import uvicorn
from config.settings import get_settings

if __name__ == "__main__":
    settings = get_settings()
    # Railway injects PORT env var; fall back to settings.api_port for local dev
    port = settings.port or settings.api_port
    is_dev = settings.environment == "development"
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=port,
        reload=is_dev,
    )
