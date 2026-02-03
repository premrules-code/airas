# ── Stage 1: Build React frontend ────────────────────────────────────
FROM node:20-alpine AS frontend

WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ .
RUN npm run build

# ── Stage 2: Python backend + static files ──────────────────────────
FROM python:3.11-slim

# System deps for psycopg2, lxml, numpy
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev libxml2-dev libxslt1-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (Docker layer caching)
# Use prod requirements (excludes test deps like ragas/pytest)
COPY backend/requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy backend source
COPY backend/ .

# Copy built frontend from stage 1
COPY --from=frontend /app/frontend/dist /app/static

# Create data/logs directories
RUN mkdir -p data/raw logs

# Railway injects PORT env var; default to 8001
ENV PORT=8001

EXPOSE ${PORT}

CMD ["python", "run_server.py"]
