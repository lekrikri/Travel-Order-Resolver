#!/usr/bin/env bash
# Start backend (FastAPI) and frontend (Vite) in parallel.
# Press Ctrl+C to stop both.

trap 'kill 0; exit' INT TERM

echo "Starting backend on http://localhost:8000 ..."
uv run uvicorn src.server.app:app --reload --port 8000 &

echo "Starting frontend on http://localhost:3000 ..."
cd frontend && npm run dev &

wait
