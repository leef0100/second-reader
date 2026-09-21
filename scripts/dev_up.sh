#!/usr/bin/env bash
# Start the local stack (Docker Compose) and the API from the working tree.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] || { cp .env.example .env; echo "created .env from .env.example — edit secrets"; }
docker compose up -d postgres orthanc-a orthanc-b
echo "waiting for postgres..."; until docker compose exec -T postgres pg_isready -U secondreader >/dev/null 2>&1; do sleep 1; done
uv run alembic upgrade head
echo "API: uv run uvicorn secondreader.core.serving.app:app --reload --port 8000"
