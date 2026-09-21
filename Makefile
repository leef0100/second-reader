.PHONY: sync check lint type test test-int up down services

sync:
	uv sync --extra dev

lint:
	uv run ruff check src tests scripts
	uv run ruff format --check src tests scripts

type:
	uv run mypy

test:
	uv run pytest -q -m "not integration and not slow"

test-int:
	uv run pytest -q -m integration

check: lint type test

up:
	docker compose up -d --build

down:
	docker compose down
