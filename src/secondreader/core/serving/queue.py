"""Postgres/SQLite-backed job queue (D-004). Small, transactional, swappable."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from secondreader.core.db.models import Job, JobStatus


def enqueue(session: Session, kind: str, payload: dict[str, Any], *, max_attempts: int = 3) -> Job:
    job = Job(kind=kind, payload=payload, max_attempts=max_attempts)
    session.add(job)
    session.flush()
    return job


def claim_next(session: Session, worker: str, kinds: tuple[str, ...] | None = None) -> Job | None:
    """Claim the oldest queued job. Uses SKIP LOCKED where the backend supports it."""
    stmt = select(Job).where(Job.status == JobStatus.queued).order_by(Job.created_at).limit(1)
    if kinds:
        stmt = stmt.where(Job.kind.in_(kinds))
    if session.bind is not None and session.bind.dialect.name == "postgresql":
        stmt = stmt.with_for_update(skip_locked=True)
    job = session.scalars(stmt).first()
    if job is None:
        return None
    job.status = JobStatus.running
    job.worker = worker
    job.attempts += 1
    job.started_at = datetime.now(UTC)
    session.flush()
    return job


def complete(session: Session, job: Job) -> None:
    job.status = JobStatus.done
    job.finished_at = datetime.now(UTC)
    session.flush()


def fail(session: Session, job: Job, error: str) -> None:
    job.error = error[:4000]
    job.finished_at = datetime.now(UTC)
    job.status = JobStatus.queued if job.attempts < job.max_attempts else JobStatus.failed
    if job.status == JobStatus.queued:
        job.started_at = None
    session.flush()
