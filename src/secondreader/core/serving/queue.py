"""Postgres/SQLite-backed job queue (D-004). Small, transactional, swappable.

Retries: `fail()` re-queues a job until `max_attempts` is exhausted. Kinds listed in
`RETRY_BACKOFF` wait `min(cap, base * 2 ** (attempts - 1))` seconds before they can be
claimed again (D-030: archive pushes back off instead of hammering an Orthanc that is
down); every other kind retries immediately, as before.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from secondreader.core.db.models import Job, JobStatus

# kind → (base seconds, cap seconds). 5, 10, 20, 40, 80 s for the five push attempts.
RETRY_BACKOFF: dict[str, tuple[float, float]] = {"push_results": (5.0, 300.0)}


def retry_delay(kind: str, attempts: int) -> float:
    """Seconds to wait before retry number `attempts + 1` of a `kind` job (0 = immediately)."""
    policy = RETRY_BACKOFF.get(kind)
    if policy is None or attempts < 1:
        return 0.0
    base, cap = policy
    return float(min(cap, base * 2 ** (attempts - 1)))


def enqueue(
    session: Session,
    kind: str,
    payload: dict[str, Any],
    *,
    max_attempts: int = 3,
    not_before: datetime | None = None,
) -> Job:
    job = Job(kind=kind, payload=payload, max_attempts=max_attempts, not_before=not_before)
    session.add(job)
    session.flush()
    return job


def claim_next(
    session: Session,
    worker: str,
    kinds: tuple[str, ...] | None = None,
    *,
    now: datetime | None = None,
) -> Job | None:
    """Claim the oldest queued job whose backoff has elapsed. SKIP LOCKED where supported."""
    now = now or datetime.now(UTC)
    stmt = (
        select(Job)
        .where(Job.status == JobStatus.queued)
        .where(or_(Job.not_before.is_(None), Job.not_before <= now))
        .order_by(Job.created_at)
        .limit(1)
    )
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
    job.not_before = None
    session.flush()
    return job


def complete(session: Session, job: Job) -> None:
    job.status = JobStatus.done
    job.finished_at = datetime.now(UTC)
    session.flush()


def fail(session: Session, job: Job, error: str) -> None:
    """Record the error; re-queue (with the kind's backoff) or mark failed when exhausted."""
    job.error = error[:4000]
    job.finished_at = datetime.now(UTC)
    job.status = JobStatus.queued if job.attempts < job.max_attempts else JobStatus.failed
    if job.status == JobStatus.queued:
        job.started_at = None
        delay = retry_delay(job.kind, job.attempts)
        job.not_before = job.finished_at + timedelta(seconds=delay) if delay > 0 else None
    session.flush()
