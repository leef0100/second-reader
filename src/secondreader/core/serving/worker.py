"""Job worker loop. Phase 2 knows how to claim/complete; Phase 4 adds the inference handler."""

from __future__ import annotations

import socket
import time
from collections.abc import Callable
from typing import Any

import structlog
from sqlalchemy.orm import Session

from secondreader.core.db.models import Job
from secondreader.core.serving import queue
from secondreader.core.serving.deps import Container, get_container

log = structlog.get_logger(__name__)

Handler = Callable[[Container, Session, Job], dict[str, Any] | None]
_HANDLERS: dict[str, Handler] = {}


def register_handler(kind: str) -> Callable[[Handler], Handler]:
    def deco(fn: Handler) -> Handler:
        _HANDLERS[kind] = fn
        return fn

    return deco


def process_one(c: Container, worker_name: str) -> bool:
    with c.sessions() as session:
        job = queue.claim_next(session, worker_name, kinds=tuple(_HANDLERS) or None)
        if job is None:
            session.commit()
            return False
        session.commit()
        handler = _HANDLERS.get(job.kind)
        try:
            if handler is None:
                raise RuntimeError(f"no handler for job kind {job.kind!r}")
            handler(c, session, job)
            queue.complete(session, job)
            session.commit()
            log.info("job_done", job_id=job.id, kind=job.kind)
        except Exception as exc:  # noqa: BLE001
            session.rollback()
            queue.fail(session, job, f"{type(exc).__name__}: {exc}")
            session.commit()
            log.warning(
                "job_failed", job_id=job.id, kind=job.kind, error=str(exc), status=job.status.value
            )
        return True


def run_worker(*, once: bool = False, poll_seconds: float = 2.0) -> None:
    from secondreader.core.serving import handlers  # noqa: F401  registers job handlers

    c = get_container()
    name = f"{socket.gethostname()}-{time.time_ns() % 100000}"
    log.info("worker_start", worker=name, handlers=sorted(_HANDLERS))
    while True:
        did = process_one(c, name)
        if once:
            return
        if not did:
            time.sleep(poll_seconds)
