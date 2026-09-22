"""Job handlers registered with the worker."""

from __future__ import annotations

from typing import Any

import structlog
from sqlalchemy.orm import Session

from secondreader.core.db.models import Instance, Job, Prediction
from secondreader.core.serving import queue
from secondreader.core.serving.deps import Container
from secondreader.core.serving.inference import (
    push_results_to_archive,
    run_inference_for_instance,
)
from secondreader.core.serving.worker import register_handler

log = structlog.get_logger(__name__)


@register_handler("infer")
def infer(c: Container, session: Session, job: Job) -> dict[str, Any] | None:
    inst = session.get(Instance, str(job.payload["instance_id"]))
    if inst is None:
        raise RuntimeError(f"instance {job.payload.get('instance_id')} not found")
    if inst.quarantined:
        return {"skipped": "quarantined"}
    pred = run_inference_for_instance(c, session, inst, request_id=job.id)
    return {"prediction_id": pred.id, "gate": pred.gate_status}


@register_handler("push_results")
def push_results(c: Container, session: Session, job: Job) -> dict[str, Any] | None:
    """Push one prediction's de-identified image + SR + SC overlays to Orthanc B (D-030).

    Raising makes the worker re-queue the job with backoff (or mark it failed on the
    last attempt); the prediction row always records where the push stands. The
    status write is committed here because the worker rolls the session back on error.
    """
    pred = session.get(Prediction, str(job.payload["prediction_id"]))
    if pred is None:
        raise RuntimeError(f"prediction {job.payload.get('prediction_id')} not found")
    try:
        status = push_results_to_archive(c, session, pred, job_id=job.id, attempt=job.attempts)
    except Exception as exc:
        final = job.attempts >= job.max_attempts
        pred.push_status = {
            "state": "failed" if final else "retrying",
            "attempts": job.attempts,
            "max_attempts": job.max_attempts,
            "job_id": job.id,
            "error": f"{type(exc).__name__}: {exc}"[:500],
            "retry_in_s": None if final else queue.retry_delay(job.kind, job.attempts),
        }
        session.commit()
        log.warning(
            "push_results_failed",
            prediction_id=pred.id,
            attempt=job.attempts,
            final=final,
            error=str(exc),
        )
        raise
    return status
