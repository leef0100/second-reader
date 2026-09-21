"""Job handlers registered with the worker."""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from secondreader.core.db.models import Instance, Job
from secondreader.core.serving.deps import Container
from secondreader.core.serving.inference import run_inference_for_instance
from secondreader.core.serving.worker import register_handler


@register_handler("infer")
def infer(c: Container, session: Session, job: Job) -> dict[str, Any] | None:
    inst = session.get(Instance, str(job.payload["instance_id"]))
    if inst is None:
        raise RuntimeError(f"instance {job.payload.get('instance_id')} not found")
    if inst.quarantined:
        return {"skipped": "quarantined"}
    pred = run_inference_for_instance(c, session, inst, request_id=job.id)
    return {"prediction_id": pred.id, "gate": pred.gate_status}
