"""Synchronous inference: DICOM in → JSON (+ SR/SC stored and downloadable) out."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from secondreader.core.audit import AuditLogger
from secondreader.core.constants import RESEARCH_USE_LABEL
from secondreader.core.db.models import Instance
from secondreader.core.ingest.service import PhiLeakError
from secondreader.core.security import Principal
from secondreader.core.serving.deps import Container
from secondreader.core.serving.inference import NoPluginError, run_inference_for_instance
from secondreader.core.serving.routes.common import audit, container, db, principal

router = APIRouter(prefix="/v1", tags=["inference"])


@router.post("/infer")
async def infer(
    file: UploadFile = File(...),
    s: Session = Depends(db),
    p: Principal = Depends(principal),
    a: AuditLogger = Depends(audit),
    c: Container = Depends(container),
) -> dict[str, Any]:
    """Ingest (de-identify) one DICOM and run the modality plugin immediately.

    The response carries findings for programmatic clients (e.g. PACS integration
    tests). The clinician UI never uses this route — it goes through the review flow.
    """
    raw = await file.read()
    try:
        res = c.ingest.ingest_bytes(s, raw, source=f"api:{p.username}")
    except PhiLeakError as exc:
        raise HTTPException(
            status_code=500, detail="de-identification verification failed"
        ) from exc
    if res.status == "rejected":
        raise HTTPException(status_code=422, detail=res.reasons)
    if res.status == "quarantined":
        a("infer_quarantined", "instance", res.instance_id, reasons=res.reasons)
        return {
            "research_use": RESEARCH_USE_LABEL,
            "status": "quarantined",
            "reasons": res.reasons,
            "instance_id": res.instance_id,
            "study_id": res.study_id,
        }
    inst = s.get(Instance, res.instance_id)
    assert inst is not None
    try:
        pred = run_inference_for_instance(c, s, inst, request_id=f"sync:{p.user_id}")
    except NoPluginError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    a("infer_sync", "prediction", pred.id, ingest_status=res.status)
    return {
        "research_use": RESEARCH_USE_LABEL,
        "status": res.status,
        "instance_id": inst.id,
        "study_id": res.study_id,
        "prediction_id": pred.id,
        "model_version_id": pred.model_version_id,
        "gate": {
            "status": pred.gate_status,
            "reasons": pred.gate_reasons,
            "scores": pred.gate_scores,
        },
        "findings": [
            {k: v for k, v in f.items() if not k.endswith("_blob_key")} for f in pred.findings
        ],
        "sr_sop_uid": pred.sr_sop_uid,
        "sc_sop_uids": pred.sc_sop_uids,
        "latency_ms": pred.latency_ms,
        "links": {
            "sr": f"/v1/predictions/{pred.id}/sr.dcm",
            "sc": {
                f["code"]: f"/v1/predictions/{pred.id}/sc/{f['code']}.dcm" for f in pred.findings
            },
        },
    }
