"""Read-first-then-reveal review workflow (principle 3) and agree/disagree capture (principle 8).

Findings are only ever served after `reveal`. In `read_first` mode reveal requires a
recorded pre-read impression; every reveal and every view is audited.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from secondreader.core.audit import AuditLogger
from secondreader.core.constants import RESEARCH_USE_LABEL
from secondreader.core.db.models import Agreement, Feedback, Prediction, ReviewSession, Study
from secondreader.core.security import Principal
from secondreader.core.serving.deps import Container
from secondreader.core.serving.routes.common import audit, container, db, principal

router = APIRouter(prefix="/v1/review", tags=["review"])


class SessionCreate(BaseModel):
    study_id: str
    mode: Literal["read_first", "immediate"] = "read_first"


class SessionOut(BaseModel):
    id: str
    study_id: str
    mode: str
    started_at: str
    revealed_at: str | None
    completed_at: str | None
    pre_read_impression: dict[str, Any] | None
    research_use: str = RESEARCH_USE_LABEL


class Impression(BaseModel):
    findings: dict[str, bool] = Field(default_factory=dict)  # code → clinician saw it
    notes: str = ""


class FindingOut(BaseModel):
    code: str
    display_name: str
    probability: float
    threshold: float
    present: bool
    boxes: list[dict[str, float]]
    heatmap_url: str | None
    overlay_url: str | None
    snomed_code: str | None


class PredictionOut(BaseModel):
    id: str
    instance_id: str
    model_version_id: str
    plugin_version: str
    gate_status: str
    gate_reasons: list[str]
    findings: list[FindingOut]
    sr_url: str
    created_at: str


class RevealOut(BaseModel):
    research_use: str = RESEARCH_USE_LABEL
    session: SessionOut
    predictions: list[PredictionOut]
    note: str = (
        "AI output is a second opinion for a clinician. Every finding shows its calibrated "
        "probability and a heatmap; the clinician's read is recorded before reveal."
    )


class FeedbackIn(BaseModel):
    prediction_id: str
    finding_code: str
    agreement: Literal["agree", "disagree", "unsure"]
    comment: str | None = None


def _out(rs: ReviewSession) -> SessionOut:
    return SessionOut(
        id=rs.id,
        study_id=rs.study_id,
        mode=rs.mode,
        started_at=rs.started_at.isoformat(),
        revealed_at=rs.revealed_at.isoformat() if rs.revealed_at else None,
        completed_at=rs.completed_at.isoformat() if rs.completed_at else None,
        pre_read_impression=rs.pre_read_impression,
    )


def _own_session(s: Session, session_id: str, p: Principal) -> ReviewSession:
    rs = s.get(ReviewSession, session_id)
    if rs is None or rs.user_id != p.user_id:
        raise HTTPException(status_code=404, detail="review session not found")
    return rs


def prediction_out(pred: Prediction) -> PredictionOut:
    return PredictionOut(
        id=pred.id,
        instance_id=pred.instance_id,
        model_version_id=pred.model_version_id,
        plugin_version=pred.plugin_version,
        gate_status=pred.gate_status,
        gate_reasons=list(pred.gate_reasons or []),
        findings=[
            FindingOut(
                code=f["code"],
                display_name=f["display_name"],
                probability=f["probability"],
                threshold=f["threshold"],
                present=f["present"],
                boxes=f.get("boxes", []),
                heatmap_url=f"/v1/predictions/{pred.id}/heatmap/{f['code']}.png"
                if f.get("heatmap_blob_key")
                else None,
                overlay_url=f"/v1/predictions/{pred.id}/overlay/{f['code']}.png"
                if f.get("sc_blob_key")
                else None,
                snomed_code=f.get("snomed_code"),
            )
            for f in (pred.findings or [])
        ],
        sr_url=f"/v1/predictions/{pred.id}/sr.dcm",
        created_at=pred.created_at.isoformat(),
    )


@router.post("/sessions", response_model=SessionOut)
def create_session(
    body: SessionCreate,
    s: Session = Depends(db),
    p: Principal = Depends(principal),
    a: AuditLogger = Depends(audit),
) -> SessionOut:
    if s.get(Study, body.study_id) is None:
        raise HTTPException(status_code=404, detail="study not found")
    rs = ReviewSession(study_id=body.study_id, user_id=p.user_id, mode=body.mode)
    s.add(rs)
    s.flush()
    a("review_started", "study", body.study_id, session_id=rs.id, mode=body.mode)
    return _out(rs)


@router.get("/sessions/{session_id}", response_model=SessionOut)
def get_session(
    session_id: str, s: Session = Depends(db), p: Principal = Depends(principal)
) -> SessionOut:
    return _out(_own_session(s, session_id, p))


@router.post("/sessions/{session_id}/impression", response_model=SessionOut)
def record_impression(
    session_id: str,
    body: Impression,
    s: Session = Depends(db),
    p: Principal = Depends(principal),
    a: AuditLogger = Depends(audit),
) -> SessionOut:
    rs = _own_session(s, session_id, p)
    if rs.revealed_at is not None:
        raise HTTPException(status_code=409, detail="AI already revealed; impression is locked")
    rs.pre_read_impression = {
        "findings": body.findings,
        "notes": body.notes,
        "recorded_at": datetime.now(UTC).isoformat(),
    }
    s.flush()
    a("pre_read_recorded", "review_session", rs.id, n_findings=sum(body.findings.values()))
    return _out(rs)


@router.post("/sessions/{session_id}/reveal", response_model=RevealOut)
def reveal(
    session_id: str,
    s: Session = Depends(db),
    p: Principal = Depends(principal),
    a: AuditLogger = Depends(audit),
) -> RevealOut:
    rs = _own_session(s, session_id, p)
    if rs.mode == "read_first" and rs.pre_read_impression is None and rs.revealed_at is None:
        raise HTTPException(
            status_code=409, detail="record your own read before revealing AI output"
        )
    if rs.revealed_at is None:
        rs.revealed_at = datetime.now(UTC)
        s.flush()
        a("ai_revealed", "review_session", rs.id, study_id=rs.study_id)
    preds = list(
        s.scalars(
            select(Prediction)
            .where(Prediction.study_id == rs.study_id)
            .order_by(Prediction.created_at.desc())
        )
    )
    return RevealOut(session=_out(rs), predictions=[prediction_out(x) for x in preds])


@router.post("/sessions/{session_id}/feedback", response_model=dict[str, str])
def feedback(
    session_id: str,
    body: FeedbackIn,
    s: Session = Depends(db),
    p: Principal = Depends(principal),
    a: AuditLogger = Depends(audit),
) -> dict[str, str]:
    rs = _own_session(s, session_id, p)
    if rs.revealed_at is None:
        raise HTTPException(status_code=409, detail="cannot give feedback before reveal")
    pred = s.get(Prediction, body.prediction_id)
    if pred is None or pred.study_id != rs.study_id:
        raise HTTPException(status_code=404, detail="prediction not found in this study")
    if (
        body.finding_code not in {f["code"] for f in (pred.findings or [])}
        and pred.gate_status == "ok"
    ):
        raise HTTPException(status_code=400, detail="unknown finding code")
    fb = Feedback(
        prediction_id=pred.id,
        user_id=p.user_id,
        finding_code=body.finding_code,
        agreement=Agreement(body.agreement),
        comment=body.comment,
    )
    s.add(fb)
    s.flush()
    a(
        "feedback_recorded",
        "prediction",
        pred.id,
        finding=body.finding_code,
        agreement=body.agreement,
    )
    return {"id": fb.id, "status": "recorded"}


@router.post("/sessions/{session_id}/complete", response_model=SessionOut)
def complete(
    session_id: str,
    s: Session = Depends(db),
    p: Principal = Depends(principal),
    a: AuditLogger = Depends(audit),
) -> SessionOut:
    rs = _own_session(s, session_id, p)
    rs.completed_at = datetime.now(UTC)
    s.flush()
    a("review_completed", "review_session", rs.id)
    return _out(rs)


def _revealed_for(s: Session, p: Principal, study_id: str) -> bool:
    return (
        s.scalars(
            select(ReviewSession).where(
                ReviewSession.study_id == study_id,
                ReviewSession.user_id == p.user_id,
                ReviewSession.revealed_at.is_not(None),
            )
        ).first()
        is not None
    )


artifacts = APIRouter(prefix="/v1/predictions", tags=["predictions"])


def _pred_for_download(s: Session, p: Principal, prediction_id: str) -> Prediction:
    pred = s.get(Prediction, prediction_id)
    if pred is None:
        raise HTTPException(status_code=404, detail="prediction not found")
    if not _revealed_for(s, p, pred.study_id):
        raise HTTPException(status_code=403, detail="reveal AI output in a review session first")
    return pred


@artifacts.get("/{prediction_id}", response_model=PredictionOut)
def get_prediction(
    prediction_id: str,
    s: Session = Depends(db),
    p: Principal = Depends(principal),
    a: AuditLogger = Depends(audit),
) -> PredictionOut:
    pred = _pred_for_download(s, p, prediction_id)
    a("prediction_viewed", "prediction", pred.id)
    return prediction_out(pred)


@artifacts.get("/{prediction_id}/heatmap/{code}.png")
def heatmap(
    prediction_id: str,
    code: str,
    s: Session = Depends(db),
    p: Principal = Depends(principal),
    a: AuditLogger = Depends(audit),
    c: Container = Depends(container),
) -> Response:
    pred = _pred_for_download(s, p, prediction_id)
    key = next((f.get("heatmap_blob_key") for f in pred.findings if f["code"] == code), None)
    if not key:
        raise HTTPException(status_code=404, detail="no heatmap")
    a("heatmap_viewed", "prediction", pred.id, finding=code)
    return Response(content=c.blobs.get(key), media_type="image/png")


@artifacts.get("/{prediction_id}/overlay/{code}.png")
def overlay(
    prediction_id: str,
    code: str,
    s: Session = Depends(db),
    p: Principal = Depends(principal),
    a: AuditLogger = Depends(audit),
    c: Container = Depends(container),
) -> Response:
    """The Secondary Capture overlay rendered to PNG (same pixels as the DICOM SC)."""
    from secondreader.core.dicom import read_dicom
    from secondreader.core.dicom.render import frame_to_png

    pred = _pred_for_download(s, p, prediction_id)
    key = next((f.get("sc_blob_key") for f in pred.findings if f["code"] == code), None)
    if not key:
        raise HTTPException(status_code=404, detail="no overlay")
    a("overlay_viewed", "prediction", pred.id, finding=code)
    ds = read_dicom(c.blobs.get(key))
    from io import BytesIO

    from PIL import Image

    im = Image.fromarray(ds.pixel_array)
    buf = BytesIO()
    im.save(buf, format="PNG")
    _ = frame_to_png  # RGB SC does not need windowing
    return Response(content=buf.getvalue(), media_type="image/png")


@artifacts.get("/{prediction_id}/sr.dcm")
def sr_download(
    prediction_id: str,
    s: Session = Depends(db),
    p: Principal = Depends(principal),
    a: AuditLogger = Depends(audit),
    c: Container = Depends(container),
) -> Response:
    pred = _pred_for_download(s, p, prediction_id)
    if not pred.sr_blob_key:
        raise HTTPException(status_code=404, detail="no SR")
    a("sr_downloaded", "prediction", pred.id)
    return Response(
        content=c.blobs.get(pred.sr_blob_key),
        media_type="application/dicom",
        headers={"Content-Disposition": f'attachment; filename="{pred.id}-sr.dcm"'},
    )


@artifacts.get("/{prediction_id}/sc/{code}.dcm")
def sc_download(
    prediction_id: str,
    code: str,
    s: Session = Depends(db),
    p: Principal = Depends(principal),
    a: AuditLogger = Depends(audit),
    c: Container = Depends(container),
) -> Response:
    pred = _pred_for_download(s, p, prediction_id)
    key = next((f.get("sc_blob_key") for f in pred.findings if f["code"] == code), None)
    if not key:
        raise HTTPException(status_code=404, detail="no SC")
    a("sc_downloaded", "prediction", pred.id, finding=code)
    return Response(
        content=c.blobs.get(key),
        media_type="application/dicom",
        headers={"Content-Disposition": f'attachment; filename="{pred.id}-{code}.dcm"'},
    )
