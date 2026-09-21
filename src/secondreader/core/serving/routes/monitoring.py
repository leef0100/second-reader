from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from secondreader.core.audit import AuditLogger
from secondreader.core.db.models import AuditEvent, ModelVersion, Prediction
from secondreader.core.monitoring import compute_drift, render_drift_markdown
from secondreader.core.security import Principal
from secondreader.core.serving.deps import Container
from secondreader.core.serving.routes.common import audit, container, db, principal

router = APIRouter(prefix="/v1/monitoring", tags=["monitoring"])


def _reference(c: Container, model_version_id: str) -> dict[str, Any] | None:
    for plugin in c.plugins().all():
        if plugin.model_version_id == model_version_id:
            card = getattr(getattr(plugin, "bundle", None), "card", {}) or {}
            return dict(card.get("reference") or {}) or None
    return None


def _model_id(s: Session, c: Container, model_version_id: str | None) -> str:
    if model_version_id:
        return model_version_id
    plugins = c.plugins().all()
    if plugins:
        return plugins[0].model_version_id
    mv = s.scalars(select(ModelVersion).order_by(ModelVersion.created_at.desc())).first()
    if mv is None:
        raise HTTPException(status_code=404, detail="no model version known")
    return mv.id


@router.get("/drift")
def drift(
    days: int = 7,
    model_version_id: str | None = None,
    s: Session = Depends(db),
    p: Principal = Depends(principal),
    a: AuditLogger = Depends(audit),
    c: Container = Depends(container),
) -> dict[str, Any]:
    mid = _model_id(s, c, model_version_id)
    a("drift_viewed", "model_version", mid, days=days)
    return compute_drift(s, mid, _reference(c, mid), days=days).as_dict()


@router.get("/drift.md")
def drift_md(
    days: int = 7,
    model_version_id: str | None = None,
    s: Session = Depends(db),
    p: Principal = Depends(principal),
    a: AuditLogger = Depends(audit),
    c: Container = Depends(container),
) -> Response:
    mid = _model_id(s, c, model_version_id)
    a("drift_viewed", "model_version", mid, days=days, format="md")
    md = render_drift_markdown(compute_drift(s, mid, _reference(c, mid), days=days))
    return Response(content=md, media_type="text/markdown")


@router.get("/models")
def models(s: Session = Depends(db), p: Principal = Depends(principal)) -> list[dict[str, Any]]:
    rows = s.scalars(select(ModelVersion).order_by(ModelVersion.created_at.desc())).all()
    counts: dict[str, int] = {
        str(mid): int(n)
        for mid, n in s.execute(
            select(Prediction.model_version_id, func.count()).group_by(Prediction.model_version_id)
        ).all()
    }
    return [
        {
            "id": m.id,
            "modality_id": m.modality_id,
            "name": m.name,
            "weights_sha256": m.weights_sha256,
            "config_sha256": m.config_sha256,
            "split_manifest_sha256": m.split_manifest_sha256,
            "config": m.config,
            "val_metrics": m.metrics,
            "n_predictions": int(counts.get(m.id, 0)),
            "created_at": m.created_at.isoformat(),
        }
        for m in rows
    ]


@router.get("/audit")
def audit_log(
    limit: int = 200, s: Session = Depends(db), p: Principal = Depends(principal)
) -> list[dict[str, Any]]:
    if p.role.value != "admin":
        raise HTTPException(status_code=403, detail="admin only")
    rows = s.scalars(select(AuditEvent).order_by(AuditEvent.id.desc()).limit(limit)).all()
    return [
        {
            "id": e.id,
            "user": e.username,
            "action": e.action,
            "object_type": e.object_type,
            "object_id": e.object_id,
            "details": e.details,
            "ip": e.client_ip,
            "at": e.created_at.isoformat(),
        }
        for e in rows
    ]
