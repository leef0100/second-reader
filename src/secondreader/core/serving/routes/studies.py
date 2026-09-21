"""Study browsing. Findings are NEVER returned here — only via the review reveal flow."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from secondreader.core.audit import AuditLogger
from secondreader.core.constants import RESEARCH_USE_LABEL
from secondreader.core.db import priors_for_study
from secondreader.core.db.models import Instance, Prediction, Series, Study
from secondreader.core.dicom import read_dicom
from secondreader.core.dicom.render import frame_to_png
from secondreader.core.serving.deps import Container
from secondreader.core.serving.routes.common import audit, container, db, principal

router = APIRouter(prefix="/v1/studies", tags=["studies"], dependencies=[Depends(principal)])


class StudySummary(BaseModel):
    id: str
    pseudo_key: str
    study_uid: str
    study_date: str | None
    modality: str | None
    description: str | None
    source: str | None
    patient_age_years: int | None
    patient_sex: str | None
    n_instances: int
    n_quarantined: int
    prediction_status: str  # none | pending | ok | unable_to_assess
    n_priors: int
    orthanc_b_id: str | None
    created_at: str


class InstanceSummary(BaseModel):
    id: str
    sop_uid: str
    instance_number: int | None
    rows: int | None
    columns: int | None
    n_frames: int
    quarantined: bool
    quarantine_reasons: list[str]
    view_position: str | None
    manufacturer: str | None


class PredictionStub(BaseModel):
    id: str
    instance_id: str
    gate_status: str
    model_version_id: str
    created_at: str


class StudyDetail(BaseModel):
    research_use: str = RESEARCH_USE_LABEL
    study: StudySummary
    instances: list[InstanceSummary]
    priors: list[StudySummary]
    predictions: list[PredictionStub]  # no findings here (anti-automation-bias)
    ohif_url: str | None


def _status(s: Session, study: Study) -> str:
    preds = list(s.scalars(select(Prediction).where(Prediction.study_id == study.id)))
    if preds:
        return "ok" if all(p.gate_status == "ok" for p in preds) else "unable_to_assess"
    n_ok = s.scalar(
        select(func.count(Instance.id))
        .join(Series)
        .where(Series.study_id == study.id, Instance.quarantined.is_(False))
    )
    return "pending" if n_ok else "none"


def _summary(s: Session, study: Study) -> StudySummary:
    n = (
        s.scalar(select(func.count(Instance.id)).join(Series).where(Series.study_id == study.id))
        or 0
    )
    nq = (
        s.scalar(
            select(func.count(Instance.id))
            .join(Series)
            .where(Series.study_id == study.id, Instance.quarantined.is_(True))
        )
        or 0
    )
    return StudySummary(
        id=study.id,
        pseudo_key=study.patient.pseudo_key,
        study_uid=study.study_uid,
        study_date=study.study_date,
        modality=study.modality,
        description=study.description,
        source=study.source,
        patient_age_years=study.patient_age_years,
        patient_sex=study.patient.sex,
        n_instances=int(n),
        n_quarantined=int(nq),
        prediction_status=_status(s, study),
        n_priors=len(priors_for_study(s, study)),
        orthanc_b_id=study.orthanc_b_id,
        created_at=study.created_at.isoformat(),
    )


@router.get("", response_model=list[StudySummary])
def list_studies(
    s: Session = Depends(db), a: AuditLogger = Depends(audit), limit: int = 100
) -> list[StudySummary]:
    a("studies_listed")
    studies = s.scalars(select(Study).order_by(Study.created_at.desc()).limit(limit)).all()
    return [_summary(s, st) for st in studies]


@router.get("/{study_id}", response_model=StudyDetail)
def get_study(
    study_id: str,
    s: Session = Depends(db),
    a: AuditLogger = Depends(audit),
    c: Container = Depends(container),
) -> StudyDetail:
    study = s.get(Study, study_id)
    if study is None:
        raise HTTPException(status_code=404, detail="study not found")
    a("study_viewed", "study", study.id)
    instances = list(
        s.scalars(
            select(Instance)
            .join(Series)
            .where(Series.study_id == study.id)
            .order_by(Instance.instance_number)
        )
    )
    preds = list(
        s.scalars(
            select(Prediction)
            .where(Prediction.study_id == study.id)
            .order_by(Prediction.created_at.desc())
        )
    )
    base = c.settings.ohif_base_url.rstrip("/")
    return StudyDetail(
        study=_summary(s, study),
        instances=[
            InstanceSummary(
                id=i.id,
                sop_uid=i.sop_uid,
                instance_number=i.instance_number,
                rows=i.rows,
                columns=i.columns,
                n_frames=i.n_frames,
                quarantined=i.quarantined,
                quarantine_reasons=list(i.quarantine_reasons or []),
                view_position=i.series.view_position,
                manufacturer=i.series.manufacturer,
            )
            for i in instances
        ],
        priors=[_summary(s, p) for p in priors_for_study(s, study)],
        predictions=[
            PredictionStub(
                id=p.id,
                instance_id=p.instance_id,
                gate_status=p.gate_status,
                model_version_id=p.model_version_id,
                created_at=p.created_at.isoformat(),
            )
            for p in preds
        ],
        ohif_url=f"{base}/viewer?StudyInstanceUIDs={study.study_uid}"
        if study.orthanc_b_id
        else None,
    )


@router.get("/{study_id}/instances/{instance_id}/preview.png")
def instance_preview(
    study_id: str,
    instance_id: str,
    s: Session = Depends(db),
    a: AuditLogger = Depends(audit),
    c: Container = Depends(container),
) -> Response:
    inst = s.get(Instance, instance_id)
    if inst is None or inst.series.study_id != study_id:
        raise HTTPException(status_code=404, detail="instance not found")
    if inst.quarantined:
        raise HTTPException(status_code=423, detail="instance is quarantined (burned-in text risk)")
    a("image_viewed", "instance", inst.id)
    png = frame_to_png(read_dicom(c.blobs.get(inst.blob_key)))
    return Response(
        content=png,
        media_type="image/png",
        headers={"X-Research-Use": RESEARCH_USE_LABEL, "Cache-Control": "private, max-age=300"},
    )
