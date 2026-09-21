"""Prior-study linkage queries (principle 4)."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from secondreader.core.db.models import Patient, Study


def priors_for_study(session: Session, study: Study, *, limit: int = 10) -> list[Study]:
    """Earlier studies of the same pseudonymized patient, newest first."""
    stmt = (
        select(Study)
        .where(Study.patient_id == study.patient_id, Study.id != study.id)
        .where((Study.study_date < study.study_date) | (Study.study_date.is_(None)))
        .order_by(Study.study_date.desc(), Study.created_at.desc())
        .limit(limit)
    )
    return list(session.scalars(stmt))


def studies_for_pseudo_key(session: Session, pseudo_key: str) -> list[Study]:
    stmt = (
        select(Study)
        .join(Patient)
        .where(Patient.pseudo_key == pseudo_key)
        .order_by(Study.study_date.asc())
    )
    return list(session.scalars(stmt))
