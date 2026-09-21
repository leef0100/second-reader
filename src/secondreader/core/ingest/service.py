"""Ingestion: raw DICOM bytes → de-identified, verified, encrypted, indexed, queued.

This is the only path by which imaging enters platform storage. Order matters:
verification (`find_phi`) runs on the *serialized* output right before the
blob is written, so a bug anywhere upstream fails closed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

import structlog
from sqlalchemy import select
from sqlalchemy.orm import Session

from secondreader.core.db.models import Instance, Patient, Series, Study
from secondreader.core.dicom import (
    Deidentifier,
    dataset_to_bytes,
    find_phi,
    header_dict,
    read_dicom,
    sha256_bytes,
)
from secondreader.core.dicom.pseudonym import parse_age_years
from secondreader.core.serving.queue import enqueue
from secondreader.core.storage import BlobStore

log = structlog.get_logger(__name__)

IngestStatus = Literal["stored", "quarantined", "duplicate", "rejected"]


class PhiLeakError(RuntimeError):
    """Raised when post-de-identification verification finds a problem. Fails closed."""


@dataclass
class IngestResult:
    status: IngestStatus
    instance_id: str | None = None
    study_id: str | None = None
    patient_id: str | None = None
    pseudo_key: str | None = None
    job_id: str | None = None
    reasons: list[str] = field(default_factory=list)
    elapsed_ms: float = 0.0


class IngestService:
    def __init__(
        self,
        deidentifier: Deidentifier,
        blobs: BlobStore,
        *,
        quarantine_prefix: str = "quarantine",
        archive_prefix: str = "dicom",
        enqueue_inference: bool = True,
    ) -> None:
        self._deid = deidentifier
        self._blobs = blobs
        self._qprefix = quarantine_prefix
        self._aprefix = archive_prefix
        self._enqueue = enqueue_inference

    def ingest_bytes(self, session: Session, raw: bytes, *, source: str) -> IngestResult:
        t0 = time.perf_counter()
        source_sha = sha256_bytes(raw)

        existing = session.scalars(
            select(Instance).where(Instance.source_sha256 == source_sha)
        ).first()
        if existing is not None:
            return IngestResult(
                status="duplicate",
                instance_id=existing.id,
                study_id=existing.series.study_id,
                reasons=["identical bytes already ingested"],
                elapsed_ms=_ms(t0),
            )

        try:
            src = read_dicom(raw)
        except Exception as exc:  # noqa: BLE001
            return IngestResult(
                status="rejected", reasons=[f"not DICOM: {exc}"], elapsed_ms=_ms(t0)
            )
        if "SOPInstanceUID" not in src or "SOPClassUID" not in src:
            return IngestResult(status="rejected", reasons=["missing SOP UIDs"], elapsed_ms=_ms(t0))

        originals = _original_values(src)
        original_uids = [
            str(src.get(k, "")) for k in ("StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID")
        ]
        try:
            res = self._deid.deidentify(src)
        except ValueError as exc:
            return IngestResult(status="rejected", reasons=[str(exc)], elapsed_ms=_ms(t0))
        ds, report = res.dataset, res.report

        out = dataset_to_bytes(ds)
        check = read_dicom(out)
        problems = find_phi(
            check, self._deid.profile, known_values=originals, known_uids=original_uids
        )
        if problems:
            log.error("phi_verification_failed", problems=problems, source=source)
            raise PhiLeakError("; ".join(problems))

        quarantined = report.quarantine
        prefix = self._qprefix if quarantined else self._aprefix
        key = f"{prefix}/{ds.StudyInstanceUID}/{ds.SOPInstanceUID}.dcm"
        ref = self._blobs.put(key, out)

        patient = _upsert_patient(session, report.pseudo_key, report.weak_identity, ds)
        study = _upsert_study(session, patient, ds, source)
        series = _upsert_series(session, study, ds)
        inst = Instance(
            series_id=series.id,
            sop_uid=str(ds.SOPInstanceUID),
            sop_class_uid=str(ds.SOPClassUID),
            instance_number=_int_or_none(ds.get("InstanceNumber")),
            blob_key=key,
            sha256=ref.sha256,
            source_sha256=source_sha,
            rows=_int_or_none(ds.get("Rows")),
            columns=_int_or_none(ds.get("Columns")),
            n_frames=int(ds.get("NumberOfFrames", 1) or 1),
            photometric=_str_or_none(ds.get("PhotometricInterpretation")),
            bits_stored=_int_or_none(ds.get("BitsStored")),
            quarantined=quarantined,
            quarantine_reasons=list(report.quarantine_reasons),
            deid_profile_hash=report.profile_hash,
            header=header_dict(ds),
        )
        session.add(inst)
        session.flush()

        job_id: str | None = None
        if self._enqueue and not quarantined:
            job = enqueue(session, "infer", {"instance_id": inst.id, "study_id": study.id})
            job_id = job.id

        log.info(
            "ingested",
            status="quarantined" if quarantined else "stored",
            instance_id=inst.id,
            study_id=study.id,
            source=source,
            removed=len(report.removed),
            private_removed=report.private_tags_removed,
        )
        return IngestResult(
            status="quarantined" if quarantined else "stored",
            instance_id=inst.id,
            study_id=study.id,
            patient_id=patient.id,
            pseudo_key=report.pseudo_key,
            job_id=job_id,
            reasons=list(report.quarantine_reasons),
            elapsed_ms=_ms(t0),
        )


# ---- helpers ------------------------------------------------------------------------------


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 2)


def _original_values(src: Any) -> list[str]:
    keys = (
        "PatientName",
        "PatientID",
        "PatientBirthDate",
        "OtherPatientIDs",
        "PatientAddress",
        "PatientTelephoneNumbers",
        "InstitutionName",
        "StationName",
        "DeviceSerialNumber",
        "ReferringPhysicianName",
        "PerformingPhysicianName",
        "OperatorsName",
        "AccessionNumber",
    )
    vals: list[str] = []
    for k in keys:
        v = src.get(k)
        if v is not None and str(v).strip():
            vals.append(str(v))
    return vals


def _int_or_none(v: Any) -> int | None:
    try:
        return int(v) if v is not None and str(v).strip() != "" else None
    except (TypeError, ValueError):
        return None


def _str_or_none(v: Any) -> str | None:
    s = str(v).strip() if v is not None else ""
    return s or None


def _upsert_patient(session: Session, pseudo_key: str, weak: bool, ds: Any) -> Patient:
    p = session.scalars(select(Patient).where(Patient.pseudo_key == pseudo_key)).first()
    if p is None:
        p = Patient(
            pseudo_key=pseudo_key, weak_identity=weak, sex=_str_or_none(ds.get("PatientSex"))
        )
        session.add(p)
        session.flush()
    return p


def _upsert_study(session: Session, patient: Patient, ds: Any, source: str) -> Study:
    uid = str(ds.StudyInstanceUID)
    s = session.scalars(select(Study).where(Study.study_uid == uid)).first()
    if s is None:
        age = _str_or_none(ds.get("PatientAge"))
        s = Study(
            patient_id=patient.id,
            study_uid=uid,
            study_date=_str_or_none(ds.get("StudyDate")),
            study_time=_str_or_none(ds.get("StudyTime")),
            modality=_str_or_none(ds.get("Modality")),
            description=_str_or_none(ds.get("StudyDescription")),
            source=source,
            patient_age_years=parse_age_years(age) if age else None,
        )
        session.add(s)
        session.flush()
    elif s.patient_id != patient.id:
        raise PhiLeakError("study UID already linked to a different pseudonymized patient")
    return s


def _upsert_series(session: Session, study: Study, ds: Any) -> Series:
    uid = str(ds.SeriesInstanceUID)
    se = session.scalars(select(Series).where(Series.series_uid == uid)).first()
    if se is None:
        se = Series(
            study_id=study.id,
            series_uid=uid,
            series_number=_int_or_none(ds.get("SeriesNumber")),
            modality=_str_or_none(ds.get("Modality")),
            body_part=_str_or_none(ds.get("BodyPartExamined")),
            view_position=_str_or_none(ds.get("ViewPosition")),
            manufacturer=_str_or_none(ds.get("Manufacturer")),
            manufacturer_model=_str_or_none(ds.get("ManufacturerModelName")),
        )
        session.add(se)
        session.flush()
    return se
