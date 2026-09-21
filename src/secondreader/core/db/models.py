"""Metadata schema. Holds de-identified metadata ONLY — never PHI.

Patient → Study → Series → Instance mirrors DICOM. `Patient.pseudo_key` is the
HMAC pseudonym: it is what links a new study to its priors.
"""

from __future__ import annotations

import enum
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _uuid() -> str:
    return str(uuid.uuid4())


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JSON, list[Any]: JSON}


class Patient(Base):
    __tablename__ = "patients"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    pseudo_key: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    weak_identity: Mapped[bool] = mapped_column(Boolean, default=False)
    sex: Mapped[str | None] = mapped_column(String(8))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    studies: Mapped[list[Study]] = relationship(back_populates="patient")


class Study(Base):
    __tablename__ = "studies"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id"), index=True)
    study_uid: Mapped[str] = mapped_column(String(64), unique=True, index=True)  # de-identified
    study_date: Mapped[str | None] = mapped_column(String(8))  # shifted YYYYMMDD
    study_time: Mapped[str | None] = mapped_column(String(16))
    modality: Mapped[str | None] = mapped_column(String(16))
    description: Mapped[str | None] = mapped_column(String(64))  # vocabulary-cleaned only
    source: Mapped[str | None] = mapped_column(String(64))  # e.g. "orthanc-a", "upload", dataset
    patient_age_years: Mapped[int | None] = mapped_column(Integer)
    orthanc_b_id: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    patient: Mapped[Patient] = relationship(back_populates="studies")
    series: Mapped[list[Series]] = relationship(back_populates="study")


class Series(Base):
    __tablename__ = "series"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    study_id: Mapped[str] = mapped_column(ForeignKey("studies.id"), index=True)
    series_uid: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    series_number: Mapped[int | None] = mapped_column(Integer)
    modality: Mapped[str | None] = mapped_column(String(16))
    body_part: Mapped[str | None] = mapped_column(String(32))
    view_position: Mapped[str | None] = mapped_column(String(16))
    manufacturer: Mapped[str | None] = mapped_column(String(64))
    manufacturer_model: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    study: Mapped[Study] = relationship(back_populates="series")
    instances: Mapped[list[Instance]] = relationship(back_populates="series")


class Instance(Base):
    __tablename__ = "instances"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    series_id: Mapped[str] = mapped_column(ForeignKey("series.id"), index=True)
    sop_uid: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    sop_class_uid: Mapped[str] = mapped_column(String(64))
    instance_number: Mapped[int | None] = mapped_column(Integer)
    blob_key: Mapped[str] = mapped_column(String(255))
    sha256: Mapped[str] = mapped_column(String(64), index=True)  # de-identified bytes
    source_sha256: Mapped[str] = mapped_column(String(64), unique=True)  # raw bytes; idempotence
    rows: Mapped[int | None] = mapped_column(Integer)
    columns: Mapped[int | None] = mapped_column(Integer)
    n_frames: Mapped[int] = mapped_column(Integer, default=1)
    photometric: Mapped[str | None] = mapped_column(String(16))
    bits_stored: Mapped[int | None] = mapped_column(Integer)
    quarantined: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    quarantine_reasons: Mapped[list[Any]] = mapped_column(JSON, default=list)
    deid_profile_hash: Mapped[str] = mapped_column(String(64))
    header: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    series: Mapped[Series] = relationship(back_populates="instances")


class JobStatus(enum.StrEnum):
    queued = "queued"
    running = "running"
    done = "done"
    failed = "failed"


class Job(Base):
    __tablename__ = "jobs"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    kind: Mapped[str] = mapped_column(String(32), index=True)  # "infer", "push_results", ...
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    status: Mapped[JobStatus] = mapped_column(Enum(JobStatus), default=JobStatus.queued, index=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, default=3)
    error: Mapped[str | None] = mapped_column(Text)
    worker: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ModelVersion(Base):
    """One row per trained model. `id` is the content hash from the registry."""

    __tablename__ = "model_versions"
    id: Mapped[str] = mapped_column(String(16), primary_key=True)
    modality_id: Mapped[str] = mapped_column(String(32), index=True)
    name: Mapped[str] = mapped_column(String(64))
    weights_sha256: Mapped[str] = mapped_column(String(64))
    config_sha256: Mapped[str] = mapped_column(String(64))
    split_manifest_sha256: Mapped[str | None] = mapped_column(String(64))
    config: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Prediction(Base):
    __tablename__ = "predictions"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    instance_id: Mapped[str] = mapped_column(ForeignKey("instances.id"), index=True)
    study_id: Mapped[str] = mapped_column(ForeignKey("studies.id"), index=True)
    model_version_id: Mapped[str] = mapped_column(ForeignKey("model_versions.id"), index=True)
    plugin_version: Mapped[str] = mapped_column(String(32))
    input_sha256: Mapped[str] = mapped_column(String(64))
    gate_status: Mapped[str] = mapped_column(String(32), index=True)
    gate_reasons: Mapped[list[Any]] = mapped_column(JSON, default=list)
    gate_scores: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    findings: Mapped[list[Any]] = mapped_column(JSON, default=list)  # serialized Finding list
    input_stats: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)  # for drift
    sr_sop_uid: Mapped[str | None] = mapped_column(String(64))
    sc_sop_uids: Mapped[list[Any]] = mapped_column(JSON, default=list)
    sr_blob_key: Mapped[str | None] = mapped_column(String(255))
    sc_blob_keys: Mapped[list[Any]] = mapped_column(JSON, default=list)
    latency_ms: Mapped[float | None] = mapped_column(Float)
    request_id: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True
    )


class Agreement(enum.StrEnum):
    agree = "agree"
    disagree = "disagree"
    unsure = "unsure"


class Feedback(Base):
    __tablename__ = "feedback"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    prediction_id: Mapped[str] = mapped_column(ForeignKey("predictions.id"), index=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    finding_code: Mapped[str] = mapped_column(String(32))
    agreement: Mapped[Agreement] = mapped_column(Enum(Agreement))
    comment: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Role(enum.StrEnum):
    clinician = "clinician"
    admin = "admin"


class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    username: Mapped[str] = mapped_column(String(64), unique=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[Role] = mapped_column(Enum(Role), default=Role.clinician)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ReviewSession(Base):
    """One clinician's read of one study — records the read-first workflow."""

    __tablename__ = "review_sessions"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    study_id: Mapped[str] = mapped_column(ForeignKey("studies.id"), index=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    mode: Mapped[str] = mapped_column(String(16), default="read_first")  # read_first | immediate
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    pre_read_impression: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    revealed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class AuditEvent(Base):
    __tablename__ = "audit_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str | None] = mapped_column(String(36), index=True)
    username: Mapped[str | None] = mapped_column(String(64))
    action: Mapped[str] = mapped_column(String(64), index=True)
    object_type: Mapped[str | None] = mapped_column(String(32))
    object_id: Mapped[str | None] = mapped_column(String(64), index=True)
    details: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    client_ip: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True
    )


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(64))
    version: Mapped[str] = mapped_column(String(32))
    manifest_sha256: Mapped[str] = mapped_column(String(64))
    license: Mapped[str] = mapped_column(String(128))
    commercial_ok: Mapped[bool] = mapped_column(Boolean, default=False)
    n_items: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    __table_args__ = (UniqueConstraint("name", "version"),)


class SplitManifest(Base):
    __tablename__ = "split_manifests"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(64))
    version: Mapped[str] = mapped_column(String(32))
    sha256: Mapped[str] = mapped_column(String(64), unique=True)
    tiers: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)  # tier → {n, sha256, path}
    locked: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    __table_args__ = (UniqueConstraint("name", "version"),)


Index("ix_studies_patient_date", Study.patient_id, Study.study_date)
