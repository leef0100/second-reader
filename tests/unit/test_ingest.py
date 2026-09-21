"""End-to-end ingestion on synthetic DICOM: de-id → verify → encrypt → index → queue."""

from __future__ import annotations

import pytest
from pydicom.uid import generate_uid
from sqlalchemy import select

from secondreader.core.db import (
    Instance,
    Job,
    Patient,
    Study,
    create_all,
    make_engine,
    make_session_factory,
)
from secondreader.core.dicom import read_dicom
from secondreader.core.ingest import IngestService
from secondreader.core.ingest.service import PhiLeakError
from secondreader.core.storage import EncryptedBlobStore, FilesystemBlobStore
from tests.fixtures.synthetic import FAKE_PHI, cxr_spec, make_bytes, secondary_capture_spec


@pytest.fixture
def session():
    engine = make_engine("sqlite+pysqlite:///:memory:")
    create_all(engine)
    s = make_session_factory(engine)()
    yield s
    s.close()


@pytest.fixture
def inner(tmp_path):
    return FilesystemBlobStore(tmp_path / "blobs")


@pytest.fixture
def service(deidentifier, inner):
    return IngestService(deidentifier, EncryptedBlobStore(inner, b"k" * 32))


def test_ingest_stores_deidentified_encrypted_and_queues(service, session, inner):
    raw = make_bytes(cxr_spec())
    res = service.ingest_bytes(session, raw, source="test")
    session.commit()
    assert res.status == "stored"
    inst = session.get(Instance, res.instance_id)
    assert inst is not None and not inst.quarantined
    assert inst.blob_key.startswith("dicom/")
    # on-disk bytes are ciphertext — no PHI, not even a DICOM preamble
    ondisk = inner.get(inst.blob_key)
    assert b"DICM" not in ondisk and b"DOE" not in ondisk
    # decrypted object is clean and re-readable
    ds = read_dicom(service._blobs.get(inst.blob_key))
    assert str(ds.PatientID) == res.pseudo_key
    assert "PatientBirthDate" not in ds
    assert inst.header["Modality"] == "CR"
    # metadata rows
    study = session.get(Study, res.study_id)
    assert study.patient.pseudo_key == res.pseudo_key
    assert study.study_date != "20240102"  # shifted
    assert study.patient_age_years == 54
    # job queued
    job = session.scalars(select(Job)).one()
    assert job.kind == "infer" and job.payload["instance_id"] == inst.id


def test_duplicate_bytes_are_idempotent(service, session):
    raw = make_bytes(cxr_spec())
    a = service.ingest_bytes(session, raw, source="t")
    b = service.ingest_bytes(session, raw, source="t")
    assert a.status == "stored" and b.status == "duplicate" and b.instance_id == a.instance_id
    assert session.scalars(select(Instance)).all().__len__() == 1


def test_two_studies_same_patient_link_via_pseudo_key(service, session):
    r1 = service.ingest_bytes(
        session, make_bytes(cxr_spec(study_date="20230101", seed=1)), source="t"
    )
    r2 = service.ingest_bytes(
        session, make_bytes(cxr_spec(study_date="20240101", seed=2)), source="t"
    )
    assert r1.patient_id == r2.patient_id
    assert session.scalars(select(Patient)).all().__len__() == 1
    assert session.scalars(select(Study)).all().__len__() == 2


def test_same_study_two_instances_share_study_row(service, session):
    suid, seuid = generate_uid(), generate_uid()
    r1 = service.ingest_bytes(
        session, make_bytes(cxr_spec(study_uid=suid, series_uid=seuid, seed=1)), source="t"
    )
    r2 = service.ingest_bytes(
        session, make_bytes(cxr_spec(study_uid=suid, series_uid=seuid, seed=2)), source="t"
    )
    assert r1.study_id == r2.study_id
    assert r1.instance_id != r2.instance_id


def test_quarantine_path_does_not_enqueue(service, session):
    res = service.ingest_bytes(session, make_bytes(secondary_capture_spec()), source="t")
    assert res.status == "quarantined" and res.job_id is None
    inst = session.get(Instance, res.instance_id)
    assert inst.quarantined and inst.blob_key.startswith("quarantine/")
    assert session.scalars(select(Job)).all() == []


def test_rejects_non_dicom(service, session):
    res = service.ingest_bytes(session, b"not a dicom file at all", source="t")
    assert res.status == "rejected"
    assert session.scalars(select(Instance)).all() == []


def test_fails_closed_if_deidentifier_leaks(service, session, monkeypatch):
    """Sabotage the de-identifier; the verifier must refuse to store."""
    original = service._deid.deidentify

    def leaky(src, **kw):
        res = original(src, **kw)
        res.dataset.PatientBirthDate = FAKE_PHI["PatientBirthDate"]
        return res

    monkeypatch.setattr(service._deid, "deidentify", leaky)
    with pytest.raises(PhiLeakError):
        service.ingest_bytes(session, make_bytes(cxr_spec()), source="t")
    session.rollback()
    assert session.scalars(select(Instance)).all() == []
    assert not service._blobs.exists("dicom")
