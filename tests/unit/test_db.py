from __future__ import annotations

import pytest
from sqlalchemy import select

from secondreader.core.db import (
    Instance,
    Job,
    JobStatus,
    Patient,
    Series,
    Study,
    create_all,
    make_engine,
    make_session_factory,
    priors_for_study,
    studies_for_pseudo_key,
)
from secondreader.core.serving import queue


@pytest.fixture
def session():
    engine = make_engine("sqlite+pysqlite:///:memory:")
    create_all(engine)
    factory = make_session_factory(engine)
    s = factory()
    yield s
    s.close()


def _study(session, patient, uid, date):
    st = Study(patient_id=patient.id, study_uid=uid, study_date=date, modality="CR")
    session.add(st)
    session.flush()
    return st


def test_patient_study_series_instance_linkage(session):
    p = Patient(pseudo_key="ab" * 16, sex="F")
    session.add(p)
    session.flush()
    st = _study(session, p, "2.25.1", "20240101")
    se = Series(study_id=st.id, series_uid="2.25.2", modality="CR")
    session.add(se)
    session.flush()
    inst = Instance(
        series_id=se.id,
        sop_uid="2.25.3",
        sop_class_uid="1.2",
        blob_key="k",
        sha256="0" * 64,
        source_sha256="1" * 64,
        deid_profile_hash="h",
    )
    session.add(inst)
    session.commit()
    got = session.scalars(select(Instance)).one()
    assert got.series.study.patient.pseudo_key == "ab" * 16


def test_priors_are_same_patient_earlier_only(session):
    p1 = Patient(pseudo_key="a" * 32)
    p2 = Patient(pseudo_key="b" * 32)
    session.add_all([p1, p2])
    session.flush()
    s_old = _study(session, p1, "2.25.10", "20230101")
    s_mid = _study(session, p1, "2.25.11", "20230601")
    s_new = _study(session, p1, "2.25.12", "20240101")
    _study(session, p2, "2.25.20", "20230301")  # other patient — never a prior
    session.commit()
    priors = priors_for_study(session, s_new)
    assert [s.study_uid for s in priors] == ["2.25.11", "2.25.10"]
    assert priors_for_study(session, s_old) == []
    assert [s.study_uid for s in studies_for_pseudo_key(session, "a" * 32)] == [
        "2.25.10",
        "2.25.11",
        "2.25.12",
    ]
    assert s_mid in priors


def test_job_queue_claim_complete_retry(session):
    j = queue.enqueue(session, "infer", {"instance_id": "x"}, max_attempts=2)
    session.commit()
    claimed = queue.claim_next(session, "w1", kinds=("infer",))
    assert claimed is not None and claimed.id == j.id
    assert claimed.status == JobStatus.running and claimed.attempts == 1
    assert queue.claim_next(session, "w2") is None  # nothing else queued
    queue.fail(session, claimed, "boom")
    assert claimed.status == JobStatus.queued  # retried
    again = queue.claim_next(session, "w1")
    assert again is not None
    queue.fail(session, again, "boom again")
    assert again.status == JobStatus.failed  # exhausted
    j2 = queue.enqueue(session, "infer", {})
    c = queue.claim_next(session, "w1")
    assert c is not None and c.id == j2.id
    queue.complete(session, c)
    assert session.get(Job, j2.id).status == JobStatus.done
