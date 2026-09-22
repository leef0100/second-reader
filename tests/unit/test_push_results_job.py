"""Archive push as a retried queue job (D-030): status on the prediction, backoff, exhaustion."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select

from secondreader.core.db import (
    Instance,
    Job,
    JobStatus,
    Prediction,
    Study,
    create_all,
    make_engine,
    make_session_factory,
)
from secondreader.core.dicom import Deidentifier, load_profile
from secondreader.core.ingest import IngestService
from secondreader.core.plugins import PluginRegistry
from secondreader.core.security import LocalAuthProvider
from secondreader.core.serving import (
    handlers,  # noqa: F401  registers job handlers
    queue,
    worker,
)
from secondreader.core.serving.deps import Container
from secondreader.core.serving.inference import (
    PUSH_JOB_KIND,
    PUSH_MAX_ATTEMPTS,
    run_inference_for_instance,
)
from secondreader.core.settings import Settings
from secondreader.core.storage import EncryptedBlobStore, FilesystemBlobStore
from tests.fixtures.synthetic import cxr_spec, make_bytes
from tests.unit.test_api_review import StubPlugin


class FlakyOrthancB:
    """Unreachable for the first `fail_times` reachability checks, then records uploads."""

    def __init__(self, fail_times: int) -> None:
        self.fail_times = fail_times
        self.checks = 0
        self.uploads: list[bytes] = []

    def reachable(self) -> bool:
        self.checks += 1
        return self.checks > self.fail_times

    def upload(self, dicom: bytes) -> dict[str, str]:
        self.uploads.append(dicom)
        return {"ID": f"inst-{len(self.uploads)}", "ParentStudy": "orthanc-b-study-1"}


@pytest.fixture
def container(tmp_path, repo_root):
    settings = Settings(
        pseudonym_secret="test-pseudonym-secret-not-for-production-0000",
        blob_encryption_key="0f" * 32,
        auth_secret="test-auth-secret-not-for-production",
        database_url="sqlite+pysqlite:///:memory:",
        blob_root=tmp_path / "blobs",
        model_registry_root=tmp_path / "models",
        deid_profile=repo_root / "configs/deid/basic_profile.yaml",
    )
    engine = make_engine(settings.database_url)
    create_all(engine)
    blobs = EncryptedBlobStore(FilesystemBlobStore(settings.blob_root), bytes.fromhex("0f" * 32))
    deid = Deidentifier(
        load_profile(settings.deid_profile), b"test-pseudonym-secret-not-for-production-0000"
    )
    c = Container(
        settings=settings,
        engine=engine,
        sessions=make_session_factory(engine),
        blobs=blobs,
        deid=deid,
        ingest=IngestService(deid, blobs),
        orthanc_a=FlakyOrthancB(0),  # type: ignore[arg-type]
        orthanc_b=FlakyOrthancB(0),  # type: ignore[arg-type]
        auth=LocalAuthProvider("test-auth-secret-not-for-production", 60),
    )
    reg = PluginRegistry()
    reg.register(StubPlugin())
    c.set_plugins(reg)
    return c


def _predict(c: Container):
    """Ingest one synthetic CXR, retire the ingest-queued infer job, run inference."""
    with c.sessions() as s:
        res = c.ingest.ingest_bytes(s, make_bytes(cxr_spec(rows=96, cols=96)), source="test")
        for j in s.scalars(select(Job).where(Job.kind == "infer")):
            j.status = JobStatus.done
        inst = s.get(Instance, res.instance_id)
        pred = run_inference_for_instance(c, s, inst)
        s.commit()
        return pred.id


def _push_jobs(s):
    return list(s.scalars(select(Job).where(Job.kind == PUSH_JOB_KIND)))


def test_inference_enqueues_push_job_and_records_status(container):
    c = container
    pid = _predict(c)
    with c.sessions() as s:
        jobs = _push_jobs(s)
        assert len(jobs) == 1 and jobs[0].max_attempts == PUSH_MAX_ATTEMPTS == 5
        assert jobs[0].payload == {"prediction_id": pid}
        pred = s.get(Prediction, pid)
        assert pred.push_status == {"state": "queued", "attempts": 0, "job_id": jobs[0].id}
        # nothing was pushed inline (the old fire-and-forget path is gone)
        assert c.orthanc_b.uploads == []


def test_push_job_pushes_image_sr_and_overlays_and_sets_study_archive_id(container):
    c = container
    pid = _predict(c)
    assert worker.process_one(c, "w1") is True
    with c.sessions() as s:
        pred = s.get(Prediction, pid)
        assert pred.push_status["state"] == "pushed" and pred.push_status["attempts"] == 1
        assert pred.push_status["n_objects"] == 1 + 1 + len(pred.sc_blob_keys)
        assert pred.push_status["orthanc_study_id"] == "orthanc-b-study-1"
        assert s.get(Study, pred.study_id).orthanc_b_id == "orthanc-b-study-1"
        job = _push_jobs(s)[0]
        assert job.status == JobStatus.done and job.attempts == 1
    # image + SR + 2 SC overlays, all de-identified DICOM
    assert len(c.orthanc_b.uploads) == 4
    assert all(b[128:132] == b"DICM" for b in c.orthanc_b.uploads)
    assert not any(b"DOE^JANE" in b for b in c.orthanc_b.uploads)


def test_push_job_retries_with_backoff_then_succeeds(container):
    c = container
    c.orthanc_b = FlakyOrthancB(fail_times=1)  # attempt 2 below is simulated at queue level
    pid = _predict(c)
    # attempt 1 fails: re-queued with a 5 s backoff, status says retrying
    assert worker.process_one(c, "w1") is True
    with c.sessions() as s:
        job = _push_jobs(s)[0]
        assert job.status == JobStatus.queued and job.attempts == 1
        assert "ArchiveUnreachableError" in (job.error or "")
        assert job.not_before is not None
        st = s.get(Prediction, pid).push_status
        assert st["state"] == "retrying" and st["attempts"] == 1 and st["retry_in_s"] == 5.0
        # not claimable before the backoff elapses ...
        now = datetime.now(UTC)
        assert queue.claim_next(s, "w2", now=now - timedelta(seconds=1)) is None
        # ... but claimable afterwards
        job = queue.claim_next(s, "w2", now=now + timedelta(minutes=10))
        assert job is not None and job.attempts == 2 and job.not_before is None
        queue.fail(s, job, "still down")  # attempt 2 → 10 s backoff
        assert job.not_before is not None and job.status == JobStatus.queued
        assert (job.not_before - job.finished_at).total_seconds() == pytest.approx(10.0)
        job.not_before = None  # make it due now for the final, successful attempt
        s.commit()
    assert worker.process_one(c, "w1") is True
    with c.sessions() as s:
        st = s.get(Prediction, pid).push_status
        assert st["state"] == "pushed" and st["attempts"] == 3
        assert _push_jobs(s)[0].status == JobStatus.done
    assert len(c.orthanc_b.uploads) == 4


def test_push_job_gives_up_after_max_attempts(container):
    c = container
    c.orthanc_b = FlakyOrthancB(fail_times=99)
    pid = _predict(c)
    for attempt in range(1, PUSH_MAX_ATTEMPTS + 1):
        with c.sessions() as s:
            j = _push_jobs(s)[0]
            j.not_before = None  # skip the backoff wait in the test
            s.commit()
        assert worker.process_one(c, "w1") is True
        with c.sessions() as s:
            j = _push_jobs(s)[0]
            st = s.get(Prediction, pid).push_status
            assert j.attempts == attempt and st["attempts"] == attempt
            if attempt < PUSH_MAX_ATTEMPTS:
                assert j.status == JobStatus.queued and st["state"] == "retrying"
            else:
                assert j.status == JobStatus.failed and st["state"] == "failed"
                assert st["retry_in_s"] is None and "ArchiveUnreachableError" in st["error"]
    assert worker.process_one(c, "w1") is False  # nothing left to claim
    assert c.orthanc_b.uploads == []


def test_retry_delay_policy():
    assert [queue.retry_delay(PUSH_JOB_KIND, a) for a in range(1, 6)] == [5, 10, 20, 40, 80]
    assert queue.retry_delay(PUSH_JOB_KIND, 20) == 300.0  # capped
    assert queue.retry_delay("infer", 3) == 0.0  # other kinds retry immediately (D-004)
