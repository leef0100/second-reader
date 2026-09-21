"""Live integration: DICOM → Orthanc A → API webhook → de-identified archive.

Skips unless Orthanc A and the API are reachable (see docker-compose.yml).
"""

from __future__ import annotations

import os
import time

import httpx
import pytest

from secondreader.core.pacs import OrthancClient
from tests.fixtures.synthetic import FAKE_PHI, cxr_spec, make_bytes

pytestmark = pytest.mark.integration

ORTHANC_A = os.environ.get("SR_ORTHANC_A_URL", "http://localhost:8042")
API = os.environ.get("SR_API_URL", "http://localhost:8000")
TOKEN = os.environ.get("SR_WEBHOOK_TOKEN", "dev-webhook-token")


@pytest.fixture(scope="module")
def orthanc_a():
    c = OrthancClient(ORTHANC_A, "orthanc", os.environ.get("ORTHANC_A_PASSWORD", "orthanc"))
    if not c.reachable():
        pytest.skip("Orthanc A not reachable")
    return c


@pytest.fixture(scope="module")
def api():
    try:
        r = httpx.get(f"{API}/health", timeout=5)
    except httpx.HTTPError:
        pytest.skip("API not reachable")
    assert r.status_code == 200
    assert "research use only" in r.headers["X-Research-Use"].lower()
    return httpx.Client(base_url=API, timeout=60)


def test_pacs_roundtrip_deidentifies(orthanc_a, api):
    raw = make_bytes(cxr_spec(seed=int(time.time()) % 1000))
    up = orthanc_a.upload(raw)
    study_id = up["ParentStudy"]
    try:
        # Simulate the Orthanc plugin's stable-study webhook.
        r = api.post(
            "/v1/ingest/orthanc",
            json={"orthanc_study_id": study_id, "source": "orthanc-a"},
            headers={"X-Webhook-Token": TOKEN},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["stored"] == 1 and body["quarantined"] == 0
        assert body["job_ids"], "an inference job should be queued"
        assert "research use only" in body["research_use"].lower()

        # Wrong token is rejected.
        bad = api.post("/v1/ingest/orthanc", json={"orthanc_study_id": study_id},
                       headers={"X-Webhook-Token": "nope"})
        assert bad.status_code == 401

        # Source PACS still holds the identifiable study (we never modify it).
        st = orthanc_a.study(study_id)
        assert st["PatientMainDicomTags"]["PatientName"] == FAKE_PHI["PatientName"]
    finally:
        orthanc_a.delete_study(study_id)


def test_upload_endpoint_quarantines_secondary_capture(api):
    from tests.fixtures.synthetic import secondary_capture_spec

    files = [("files", ("sc.dcm", make_bytes(secondary_capture_spec(seed=7)), "application/dicom"))]
    r = api.post("/v1/ingest/upload", files=files)
    assert r.status_code == 200, r.text
    assert r.json()["quarantined"] == 1
