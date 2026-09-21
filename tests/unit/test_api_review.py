"""API tests with a stub plugin: read-first gating, feedback, audit, artefact access, drift."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from secondreader.core.constants import RESEARCH_USE_LABEL
from secondreader.core.db import (
    AuditEvent,
    Prediction,
    create_all,
    make_engine,
    make_session_factory,
)
from secondreader.core.dicom import Deidentifier, load_profile
from secondreader.core.ingest import IngestService
from secondreader.core.plugins import PluginRegistry
from secondreader.core.plugins.protocol import (
    BoundingBox,
    Finding,
    GateResult,
    ImagingInput,
    Localization,
    PluginInfo,
)
from secondreader.core.security import LocalAuthProvider, create_user
from secondreader.core.serving.deps import Container
from secondreader.core.settings import Settings
from secondreader.core.storage import EncryptedBlobStore, FilesystemBlobStore
from tests.fixtures.synthetic import cxr_spec, make_bytes


class StubPlugin:
    """Deterministic plugin: gate by view, findings with a fixed probability and heatmap."""

    def __init__(self) -> None:
        self.info = PluginInfo(
            "cxr", "stub-1", "stub", Path("/dev/null"), ("pneumothorax", "pleural_effusion")
        )
        self.model_version_id = "stubmodel0000001"
        self.bundle = type(
            "B",
            (),
            {
                "card": {
                    "name": "stub",
                    "reference": {
                        "input_stats": {
                            "mean": {
                                "edges": [-np.inf, 0.4, 0.6, np.inf],
                                "counts": [1, 8, 1],
                                "mean": 0.5,
                                "std": 0.1,
                            }
                        },
                        "outputs": {
                            "pneumothorax": {
                                "edges": [-np.inf, 0.5, np.inf],
                                "counts": [5, 5],
                                "mean": 0.5,
                            }
                        },
                        "gate_out_rate": 0.05,
                        "n": 10,
                    },
                }
            },
        )()
        self._last: ImagingInput | None = None

    def accepts(self, header):  # noqa: ANN001
        return str(header.get("Modality")) in ("CR", "DX")

    def quality_gate(self, inp: ImagingInput) -> GateResult:
        self._last = inp
        if (inp.view_position or "").upper() not in ("PA", "AP", ""):
            return GateResult.unable("view position is not frontal", p_frontal=0.1)
        return GateResult.passed(p_frontal=0.95, ood_distance=1.0)

    def predict(self, inp: ImagingInput) -> list[Finding]:
        h, w = inp.frames.shape[1:3]
        heat = np.zeros((h, w), dtype=np.float32)
        heat[h // 4 : h // 2, w // 4 : w // 2] = 1.0
        loc = Localization(heatmap=heat, boxes=(BoundingBox(0.25, 0.25, 0.5, 0.5),))
        return [
            Finding("pneumothorax", "Pneumothorax", 0.82, 0.3, loc, "36118008", "Pneumothorax"),
            Finding(
                "pleural_effusion",
                "Pleural effusion",
                0.07,
                0.5,
                Localization(heatmap=heat * 0.1),
                "60046008",
                "Pleural effusion",
            ),
        ]

    def input_stats(self, inp: ImagingInput) -> dict[str, float]:
        return {"mean": 0.5, "std": 0.1}

    def display_image(self):  # noqa: ANN201
        assert self._last is not None
        f = self._last.frames[0].astype(np.float32)
        return f / max(float(f.max()), 1.0)


class NoOrthanc:
    def reachable(self) -> bool:
        return False


@pytest.fixture
def client(tmp_path, monkeypatch, repo_root):
    settings = Settings(
        pseudonym_secret="test-pseudonym-secret-not-for-production-0000",
        blob_encryption_key="0f" * 32,
        auth_secret="test-auth-secret-not-for-production",
        database_url="sqlite+pysqlite:///:memory:",
        blob_root=tmp_path / "blobs",
        model_registry_root=tmp_path / "models",
        deid_profile=repo_root / "configs/deid/basic_profile.yaml",
        bootstrap_admin_password="bootstrap-admin-pw-1",
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
        orthanc_a=NoOrthanc(),
        orthanc_b=NoOrthanc(),  # type: ignore[arg-type]
        auth=LocalAuthProvider("test-auth-secret-not-for-production", 60),
    )
    reg = PluginRegistry()
    reg.register(StubPlugin())
    c.set_plugins(reg)
    with c.sessions() as s:
        create_user(s, "drhouse", "clinician-pw-12345", "clinician")
        s.commit()
    from secondreader.core.serving import app as app_module
    from secondreader.core.serving.routes import common

    monkeypatch.setattr(app_module, "get_container", lambda: c)
    monkeypatch.setattr(common, "get_container", lambda: c)
    app_module.app.dependency_overrides[app_module.container] = lambda: c
    app_module.app.dependency_overrides[common.container] = lambda: c
    with TestClient(app_module.app) as tc:
        yield tc, c
    app_module.app.dependency_overrides.clear()


def _login(tc, user="drhouse", pw="clinician-pw-12345"):
    r = tc.post("/v1/auth/login", json={"username": user, "password": pw})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def test_login_and_header(client):
    tc, _ = client
    r = tc.post("/v1/auth/login", json={"username": "drhouse", "password": "wrong-password"})
    assert r.status_code == 401
    h = _login(tc)
    me = tc.get("/v1/auth/me", headers=h)
    assert me.json()["role"] == "clinician" and me.headers["X-Research-Use"] == RESEARCH_USE_LABEL
    assert tc.get("/v1/studies").status_code == 401  # no token
    # bootstrap admin was created from settings
    assert _login(tc, "admin", "bootstrap-admin-pw-1")


def test_sync_infer_then_read_first_workflow(client):
    tc, c = client
    h = _login(tc)
    # 1. sync inference (programmatic path)
    r = tc.post(
        "/v1/infer",
        files={"file": ("a.dcm", make_bytes(cxr_spec(rows=96, cols=96)), "application/dicom")},
        headers=h,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["gate"]["status"] == "ok" and body["research_use"] == RESEARCH_USE_LABEL
    assert body["findings"][0]["present"] is True and body["findings"][0]["has_heatmap"]
    assert not any(k.endswith("_blob_key") for k in body["findings"][0])
    study_id, pred_id = body["study_id"], body["prediction_id"]

    # 2. study browsing never exposes findings
    lst = tc.get("/v1/studies", headers=h).json()
    assert lst[0]["id"] == study_id and lst[0]["prediction_status"] == "ok"
    det = tc.get(f"/v1/studies/{study_id}", headers=h).json()
    assert det["predictions"][0]["id"] == pred_id and "findings" not in det["predictions"][0]
    assert det["ohif_url"] is None  # no Orthanc B in tests
    img = tc.get(
        f"/v1/studies/{study_id}/instances/{det['instances'][0]['id']}/preview.png", headers=h
    )
    assert img.status_code == 200 and img.headers["content-type"] == "image/png"

    # 3. artefacts are locked until reveal
    assert tc.get(f"/v1/predictions/{pred_id}", headers=h).status_code == 403
    assert (
        tc.get(f"/v1/predictions/{pred_id}/heatmap/pneumothorax.png", headers=h).status_code == 403
    )

    # 4. read-first: reveal refused before impression
    rs = tc.post(
        "/v1/review/sessions", json={"study_id": study_id, "mode": "read_first"}, headers=h
    ).json()
    assert tc.post(f"/v1/review/sessions/{rs['id']}/reveal", headers=h).status_code == 409
    assert (
        tc.post(
            f"/v1/review/sessions/{rs['id']}/feedback",
            headers=h,
            json={"prediction_id": pred_id, "finding_code": "pneumothorax", "agreement": "agree"},
        ).status_code
        == 409
    )
    imp = tc.post(
        f"/v1/review/sessions/{rs['id']}/impression",
        headers=h,
        json={
            "findings": {"pneumothorax": False, "pleural_effusion": False},
            "notes": "looks clear",
        },
    )
    assert imp.status_code == 200 and imp.json()["pre_read_impression"]["notes"] == "looks clear"
    rev = tc.post(f"/v1/review/sessions/{rs['id']}/reveal", headers=h)
    assert rev.status_code == 200, rev.text
    rj = rev.json()
    assert rj["session"]["revealed_at"] is not None
    f0 = rj["predictions"][0]["findings"][0]
    assert f0["probability"] == 0.82 and f0["heatmap_url"] and f0["overlay_url"] and f0["boxes"]
    # impression is locked after reveal
    assert (
        tc.post(
            f"/v1/review/sessions/{rs['id']}/impression", headers=h, json={"findings": {}}
        ).status_code
        == 409
    )

    # 5. artefacts now accessible, all audited
    assert (
        tc.get(f"/v1/predictions/{pred_id}/heatmap/pneumothorax.png", headers=h).status_code == 200
    )
    ov = tc.get(f"/v1/predictions/{pred_id}/overlay/pneumothorax.png", headers=h)
    assert ov.status_code == 200 and ov.headers["content-type"] == "image/png"
    sr = tc.get(f"/v1/predictions/{pred_id}/sr.dcm", headers=h)
    assert sr.status_code == 200 and sr.content[128:132] == b"DICM"
    sc = tc.get(f"/v1/predictions/{pred_id}/sc/pneumothorax.dcm", headers=h)
    assert sc.status_code == 200 and sc.content[128:132] == b"DICM"

    # 6. feedback
    fb = tc.post(
        f"/v1/review/sessions/{rs['id']}/feedback",
        headers=h,
        json={
            "prediction_id": pred_id,
            "finding_code": "pneumothorax",
            "agreement": "disagree",
            "comment": "artefact",
        },
    )
    assert fb.status_code == 200
    assert (
        tc.post(
            f"/v1/review/sessions/{rs['id']}/feedback",
            headers=h,
            json={"prediction_id": pred_id, "finding_code": "bogus", "agreement": "agree"},
        ).status_code
        == 400
    )
    assert tc.post(f"/v1/review/sessions/{rs['id']}/complete", headers=h).json()["completed_at"]

    # 7. audit trail records who viewed what
    with c.sessions() as s:
        actions = [e.action for e in s.query(AuditEvent).order_by(AuditEvent.id)]
        assert {
            "login",
            "infer_sync",
            "studies_listed",
            "study_viewed",
            "image_viewed",
            "review_started",
            "pre_read_recorded",
            "ai_revealed",
            "heatmap_viewed",
            "sr_downloaded",
            "feedback_recorded",
            "review_completed",
        } <= set(actions)
        who = {e.username for e in s.query(AuditEvent) if e.action == "ai_revealed"}
        assert who == {"drhouse"}
        pred = s.get(Prediction, pred_id)
        assert (
            pred.model_version_id == "stubmodel0000001"
            and pred.input_sha256
            and pred.input_stats == {"mean": 0.5, "std": 0.1}
        )

    # 8. drift + monitoring endpoints
    d = tc.get("/v1/monitoring/drift?days=1", headers=h).json()
    assert d["n_predictions"] == 1 and d["feedback"]["pneumothorax"]["disagree"] == 1
    assert "input_psi" in d and "pneumothorax" in d["output_psi"]
    md = tc.get("/v1/monitoring/drift.md", headers=h)
    assert md.status_code == 200 and RESEARCH_USE_LABEL in md.text
    models = tc.get("/v1/monitoring/models", headers=h).json()
    assert models[0]["id"] == "stubmodel0000001" and models[0]["n_predictions"] == 1
    assert tc.get("/v1/monitoring/audit", headers=h).status_code == 403  # clinician
    assert (
        tc.get(
            "/v1/monitoring/audit", headers=_login(tc, "admin", "bootstrap-admin-pw-1")
        ).status_code
        == 200
    )


def test_immediate_mode_and_unable_to_assess(client):
    tc, _ = client
    h = _login(tc)
    r = tc.post(
        "/v1/infer",
        files={
            "file": (
                "l.dcm",
                make_bytes(cxr_spec(rows=96, cols=96, view_position="LL")),
                "application/dicom",
            )
        },
        headers=h,
    )
    assert r.status_code == 200
    assert r.json()["gate"]["status"] == "unable_to_assess" and r.json()["findings"] == []
    study_id = r.json()["study_id"]
    rs = tc.post(
        "/v1/review/sessions", json={"study_id": study_id, "mode": "immediate"}, headers=h
    ).json()
    rev = tc.post(f"/v1/review/sessions/{rs['id']}/reveal", headers=h).json()
    assert rev["predictions"][0]["gate_status"] == "unable_to_assess"
    assert rev["predictions"][0]["gate_reasons"] == ["view position is not frontal"]
    # SR exists even when unable to assess; no SC
    pid = rev["predictions"][0]["id"]
    assert tc.get(f"/v1/predictions/{pid}/sr.dcm", headers=h).status_code == 200
    assert tc.get(f"/v1/predictions/{pid}/sc/pneumothorax.dcm", headers=h).status_code == 404


def test_quarantined_upload_never_reaches_model(client):
    tc, _ = client
    h = _login(tc)
    from tests.fixtures.synthetic import secondary_capture_spec

    r = tc.post(
        "/v1/infer",
        files={"file": ("sc.dcm", make_bytes(secondary_capture_spec()), "application/dicom")},
        headers=h,
    )
    assert r.status_code == 200 and r.json()["status"] == "quarantined"
    det = tc.get(f"/v1/studies/{r.json()['study_id']}", headers=h).json()
    inst = det["instances"][0]
    assert inst["quarantined"]
    assert (
        tc.get(
            f"/v1/studies/{det['study']['id']}/instances/{inst['id']}/preview.png", headers=h
        ).status_code
        == 423
    )
