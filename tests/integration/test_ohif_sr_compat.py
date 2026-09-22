"""Live integration: our Comprehensive 3D SR survives Orthanc B and its DICOMweb API.

Uploads a de-identified synthetic CXR + the SR we write for it to Orthanc B (the
platform archive OHIF reads), then fetches the SR back through QIDO-RS and WADO-RS
(metadata JSON and the instance bytes) and checks that everything OHIF's SR extension
needs to display and hydrate the box (SCOORD POLYLINE, codes, tracking UID, referenced
SOP in the same study) is intact. What this does NOT verify — rendering in the pinned
OHIF v3.9.2 UI — is listed in docs/reports/ohif_sr_compat.md.

Skips unless Orthanc B is reachable (see docker-compose.yml: `docker compose up -d orthanc-b`).
"""

from __future__ import annotations

import os
import time
from io import BytesIO
from typing import Any

import httpx
import numpy as np
import pytest
from pydicom import dcmread

from secondreader.core.dicom.imaging import to_imaging_input
from secondreader.core.pacs import OrthancClient
from secondreader.core.plugins.protocol import BoundingBox, Finding, GateResult, Localization
from secondreader.core.results import build_results
from tests.fixtures.synthetic import FAKE_PHI, cxr_spec, make_dataset

pytestmark = pytest.mark.integration

ORTHANC_B = os.environ.get("SR_ORTHANC_B_URL", "http://localhost:8043")
AUTH = ("orthanc", os.environ.get("ORTHANC_B_PASSWORD", "orthanc"))
COMPREHENSIVE_3D_SR = "1.2.840.10008.5.1.4.1.1.88.34"


@pytest.fixture(scope="module")
def orthanc_b():
    c = OrthancClient(ORTHANC_B, *AUTH)
    if not c.reachable():
        pytest.skip("Orthanc B not reachable")
    return c


@pytest.fixture(scope="module")
def dicomweb():
    return httpx.Client(base_url=f"{ORTHANC_B}/dicom-web", auth=AUTH, timeout=60)


def _findings(rows: int, cols: int) -> list[Finding]:
    heat = np.zeros((rows, cols), dtype=np.float32)
    heat[rows // 4 : rows // 2, cols // 4 : cols // 2] = 1.0
    box = Localization(heatmap=heat, boxes=(BoundingBox(0.25, 0.25, 0.5, 0.5),))
    return [
        Finding("pneumothorax", "Pneumothorax", 0.81, 0.3, box, "36118008", "Pneumothorax"),
        Finding(
            "pleural_effusion",
            "Pleural effusion",
            0.06,
            0.5,
            Localization(heatmap=heat * 0.1),
            "60046008",
            "Pleural effusion",
        ),
        Finding(
            "nodule_mass", "Pulmonary nodule/mass", 0.35, 0.4, box, "427359005", "Solitary nodule"
        ),
    ]


def _tag(ds_json: dict[str, Any], tag: str) -> Any:
    v = ds_json.get(tag, {}).get("Value")
    return v[0] if v else None


def _walk(items: list[dict[str, Any]]):
    for it in items:
        yield it
        yield from _walk(it.get("0040A730", {}).get("Value", []))  # ContentSequence


def _code(item: dict[str, Any], seq: str = "0040A043") -> str | None:  # ConceptNameCodeSequence
    v = _tag(item, seq)
    return _tag(v, "00080100") if v else None  # CodeValue


def _multipart_dicom(resp: httpx.Response) -> bytes:
    """First application/dicom part of a WADO-RS multipart/related response."""
    ctype = resp.headers["content-type"]
    boundary = ctype.split("boundary=")[1].split(";")[0].strip('"')
    parts = resp.content.split(f"--{boundary}".encode())
    for p in parts:
        if b"application/dicom" in p.split(b"\r\n\r\n", 1)[0]:
            body = p.split(b"\r\n\r\n", 1)[1]
            return body[:-2] if body.endswith(b"\r\n") else body
    raise AssertionError("no application/dicom part in WADO-RS response")


def test_sr_roundtrip_through_orthanc_b_dicomweb(orthanc_b, dicomweb, deidentifier):
    rows, cols = 256, 256
    src = deidentifier.deidentify(
        make_dataset(cxr_spec(rows=rows, cols=cols, seed=int(time.time()) % 997))
    ).dataset
    inp = to_imaging_input(src, input_sha256="b" * 64)
    rb = build_results(
        src,
        inp,
        GateResult.passed(),
        _findings(rows, cols),
        model_version_id="ohifcompat0000001",
        plugin_version="0.1.0",
        prediction_id="ohif-compat",
        image01=None,
        latency_ms=1.0,
    )
    buf = BytesIO()
    src.save_as(buf, enforce_file_format=True)
    up = orthanc_b.upload(buf.getvalue())
    study_uid = str(src.StudyInstanceUID)
    try:
        sr_up = orthanc_b.upload(rb.sr_bytes)
        assert sr_up["ParentStudy"] == up["ParentStudy"], "SR must land in the image's study"

        # --- QIDO-RS: study, SR series, SR instance (what OHIF's study list / loader query)
        studies = dicomweb.get("/studies", params={"StudyInstanceUID": study_uid}).json()
        assert len(studies) == 1 and _tag(studies[0], "0020000D") == study_uid
        series = dicomweb.get(f"/studies/{study_uid}/series").json()
        mods = sorted(_tag(s, "00080060") for s in series)
        assert mods == ["CR", "SR"]
        (sr_series,) = [s for s in series if _tag(s, "00080060") == "SR"]
        sr_series_uid = _tag(sr_series, "0020000E")
        insts = dicomweb.get(f"/studies/{study_uid}/series/{sr_series_uid}/instances").json()
        assert len(insts) == 1
        assert _tag(insts[0], "00080016") == COMPREHENSIVE_3D_SR
        assert _tag(insts[0], "00080018") == rb.sr_sop_uid
        # the image the SR references is in the same study (OHIF's SOPInstanceUID lookup)
        all_insts = dicomweb.get(f"/studies/{study_uid}/instances").json()
        assert str(src.SOPInstanceUID) in {_tag(i, "00080018") for i in all_insts}

        # --- WADO-RS metadata JSON: the representation OHIF parses for SR content
        meta = dicomweb.get(
            f"/studies/{study_uid}/series/{sr_series_uid}/instances/{rb.sr_sop_uid}/metadata"
        ).json()
        (m,) = meta
        assert _tag(m, "00080016") == COMPREHENSIVE_3D_SR
        assert _code(m) == "126000"  # Imaging Measurement Report root
        assert "0040A730" in m  # root ContentSequence present
        # ContributingEquipmentSequence: source acquisition equipment + our processing item
        (ce,) = [i for i in m["0018A001"]["Value"] if _tag(i, "00081090") == "secondreader-cxr"]
        assert _code(ce, "0040A170") == "109102"  # PurposeOfReferenceCodeSequence = Processing
        assert _tag(ce, "00181020") == "ohifcompat0000001"  # SoftwareVersions = model version
        items = list(_walk(m["0040A730"]["Value"]))
        scoords = [i for i in items if _tag(i, "0040A040") == "SCOORD"]
        assert len(scoords) == 2
        for sc in scoords:
            assert _code(sc) == "111030"  # Image Region
            assert _tag(sc, "00700023") == "POLYLINE"
            assert len(sc["00700022"]["Value"]) == 10  # 5 closed points × (col,row)
            (img,) = sc["0040A730"]["Value"]  # IMAGE child: SELECTED FROM the source
            assert _tag(img, "0040A040") == "IMAGE"
            ref = _tag(img, "00081199")  # ReferencedSOPSequence
            assert _tag(ref, "00081155") == str(src.SOPInstanceUID)
        groups = [i for i in items if _code(i) == "125007"]  # Measurement Group
        assert len(groups) == 3 and all(_tag(g, "0040A171") for g in groups)  # ObservationUID
        codes_present = {_code(i) for i in items}
        assert {"112040", "373572006", "130400", "246112005", "99SRD-PROB"} <= codes_present
        negs = [i for i in items if _code(i) == "373572006"]  # the two absent findings
        assert {_tag(_tag(n, "0040A168"), "00080100") for n in negs} == {"60046008", "427359005"}
        (purpose,) = {_tag(_tag(i, "0040A168"), "00080100") for i in items if _code(i) == "130400"}
        assert purpose == "75958009"  # Bounded by

        # --- WADO-RS instance bytes: identical content after Orthanc storage, no PHI
        r = dicomweb.get(
            f"/studies/{study_uid}/series/{sr_series_uid}/instances/{rb.sr_sop_uid}",
            headers={"Accept": 'multipart/related; type="application/dicom"'},
        )
        assert r.status_code == 200, r.text
        back = dcmread(BytesIO(_multipart_dicom(r)))
        orig = dcmread(BytesIO(rb.sr_bytes))
        assert back.SOPClassUID == COMPREHENSIVE_3D_SR
        assert back.ContentSequence == orig.ContentSequence
        assert back.ContributingEquipmentSequence == orig.ContributingEquipmentSequence
        assert back.VerificationFlag == "UNVERIFIED" and back.PreliminaryFlag == "PRELIMINARY"
        assert FAKE_PHI["PatientName"] not in str(back) and "MRN-0012345" not in str(back)
        # OHIF RectangleROI adapter match: (DCM 111030, POLYLINE, 5 points) per group with a box
        import highdicom as hd

        doc = hd.sr.Comprehensive3DSR.from_dataset(back)
        planar = doc.content.get_planar_roi_measurement_groups()
        assert len(planar) == 2 and all(
            g.roi.graphic_type.value == "POLYLINE" and g.roi.value.shape == (5, 2) for g in planar
        )
    finally:
        orthanc_b.delete_study(up["ParentStudy"])
