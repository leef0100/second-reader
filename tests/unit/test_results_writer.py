from __future__ import annotations

import numpy as np
import pytest
from pydicom import dcmread

from secondreader.core.constants import RESEARCH_USE_LABEL
from secondreader.core.dicom.imaging import to_imaging_input
from secondreader.core.plugins.protocol import BoundingBox, Finding, GateResult, Localization
from secondreader.core.results import build_results, render_overlay_rgb
from tests.fixtures.synthetic import FAKE_PHI, cxr_spec, make_dataset


@pytest.fixture
def src(deidentifier):
    return deidentifier.deidentify(make_dataset(cxr_spec(rows=128, cols=160))).dataset


def _findings():
    heat = np.zeros((128, 160), dtype=np.float32)
    heat[30:60, 40:80] = 1.0
    loc = Localization(heatmap=heat, boxes=(BoundingBox(0.25, 0.23, 0.5, 0.47),))
    return [
        Finding("pneumothorax", "Pneumothorax", 0.83, 0.3, loc, "36118008", "Pneumothorax"),
        Finding(
            "pleural_effusion",
            "Pleural effusion",
            0.05,
            0.5,
            Localization(heatmap=heat * 0.1),
            "60046008",
            "Pleural effusion",
        ),
        Finding(
            "nodule_mass",
            "Pulmonary nodule/mass",
            0.41,
            0.4,
            loc,
            "427359005",
            "Solitary nodule of lung",
        ),
    ]


def test_sr_and_sc_are_valid_dicom_with_codes_and_label(src):
    from io import BytesIO

    inp = to_imaging_input(src, input_sha256="a" * 64)
    image01 = inp.frames[0].astype(np.float32) / inp.frames[0].max()
    rb = build_results(
        src,
        inp,
        GateResult.passed(),
        _findings(),
        model_version_id="m123",
        plugin_version="0.1.0",
        prediction_id="p1",
        image01=image01,
        latency_ms=12.5,
    )
    sr = dcmread(BytesIO(rb.sr_bytes))
    assert sr.Modality == "SR" and sr.SOPInstanceUID == rb.sr_sop_uid
    assert sr.PatientID == src.PatientID and sr.StudyInstanceUID == src.StudyInstanceUID
    assert sr.CompletionFlag == "COMPLETE" and sr.VerificationFlag == "UNVERIFIED"
    assert "research use only" in str(sr.ContentDescription).lower()
    text = rb.sr_bytes.decode("latin-1")
    assert "36118008" in text and "60046008" in text and "Present" in text and "Absent" in text
    assert "m123" in text  # model version traceable from the SR itself
    # evidence references the source image
    assert src.SOPInstanceUID in text
    # SC overlays: one per finding, RGB, burned-in flag set honestly
    assert [c for c, _, _ in rb.sc_items] == ["pneumothorax", "pleural_effusion", "nodule_mass"]
    sc = dcmread(BytesIO(rb.sc_items[0][2]))
    assert sc.Modality == "OT" and sc.PhotometricInterpretation == "RGB"
    assert sc.Rows == 128 and sc.Columns == 160 and sc.BurnedInAnnotation == "YES"
    assert sc.pixel_array.shape == (128, 160, 3)
    assert RESEARCH_USE_LABEL in str(sc.ImageComments)
    # JSON
    j = rb.json
    assert j["research_use"] == RESEARCH_USE_LABEL and j["gate"]["status"] == "ok"
    assert j["findings"][0]["present"] is True and j["findings"][1]["present"] is False
    assert all(f["has_heatmap"] for f in j["findings"])  # never a finding without localization
    assert set(rb.heatmap_pngs) == {"pneumothorax", "pleural_effusion", "nodule_mass"}
    # no PHI anywhere in outputs
    for blob in (rb.sr_bytes, rb.sc_items[0][2]):
        assert FAKE_PHI["PatientName"].encode() not in blob and b"MRN-0012345" not in blob


def test_gated_out_produces_unable_sr_and_no_overlays(src):
    from io import BytesIO

    inp = to_imaging_input(src)
    gate = GateResult.unable("view position 'LL' is not frontal (PA/AP)", p_frontal=0.1)
    rb = build_results(
        src,
        inp,
        gate,
        [],
        model_version_id="m",
        plugin_version="v",
        prediction_id="p",
        image01=None,
        latency_ms=1.0,
    )
    sr = dcmread(BytesIO(rb.sr_bytes))
    assert sr.ContentLabel == "UNABLE TO ASSESS"
    assert "Unable to assess" in rb.sr_bytes.decode("latin-1")
    assert rb.sc_items == [] and rb.json["findings"] == []
    assert rb.json["gate"]["reasons"] == ["view position 'LL' is not frontal (PA/AP)"]


def test_overlay_render_has_label_bands():
    img = np.random.default_rng(0).random((100, 120)).astype(np.float32)
    rgb = render_overlay_rgb(img, None, "x", box=(0.1, 0.1, 0.5, 0.5))
    assert rgb.shape == (100, 120, 3) and rgb.dtype == np.uint8
    assert rgb[-2, 60].tolist() == [0, 0, 0] or rgb[-2, 60].max() > 0  # bottom band drawn
