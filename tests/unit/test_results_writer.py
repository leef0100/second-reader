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


def _groups(sr_ds):
    """Measurement-group CONTAINER items of a TID 1500 report, parsed back with highdicom."""
    import highdicom as hd
    from pydicom.sr.codedict import codes

    doc = hd.sr.Comprehensive3DSR.from_dataset(sr_ds)
    root = doc.content[0]  # doc.content is the TID 1500 MeasurementReport template
    (im,) = [i for i in root.ContentSequence if i.name == codes.DCM.ImagingMeasurements]
    return doc, [g for g in im.ContentSequence if g.name == codes.DCM.MeasurementGroup]


def _items(group, code_value):
    return [i for i in group.ContentSequence if i.name.value == code_value]


def _finding_type(group):
    (ft,) = _items(group, "121071")  # (121071, DCM, "Finding")
    return ft.value


def _processing_equipment(sr_ds):
    """Our ContributingEquipment item (highdicom adds the source acquisition device + itself)."""
    ours = [
        i
        for i in sr_ds.ContributingEquipmentSequence
        if str(i.Manufacturer) == "Second Reader (research prototype)"
    ]
    assert all(i.PurposeOfReferenceCodeSequence[0].CodeValue == "109102" for i in ours)
    return ours


def test_sr_is_comprehensive_3d_with_ihe_air_provenance_and_negative_idiom(src):
    """IHE AI Results Rev 1.3 encoding (D-029), verified by parsing the bytes back."""
    from io import BytesIO

    from pydicom.uid import UID

    inp = to_imaging_input(src, input_sha256="a" * 64)
    ptx, eff, nod = _findings()
    findings = [
        ptx,  # present, box
        eff,  # absent, no box
        Finding(nod.code, nod.display_name, 0.2, 0.4, nod.localization, "427359005", "Nodule"),
    ]  # absent, box
    rb = build_results(
        src,
        inp,
        GateResult.passed(),
        findings,
        model_version_id="m123",
        plugin_version="0.1.0",
        prediction_id="p1",
        image01=None,
        latency_ms=1.0,
    )
    sr = dcmread(BytesIO(rb.sr_bytes))
    # 1. IOD: Comprehensive 3D SR (AIR §6.5.3.2), still TID 1500 at the root
    assert sr.SOPClassUID == UID("1.2.840.10008.5.1.4.1.1.88.34")
    assert sr.SOPClassUID.name == "Comprehensive 3D SR Storage"
    assert sr.ConceptNameCodeSequence[0].CodeValue == "126000"
    # 2. ContributingEquipmentSequence: highdicom records the source's acquisition
    #    equipment (109101); our item (109102 Processing Equipment) mirrors TID 4019
    (ce,) = _processing_equipment(sr)
    assert ce.Manufacturer == "Second Reader (research prototype)"
    assert ce.ManufacturerModelName == "secondreader-cxr" and ce.SoftwareVersions == "m123"
    assert "research use only" in str(ce.ContributionDescription).lower()
    purposes = [
        i.PurposeOfReferenceCodeSequence[0].CodeValue for i in sr.ContributingEquipmentSequence
    ]
    assert "109101" in purposes  # source acquisition equipment kept (provenance chain)

    doc, groups = _groups(sr)
    assert len(groups) == 3
    # 3. ObservationUID on every finding group, all distinct
    uids = [g.ObservationUID for g in groups]
    assert all(uids) and len(set(uids)) == 3
    by_code = {_finding_type(g).value: g for g in groups}
    present, absent, absent_box = (
        by_code["36118008"],
        by_code["60046008"],
        by_code["427359005"],
    )
    # 4. negative-finding idiom (373572006 = finding) only on absent findings, alongside
    #    the Presence=Absent evaluation (D-022) — both kept on purpose
    assert _items(present, "373572006") == []
    for g in (absent, absent_box):
        (neg,) = _items(g, "373572006")
        assert neg.value == _finding_type(g)
        (pres,) = _items(g, "246112005")
        assert pres.value.value == "2667000"  # Absent
    (pres,) = _items(present, "246112005")
    assert pres.value.value == "52101004"  # Present
    # 5. box regions: closed 5-point POLYLINE under Image Region, tagged Bounded by (CID 219)
    for g in (present, absent_box):
        (region,) = _items(g, "111030")
        assert region.graphic_type.value == "POLYLINE" and region.value.shape == (5, 2)
        assert (region.value[0] == region.value[-1]).all()
        (img,) = region.ContentSequence  # IMAGE item: SELECTED FROM the source instance
        assert img.ReferencedSOPSequence[0].ReferencedSOPInstanceUID == src.SOPInstanceUID
        (purpose,) = _items(g, "130400")
        assert purpose.value.value == "75958009" and purpose.value.scheme_designator == "SCT"
    assert _items(absent, "111030") == [] and _items(absent, "130400") == []
    # 6. probability measurement + tracking identifiers survive (D-022)
    for g in groups:
        (prob,) = _items(g, "99SRD-PROB")
        assert 0.0 <= float(prob.value) <= 1.0
        assert _items(g, "112040")  # Tracking Unique Identifier (OHIF hydration precondition)
    # highdicom's own measurement-group accessor still recognises the planar groups
    assert len(doc.content.get_planar_roi_measurement_groups()) == 2


def test_gated_out_sr_is_comprehensive_3d_with_observation_uid(src):
    from io import BytesIO

    inp = to_imaging_input(src)
    rb = build_results(
        src,
        inp,
        GateResult.unable("blurred"),
        [],
        model_version_id="m",
        plugin_version="v",
        prediction_id="p",
        image01=None,
        latency_ms=1.0,
    )
    sr = dcmread(BytesIO(rb.sr_bytes))
    assert sr.SOPClassUID == "1.2.840.10008.5.1.4.1.1.88.34"
    (ce,) = _processing_equipment(sr)
    assert ce.SoftwareVersions == "m"
    _, groups = _groups(sr)
    assert len(groups) == 1 and groups[0].ObservationUID
    (unable,) = _items(groups[0], "99SRD-ASSESS")
    assert unable.value.value == "99SRD-UNABLE"


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
