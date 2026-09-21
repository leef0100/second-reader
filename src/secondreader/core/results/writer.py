"""Results output: JSON + DICOM SR (TID 1500) + Secondary Capture overlays (D-005).

Every artefact carries the research-use label. The SR references the source
image and, per finding, records: finding type (SNOMED), calibrated probability
(numeric measurement), presence at the operating point (qualitative evaluation),
and a bounding-box image region. Gated-out inputs produce an SR stating
"unable to assess" with the reasons, and no SC overlays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from io import BytesIO
from typing import Any

import highdicom as hd
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw
from pydicom import Dataset
from pydicom.sr.codedict import codes
from pydicom.uid import generate_uid

from secondreader.core.constants import MANUFACTURER, RESEARCH_USE_LABEL, SOFTWARE_VERSIONS
from secondreader.core.plugins.protocol import Finding, GateResult, ImagingInput

_PROBABILITY = hd.sr.CodedConcept(
    value="99SRD-PROB", scheme_designator="99SRD", meaning="Calibrated probability of finding"
)
_ABSENT = hd.sr.CodedConcept(value="2667000", scheme_designator="SCT", meaning="Absent")
_PRESENT = hd.sr.CodedConcept(value="52101004", scheme_designator="SCT", meaning="Present")
_UNABLE = hd.sr.CodedConcept(
    value="99SRD-UNABLE", scheme_designator="99SRD", meaning="Unable to assess"
)
_PRESENCE = hd.sr.CodedConcept(value="246112005", scheme_designator="SCT", meaning="Presence")
_ASSESSABILITY = hd.sr.CodedConcept(
    value="99SRD-ASSESS", scheme_designator="99SRD", meaning="Assessability"
)
_CHEST_XRAY = hd.sr.CodedConcept(
    value="399208008", scheme_designator="SCT", meaning="Plain X-ray of chest"
)


@dataclass
class ResultBundle:
    json: dict[str, Any]
    sr_bytes: bytes
    sr_sop_uid: str
    sc_items: list[tuple[str, str, bytes]] = field(default_factory=list)  # (code, sop_uid, bytes)
    heatmap_pngs: dict[str, bytes] = field(default_factory=dict)


def findings_to_json(findings: list[Finding]) -> list[dict[str, Any]]:
    out = []
    for f in findings:
        boxes = []
        if f.localization is not None:
            boxes = [
                {"x_min": b.x_min, "y_min": b.y_min, "x_max": b.x_max, "y_max": b.y_max}
                for b in f.localization.boxes
            ]
        out.append(
            {
                "code": f.code,
                "display_name": f.display_name,
                "probability": round(f.probability, 4),
                "threshold": f.threshold,
                "present": f.present,
                "boxes": boxes,
                "has_heatmap": f.localization is not None and f.localization.heatmap is not None,
                "snomed_code": f.snomed_code,
                "snomed_meaning": f.snomed_meaning,
            }
        )
    return out


def _algorithm(model_version_id: str) -> hd.sr.AlgorithmIdentification:
    return hd.sr.AlgorithmIdentification(name="secondreader-cxr", version=model_version_id)


def _observation_context(model_version_id: str) -> hd.sr.ObservationContext:
    device = hd.sr.DeviceObserverIdentifyingAttributes(
        uid=generate_uid(prefix="2.25."),
        name="Second Reader (research prototype)",
        manufacturer_name=MANUFACTURER,
        model_name="secondreader-cxr",
    )
    return hd.sr.ObservationContext(
        observer_person_context=None,
        observer_device_context=hd.sr.ObserverContext(
            observer_type=codes.DCM.Device, observer_identifying_attributes=device
        ),
    )


def _box_region(f: Finding, src: Dataset) -> hd.sr.ImageRegion | None:
    if f.localization is None or not f.localization.boxes:
        return None
    b = f.localization.boxes[0]
    rows, cols = int(src.Rows), int(src.Columns)
    # DICOM 2D image coordinates are 1-based (column, row) pixel indices, closed polyline
    pts = np.array(
        [
            [b.x_min * cols + 1, b.y_min * rows + 1],
            [b.x_max * cols, b.y_min * rows + 1],
            [b.x_max * cols, b.y_max * rows],
            [b.x_min * cols + 1, b.y_max * rows],
            [b.x_min * cols + 1, b.y_min * rows + 1],
        ],
        dtype=float,
    )
    return hd.sr.ImageRegion(
        graphic_type=hd.sr.GraphicTypeValues.POLYLINE,
        graphic_data=pts,
        source_image=hd.sr.SourceImageForRegion.from_source_image(src),
    )


# Type-2 attributes highdicom copies from the source object. De-identified inputs
# deliberately lack them (D-002); we supply empty values on a throwaway copy so the
# output stays conformant without re-introducing anything.
_TYPE2_EMPTY = (
    "PatientBirthDate",
    "PatientSex",
    "ReferringPhysicianName",
    "AccessionNumber",
    "StudyID",
    "StudyDate",
    "StudyTime",
    "PatientName",
    "PatientID",
)


def _conformant_copy(src: Dataset) -> Dataset:
    import copy

    ds = copy.deepcopy(src)
    for kw in _TYPE2_EMPTY:
        if kw not in ds:
            setattr(ds, kw, "")
    return ds


def build_sr(
    src: Dataset,
    findings: list[Finding],
    gate: GateResult,
    *,
    model_version_id: str,
    series_number: int = 900,
) -> Dataset:
    src = _conformant_copy(src)
    groups: list[
        hd.sr.PlanarROIMeasurementsAndQualitativeEvaluations
        | hd.sr.MeasurementsAndQualitativeEvaluations
    ] = []
    if gate.ok:
        for i, f in enumerate(findings, start=1):
            finding_type = hd.sr.CodedConcept(
                value=f.snomed_code or f"99SRD-{f.code}",
                scheme_designator="SCT" if f.snomed_code else "99SRD",
                meaning=f.snomed_meaning or f.display_name,
            )
            measurements = [
                hd.sr.Measurement(
                    name=_PROBABILITY,
                    value=float(round(f.probability, 4)),
                    unit=codes.UCUM.NoUnits,
                    algorithm_id=_algorithm(model_version_id),
                )
            ]
            quals = [
                hd.sr.QualitativeEvaluation(
                    name=_PRESENCE, value=_PRESENT if f.present else _ABSENT
                )
            ]
            tid = hd.sr.TrackingIdentifier(
                uid=generate_uid(prefix="2.25."), identifier=f"{f.code}-{i}"
            )
            region = _box_region(f, src)
            if region is not None:
                groups.append(
                    hd.sr.PlanarROIMeasurementsAndQualitativeEvaluations(
                        tracking_identifier=tid,
                        referenced_region=region,
                        finding_type=finding_type,
                        algorithm_id=_algorithm(model_version_id),
                        measurements=measurements,
                        qualitative_evaluations=quals,
                    )
                )
            else:
                groups.append(
                    hd.sr.MeasurementsAndQualitativeEvaluations(
                        tracking_identifier=tid,
                        finding_type=finding_type,
                        algorithm_id=_algorithm(model_version_id),
                        measurements=measurements,
                        qualitative_evaluations=quals,
                    )
                )
    else:
        tid = hd.sr.TrackingIdentifier(uid=generate_uid(prefix="2.25."), identifier="gate")
        groups.append(
            hd.sr.MeasurementsAndQualitativeEvaluations(
                tracking_identifier=tid,
                algorithm_id=_algorithm(model_version_id),
                qualitative_evaluations=[
                    hd.sr.QualitativeEvaluation(name=_ASSESSABILITY, value=_UNABLE)
                ],
            )
        )

    report = hd.sr.MeasurementReport(
        observation_context=_observation_context(model_version_id),
        procedure_reported=_CHEST_XRAY,
        imaging_measurements=groups,
        title=codes.DCM.ImagingMeasurementReport,
        # TID 1600 image library requires several Type-2C/3 CR attributes that real
        # exports often lack; the source is still referenced as evidence and by the
        # SCOORD regions, so the library is omitted for robustness.
        referenced_images=None,
    )
    sr = hd.sr.ComprehensiveSR(
        evidence=[src],
        content=report[0],
        series_instance_uid=generate_uid(prefix="2.25."),
        series_number=series_number,
        sop_instance_uid=generate_uid(prefix="2.25."),
        instance_number=1,
        manufacturer=MANUFACTURER,
        is_complete=True,
        is_final=False,  # never "final": a clinician reads it
        series_description=f"AI second reader SR - {RESEARCH_USE_LABEL}"[:64],
    )
    sr.SoftwareVersions = SOFTWARE_VERSIONS
    sr.ContentDescription = f"{RESEARCH_USE_LABEL} model={model_version_id}"[:64]
    if not gate.ok:
        sr.ContentLabel = "UNABLE TO ASSESS"
        sr.add_new(
            (0x0070, 0x0081), "LO", "; ".join(gate.reasons)[:64]
        )  # ContentDescription-ish note
    return sr


_JET = np.array(
    [
        [0, 0, 128],
        [0, 0, 255],
        [0, 128, 255],
        [0, 255, 255],
        [128, 255, 128],
        [255, 255, 0],
        [255, 128, 0],
        [255, 0, 0],
        [128, 0, 0],
    ],
    dtype=np.float32,
)


def _colormap(h: NDArray[np.float32]) -> NDArray[np.float32]:
    x = np.clip(h, 0, 1) * (len(_JET) - 1)
    i = np.floor(x).astype(int)
    j = np.clip(i + 1, 0, len(_JET) - 1)
    t = (x - i)[..., None]
    return np.asarray(_JET[i] * (1 - t) + _JET[j] * t, dtype=np.float32)


def render_overlay_rgb(
    image01: NDArray[np.float32],
    heat: NDArray[np.float32] | None,
    label: str,
    box: tuple[float, float, float, float] | None = None,
    alpha: float = 0.45,
) -> NDArray[np.uint8]:
    """Grey image + coloured heatmap + box + research label, as RGB uint8."""
    g = (np.clip(image01, 0, 1) * 255).astype(np.float32)
    rgb = np.repeat(g[..., None], 3, axis=-1)
    if heat is not None:
        w = (np.clip(heat, 0, 1) ** 1.5)[..., None] * alpha
        rgb = rgb * (1 - w) + _colormap(heat.astype(np.float32)) * w
    im = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    h, wdt = im.height, im.width
    if box is not None:
        d.rectangle(
            [box[0] * wdt, box[1] * h, box[2] * wdt, box[3] * h],
            outline=(255, 255, 0),
            width=max(2, wdt // 200),
        )
    band = max(14, h // 28)
    d.rectangle([0, 0, wdt, band], fill=(0, 0, 0))
    d.text((4, 2), label[:120], fill=(255, 255, 0))
    d.rectangle([0, h - band, wdt, h], fill=(0, 0, 0))
    d.text((4, h - band + 2), RESEARCH_USE_LABEL, fill=(255, 80, 80))
    return np.asarray(im, dtype=np.uint8)


def _orientation(src: Dataset) -> tuple[str, str]:
    v = src.get("PatientOrientation")
    if v is not None and len(v) == 2 and all(str(x).strip() for x in v):
        return str(v[0]), str(v[1])
    return "L", "F"


def build_sc(
    src: Dataset,
    rgb: NDArray[np.uint8],
    *,
    series_uid: str,
    series_number: int,
    instance_number: int,
    description: str,
) -> Dataset:
    src = _conformant_copy(src)
    sc = hd.sc.SCImage.from_ref_dataset(
        ref_dataset=src,
        pixel_array=rgb,
        photometric_interpretation=hd.PhotometricInterpretationValues.RGB,
        bits_allocated=8,
        coordinate_system=hd.CoordinateSystemNames.PATIENT,
        # Frontal radiograph convention when the source omits Type-2C PatientOrientation.
        patient_orientation=_orientation(src),
        series_instance_uid=series_uid,
        series_number=series_number,
        sop_instance_uid=generate_uid(prefix="2.25."),
        instance_number=instance_number,
        manufacturer=MANUFACTURER,
        transfer_syntax_uid="1.2.840.10008.1.2.1",
    )
    sc.SeriesDescription = description[:64]
    sc.BurnedInAnnotation = "YES"  # the label + overlay are burned in by design (no PHI)
    sc.ImageComments = RESEARCH_USE_LABEL
    sc.SoftwareVersions = SOFTWARE_VERSIONS
    return sc


def _to_bytes(ds: Dataset) -> bytes:
    buf = BytesIO()
    ds.save_as(buf, enforce_file_format=True)
    return buf.getvalue()


def _heatmap_png(heat: NDArray[np.float32]) -> bytes:
    buf = BytesIO()
    Image.fromarray((np.clip(heat, 0, 1) * 255).astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def build_results(
    src: Dataset,
    inp: ImagingInput,
    gate: GateResult,
    findings: list[Finding],
    *,
    model_version_id: str,
    plugin_version: str,
    prediction_id: str,
    image01: NDArray[np.float32] | None,
    latency_ms: float,
) -> ResultBundle:
    """Assemble JSON + SR + SC overlays for one prediction."""
    sr = build_sr(src, findings, gate, model_version_id=model_version_id)
    sc_items: list[tuple[str, str, bytes]] = []
    pngs: dict[str, bytes] = {}
    if gate.ok and image01 is not None:
        series_uid = generate_uid(prefix="2.25.")
        for i, f in enumerate(findings, start=1):
            heat = f.localization.heatmap if f.localization is not None else None
            box = None
            if f.localization is not None and f.localization.boxes:
                b = f.localization.boxes[0]
                box = (b.x_min, b.y_min, b.x_max, b.y_max)
            state = "PRESENT" if f.present else "absent"
            label = f"{f.display_name}: p={f.probability:.2f} ({state} @ {f.threshold:.2f})"
            rgb = render_overlay_rgb(image01, heat, label, box)
            sc = build_sc(
                src,
                rgb,
                series_uid=series_uid,
                series_number=901,
                instance_number=i,
                description=f"AI overlay {f.code} - research use only",
            )
            sc_items.append((f.code, str(sc.SOPInstanceUID), _to_bytes(sc)))
            if heat is not None:
                pngs[f.code] = _heatmap_png(heat)
    payload = {
        "research_use": RESEARCH_USE_LABEL,
        "prediction_id": prediction_id,
        "model_version_id": model_version_id,
        "plugin_version": plugin_version,
        "input_sha256": inp.input_sha256,
        "source_sop_instance_uid": str(src.SOPInstanceUID),
        "study_instance_uid": str(src.StudyInstanceUID),
        "gate": {
            "status": gate.status,
            "reasons": list(gate.reasons),
            "scores": {
                k: (None if isinstance(v, float) and np.isnan(v) else v)
                for k, v in gate.scores.items()
            },
        },
        "findings": findings_to_json(findings) if gate.ok else [],
        "sr_sop_instance_uid": str(sr.SOPInstanceUID),
        "sc_sop_instance_uids": {code: uid for code, uid, _ in sc_items},
        "latency_ms": latency_ms,
        "created_at": datetime.now(UTC).isoformat(),
    }
    return ResultBundle(
        json=payload,
        sr_bytes=_to_bytes(sr),
        sr_sop_uid=str(sr.SOPInstanceUID),
        sc_items=sc_items,
        heatmap_pngs=pngs,
    )
