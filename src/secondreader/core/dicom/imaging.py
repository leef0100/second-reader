"""De-identified DICOM dataset → `ImagingInput` (modality-agnostic)."""

from __future__ import annotations

from typing import Any

from pydicom import Dataset

from secondreader.core.dicom.pseudonym import parse_age_years
from secondreader.core.dicom.reader import header_dict, pixel_frames
from secondreader.core.plugins.protocol import FrameMeta, ImagingInput


def _f(ds: Dataset, kw: str) -> float | None:
    v = ds.get(kw)
    if v is None:
        return None
    try:
        if isinstance(v, list | tuple) or type(v).__name__ == "MultiValue":
            v = v[0]
        return float(v)
    except (TypeError, ValueError):
        return None


def _spacing(ds: Dataset) -> tuple[float, float] | None:
    for kw in ("PixelSpacing", "ImagerPixelSpacing"):
        v = ds.get(kw)
        if v is not None and len(v) == 2:
            try:
                return float(v[0]), float(v[1])
            except (TypeError, ValueError):
                continue
    return None


def to_imaging_input(ds: Dataset, *, input_sha256: str = "") -> ImagingInput:
    frames = pixel_frames(ds)
    n = int(frames.shape[0])
    meta = FrameMeta(
        photometric_interpretation=str(ds.get("PhotometricInterpretation", "") or "") or None,
        rescale_slope=_f(ds, "RescaleSlope"),
        rescale_intercept=_f(ds, "RescaleIntercept"),
        window_center=_f(ds, "WindowCenter"),
        window_width=_f(ds, "WindowWidth"),
        pixel_spacing_mm=_spacing(ds),
        frame_time_ms=_f(ds, "FrameTime"),
    )
    age_s = str(ds.get("PatientAge", "") or "")
    hdr: dict[str, Any] = header_dict(ds)
    return ImagingInput(
        frames=frames,
        frame_meta=tuple(meta for _ in range(n)),
        modality=str(ds.get("Modality", "") or ""),
        sop_class_uid=str(ds.get("SOPClassUID", "") or ""),
        body_part_examined=str(ds.get("BodyPartExamined", "") or "") or None,
        view_position=str(ds.get("ViewPosition", "") or "") or None,
        patient_sex=str(ds.get("PatientSex", "") or "") or None,
        patient_age_years=parse_age_years(age_s) if age_s else None,
        manufacturer=str(ds.get("Manufacturer", "") or "") or None,
        manufacturer_model=str(ds.get("ManufacturerModelName", "") or "") or None,
        source_tags=hdr,
        input_sha256=input_sha256,
    )
