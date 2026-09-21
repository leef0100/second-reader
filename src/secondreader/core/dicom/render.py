"""Render a de-identified DICOM frame to PNG for the review UI (modality-agnostic)."""

from __future__ import annotations

from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image
from pydicom import Dataset

from secondreader.core.dicom.reader import pixel_frames


def frame_to_png(ds: Dataset, frame: int = 0, max_side: int = 1024) -> bytes:
    arr: Any = pixel_frames(ds)[frame].astype(np.float32)
    if arr.ndim == 3:
        arr = arr[..., :3].mean(axis=-1)
    slope, icpt = float(ds.get("RescaleSlope", 1) or 1), float(ds.get("RescaleIntercept", 0) or 0)
    arr = arr * slope + icpt
    if str(ds.get("PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        arr = arr.max() - arr
    lo, hi = np.percentile(arr, [0.5, 99.5])
    if hi <= lo:
        lo, hi = float(arr.min()), float(arr.max()) + 1e-6
    a8 = (np.clip((arr - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    im = Image.fromarray(a8)
    if max(im.size) > max_side:
        im.thumbnail((max_side, max_side))
    buf = BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()
