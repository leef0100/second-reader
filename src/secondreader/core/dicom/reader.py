"""Reading / writing DICOM objects and extracting a plain header dict."""

from __future__ import annotations

import hashlib
from io import BytesIO
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydicom import Dataset, dcmread
from pydicom.uid import UID, ExplicitVRLittleEndian, generate_uid

from secondreader.core.constants import SOFTWARE_VERSIONS

# Registered-style implementation UID for objects we write. Derived from a fixed
# namespace so it is stable across runs (not a real ANSI-registered root).
IMPLEMENTATION_CLASS_UID = generate_uid(prefix="2.25.", entropy_srcs=["secondreader-impl-0.1"])


def read_dicom(data: bytes, *, force: bool = False) -> Dataset:
    return dcmread(BytesIO(data), force=force)


def dataset_to_bytes(ds: Dataset) -> bytes:
    buf = BytesIO()
    ds.save_as(buf, enforce_file_format=True)
    return buf.getvalue()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def new_file_meta(ds: Dataset, transfer_syntax: str | None = None) -> None:
    """Rebuild file meta from scratch — old file meta can carry source AE titles."""
    from pydicom.dataset import FileMetaDataset

    fm = FileMetaDataset()
    fm.MediaStorageSOPClassUID = ds.SOPClassUID
    fm.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    existing = getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", None)
    fm.TransferSyntaxUID = UID(transfer_syntax or str(existing or ExplicitVRLittleEndian))
    fm.ImplementationClassUID = IMPLEMENTATION_CLASS_UID
    fm.ImplementationVersionName = SOFTWARE_VERSIONS[:16]
    ds.file_meta = fm


HEADER_KEYWORDS: tuple[str, ...] = (
    "SOPClassUID",
    "SOPInstanceUID",
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "Modality",
    "BodyPartExamined",
    "ViewPosition",
    "PatientSex",
    "PatientAge",
    "Manufacturer",
    "ManufacturerModelName",
    "Rows",
    "Columns",
    "NumberOfFrames",
    "PhotometricInterpretation",
    "BitsStored",
    "ImageType",
    "StudyDate",
    "SeriesNumber",
    "InstanceNumber",
    "PixelSpacing",
    "ImagerPixelSpacing",
    "BurnedInAnnotation",
    "PresentationIntentType",
    "StudyDescription",
    "SeriesDescription",
)


def header_dict(ds: Dataset) -> dict[str, Any]:
    """A JSON-friendly subset of the header used for routing and metadata rows."""
    out: dict[str, Any] = {}
    for kw in HEADER_KEYWORDS:
        if kw in ds:
            v = ds.get(kw)
            if v is None:
                continue
            if isinstance(v, list | tuple) or type(v).__name__ == "MultiValue":
                out[kw] = [str(x) for x in v]
            else:
                out[kw] = str(v)
    return out


def pixel_frames(ds: Dataset) -> NDArray[Any]:
    """Pixel data as (F, H, W[, C]). Rescale is NOT applied here."""
    arr = ds.pixel_array
    n = int(ds.get("NumberOfFrames", 1) or 1)
    samples = int(ds.get("SamplesPerPixel", 1) or 1)
    if samples == 1:
        if arr.ndim == 2:
            arr = arr[None, ...]
    else:
        if arr.ndim == 3:
            arr = arr[None, ...]
    if arr.shape[0] != n and n > 1:
        raise ValueError(f"NumberOfFrames={n} but decoded {arr.shape[0]} frames")
    return np.asarray(arr)
