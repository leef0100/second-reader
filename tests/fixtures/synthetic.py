"""Synthetic DICOM generators for tests. FAKE PHI ONLY — never real data.

The chest-radiograph generator draws a crude lung field so image-level tests
(quality gate, preprocessing, burned-in heuristic) have something to look at.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import numpy as np
from pydicom import Dataset, Sequence
from pydicom.dataset import FileMetaDataset
from pydicom.uid import (
    ComputedRadiographyImageStorage,
    DigitalXRayImageStorageForPresentation,
    ExplicitVRLittleEndian,
    SecondaryCaptureImageStorage,
    UltrasoundMultiFrameImageStorage,
    generate_uid,
)

from secondreader.modalities.cxr.synthetic import lung_image

FAKE_PHI = {
    "PatientName": "DOE^JANE^Q",
    "PatientID": "MRN-0012345",
    "PatientBirthDate": "19700315",
    "PatientAddress": "1 Fake Street, Nowhere",
    "PatientTelephoneNumbers": "555-0100",
    "OtherPatientIDs": "ALT-99",
    "InstitutionName": "Fictional General Hospital",
    "InstitutionAddress": "2 Fake Avenue",
    "StationName": "XR-ROOM-7",
    "DeviceSerialNumber": "SN-ABC123",
    "ReferringPhysicianName": "HOUSE^GREGORY",
    "PerformingPhysicianName": "WILSON^JAMES",
    "OperatorsName": "TECH^ONE",
    "AccessionNumber": "ACC-777",
    "StudyID": "STUDY-42",
    "ImageComments": "Patient Jane Doe, MRN-0012345, walk-in",
    "AdditionalPatientHistory": "Smoker, prior pneumothorax 2019",
    "IssuerOfPatientID": "FGH",
}


@dataclass
class SyntheticSpec:
    rows: int = 256
    cols: int = 256
    frames: int = 1
    modality: str = "CR"
    sop_class: str = ComputedRadiographyImageStorage
    view_position: str = "PA"
    body_part: str = "CHEST"
    study_date: str = "20240102"
    patient_age: str | None = "054Y"
    patient_sex: str = "F"
    manufacturer: str = "ACME Imaging"
    model: str = "Fauxray 3000"
    photometric: str = "MONOCHROME2"
    bits: int = 12
    burned_in_tag: str | None = "NO"
    add_private: bool = True
    add_overlay: bool = True
    add_sequences: bool = True
    burn_text: bool = False
    study_uid: str | None = None
    series_uid: str | None = None
    seed: int = 0
    study_description: str = "CHEST PA"
    series_description: str = "Some free text with DOE JANE in it"


def burn_text_band(img: np.ndarray, bits: int = 12) -> np.ndarray:
    """Simulate burned-in annotation: bright blocky glyphs along the top edge."""
    out = img.copy()
    rows, cols = out.shape
    rng = np.random.default_rng(1)
    band_h = max(6, rows // 20)
    maxv = 2**bits - 1
    for gx in range(4, cols - 8, 9):
        glyph = rng.integers(0, 2, size=(band_h - 2, 6)).astype(bool)
        out[2:band_h, gx : gx + 6][glyph] = maxv
    return out


def make_dataset(spec: SyntheticSpec | None = None, phi: dict[str, str] | None = None) -> Dataset:
    spec = spec or SyntheticSpec()
    phi = dict(FAKE_PHI if phi is None else phi)
    ds = Dataset()
    fm = FileMetaDataset()
    fm.MediaStorageSOPClassUID = spec.sop_class
    fm.MediaStorageSOPInstanceUID = generate_uid()
    fm.TransferSyntaxUID = ExplicitVRLittleEndian
    fm.ImplementationClassUID = generate_uid()
    fm.SourceApplicationEntityTitle = "HOSPITAL_PACS"
    ds.file_meta = fm

    ds.SOPClassUID = spec.sop_class
    ds.SOPInstanceUID = fm.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = spec.study_uid or generate_uid()
    ds.SeriesInstanceUID = spec.series_uid or generate_uid()
    ds.FrameOfReferenceUID = generate_uid()
    ds.Modality = spec.modality
    ds.ImageType = ["ORIGINAL", "PRIMARY"]
    ds.StudyDate = spec.study_date
    ds.SeriesDate = spec.study_date
    ds.ContentDate = spec.study_date
    ds.AcquisitionDateTime = spec.study_date + "101500.000000"
    ds.StudyTime = "101500"
    ds.SeriesNumber = 1
    ds.InstanceNumber = 1
    ds.PatientSex = spec.patient_sex
    if spec.patient_age is not None:
        ds.PatientAge = spec.patient_age
    ds.BodyPartExamined = spec.body_part
    ds.ViewPosition = spec.view_position
    ds.Manufacturer = spec.manufacturer
    ds.ManufacturerModelName = spec.model
    ds.SoftwareVersions = "1.0"
    ds.StudyDescription = spec.study_description
    ds.SeriesDescription = spec.series_description
    ds.PixelSpacing = [0.14, 0.14]
    ds.KVP = 120
    if spec.burned_in_tag is not None:
        ds.BurnedInAnnotation = spec.burned_in_tag
    for k, v in phi.items():
        setattr(ds, k, v)

    if spec.add_sequences:
        item = Dataset()
        item.CodeValue = "51185008"
        item.CodingSchemeDesignator = "SCT"
        item.CodeMeaning = "Chest"
        item.OperatorsName = "NESTED^PHI"  # must be stripped inside the sequence
        ds.AnatomicRegionSequence = Sequence([item])
        req = Dataset()
        req.RequestedProcedureDescription = "CXR for Jane Doe"
        req.ScheduledProcedureStepID = "SPS-1"
        ds.RequestAttributesSequence = Sequence([req])

    if spec.add_private:
        block = ds.private_block(0x0009, "FICTIONAL VENDOR", create=True)
        block.add_new(0x01, "LO", "operator: TECH ONE; pt DOE JANE")
        block.add_new(0x02, "LO", "MRN-0012345")

    # pixels
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = spec.photometric
    ds.Rows = spec.rows
    ds.Columns = spec.cols
    ds.BitsAllocated = 16
    ds.BitsStored = spec.bits
    ds.HighBit = spec.bits - 1
    ds.PixelRepresentation = 0
    ds.WindowCenter = 2**spec.bits // 2
    ds.WindowWidth = 2**spec.bits
    frames = []
    for f in range(spec.frames):
        img = lung_image(spec.rows, spec.cols, seed=spec.seed + f, bits=spec.bits)
        if spec.photometric == "MONOCHROME1":
            img = (2**spec.bits - 1) - img
        if spec.burn_text:
            img = burn_text_band(img, spec.bits)
        frames.append(img)
    arr = np.stack(frames, axis=0)
    if spec.frames > 1:
        ds.NumberOfFrames = spec.frames
        ds.FrameTime = 33.3
    ds.PixelData = (arr[0] if spec.frames == 1 else arr).tobytes()

    if spec.add_overlay:
        ds.add_new((0x6000, 0x0010), "US", spec.rows)
        ds.add_new((0x6000, 0x0011), "US", spec.cols)
        ds.add_new((0x6000, 0x0040), "CS", "G")
        ds.add_new((0x6000, 0x0050), "SS", [1, 1])
        ds.add_new((0x6000, 0x0100), "US", 1)
        ds.add_new((0x6000, 0x0102), "US", 0)
        ds.add_new((0x6000, 0x3000), "OW", bytes(spec.rows * spec.cols // 8))
    return ds


def make_bytes(spec: SyntheticSpec | None = None, phi: dict[str, str] | None = None) -> bytes:
    ds = make_dataset(spec, phi)
    buf = BytesIO()
    ds.save_as(buf, enforce_file_format=True)
    return buf.getvalue()


def cxr_spec(**overrides: object) -> SyntheticSpec:
    return SyntheticSpec(**overrides)  # type: ignore[arg-type]


def dx_spec(**overrides: object) -> SyntheticSpec:
    base = {"modality": "DX", "sop_class": DigitalXRayImageStorageForPresentation}
    base.update(overrides)
    return SyntheticSpec(**base)  # type: ignore[arg-type]


def secondary_capture_spec(**overrides: object) -> SyntheticSpec:
    base = {"modality": "SC", "sop_class": SecondaryCaptureImageStorage, "bits": 8}
    base.update(overrides)
    return SyntheticSpec(**base)  # type: ignore[arg-type]


def ultrasound_cine_spec(**overrides: object) -> SyntheticSpec:
    base = {
        "modality": "US",
        "sop_class": UltrasoundMultiFrameImageStorage,
        "frames": 8,
        "body_part": "ABDOMEN",
        "view_position": "",
        "bits": 8,
        "rows": 128,
        "cols": 128,
        "add_overlay": False,
        "study_description": "US ABDOMEN",
        "series_description": "",
    }
    base.update(overrides)
    return SyntheticSpec(**base)  # type: ignore[arg-type]
