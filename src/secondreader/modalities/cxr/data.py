"""Unified manifest schema, image loading, and the training Dataset.

Manifest columns (CSV):
  dataset, image_id, patient_id, path, tier,
  pneumothorax, pleural_effusion, nodule_mass   (1 / 0 / -1 uncertain / blank unknown)
  age, sex, view, vendor, view_label
  (view_label: 0 frontal, 1 lateral, 2 other, -1 unknown)
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from PIL import Image
from torch.utils.data import Dataset

from secondreader.core.plugins.protocol import FrameMeta
from secondreader.modalities.cxr.config import FINDING_CODES, PreprocessConfig
from secondreader.modalities.cxr.preprocess import preprocess_frame

MANIFEST_COLUMNS: tuple[str, ...] = (
    "dataset",
    "image_id",
    "patient_id",
    "path",
    "tier",
    *FINDING_CODES,
    "age",
    "sex",
    "view",
    "vendor",
    "view_label",
)


def empty_manifest() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in MANIFEST_COLUMNS})


def load_manifest(path: Path | str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        dtype={
            "image_id": str,
            "patient_id": str,
            "path": str,
            "sex": str,
            "view": str,
            "vendor": str,
            "dataset": str,
            "tier": str,
        },
    )
    for c in MANIFEST_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    for c in FINDING_CODES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["view_label"] = pd.to_numeric(df["view_label"], errors="coerce").fillna(-1).astype(int)
    return df[list(MANIFEST_COLUMNS)]


def save_manifest(df: pd.DataFrame, path: Path | str) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = df[list(MANIFEST_COLUMNS)].sort_values(["dataset", "image_id"]).reset_index(drop=True)
    df.to_csv(path, index=False)
    return manifest_sha256(path)


def manifest_sha256(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_image(path: Path | str) -> tuple[NDArray[Any], FrameMeta | None]:
    """PNG/JPEG/DICOM → (2-D array, frame meta). Rescale/photometric is NOT applied here."""
    p = Path(path)
    if p.suffix.lower() in (".dcm", ".dicom") or p.suffix == "":
        from pydicom import dcmread

        from secondreader.core.dicom.imaging import to_imaging_input

        ds = dcmread(p)
        inp = to_imaging_input(ds)
        return inp.frames[0], inp.frame_meta[0]
    im = Image.open(p)
    if im.mode in ("I;16", "I;16B", "I;16L", "I"):
        arr = np.asarray(im, dtype=np.int32)
    else:
        arr = np.asarray(im.convert("L"))
    return arr, None


def labels_from_row(row: pd.Series) -> NDArray[np.float32]:
    """Finding targets with -1 for uncertain/unknown (masked in the loss)."""
    out = np.full(len(FINDING_CODES), -1.0, dtype=np.float32)
    for i, c in enumerate(FINDING_CODES):
        v = row.get(c)
        if v is not None and not (isinstance(v, float) and np.isnan(v)) and float(v) in (0.0, 1.0):
            out[i] = float(v)
    return out


class ManifestDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        df: pd.DataFrame,
        pre: PreprocessConfig,
        *,
        augment: bool = False,
        root: Path | None = None,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.pre = pre
        self.augment = augment
        self.root = root
        self._aug: Any = None
        if augment:
            from torchvision.transforms import v2

            self._aug = v2.RandomAffine(degrees=7, translate=(0.05, 0.05), scale=(0.9, 1.1))

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> dict[str, Any]:
        row = self.df.iloc[i]
        path = Path(row["path"])
        if self.root is not None and not path.is_absolute():
            path = self.root / path
        arr, meta = load_image(path)
        pp = preprocess_frame(arr, meta, self.pre)
        x = pp.tensor
        if self.augment:
            x = self._aug(x)
            if torch.rand(1).item() < 0.8:
                c = 1.0 + (torch.rand(1).item() - 0.5) * 0.2
                b = (torch.rand(1).item() - 0.5) * 0.2
                x = x * c + b
        view = int(row.get("view_label", -1))
        return {
            "x": x,
            "y": torch.from_numpy(labels_from_row(row)),
            "view": torch.tensor(view, dtype=torch.long),
            "index": i,
        }
