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
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import structlog
import torch
from numpy.typing import NDArray
from PIL import Image
from torch.utils.data import Dataset

from secondreader.core.plugins.protocol import FrameMeta
from secondreader.modalities.cxr.config import FINDING_CODES, PreprocessConfig
from secondreader.modalities.cxr.preprocess import preprocess_frame

if TYPE_CHECKING:
    from secondreader.modalities.cxr.cache import FrameCache

log = structlog.get_logger(__name__)

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
    """Manifest rows → model tensors.

    With a `FrameCache` (built by `secondreader cxr cache`) rows are read from the uint8
    memmap and only normalization + augmentation run per item; rows the cache lacks are
    decoded on the fly, so a `--limit` cache or a failed decode never changes the epoch.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        pre: PreprocessConfig,
        *,
        augment: bool = False,
        root: Path | None = None,
        cache: FrameCache | None = None,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.pre = pre
        self.augment = augment
        self.root = root
        self.cache = cache
        self._aug: Any = None
        if augment:
            from torchvision.transforms import v2

            self._aug = v2.RandomAffine(degrees=7, translate=(0.05, 0.05), scale=(0.9, 1.1))
        self._cache_rows: list[int | None] = [None] * len(self.df)
        if cache is not None:
            if cache.size != pre.image_size:
                raise ValueError(
                    f"cache size {cache.size} != preprocess.image_size {pre.image_size}"
                )
            self._cache_rows = [cache.row_for(i) for i in self.df["image_id"].astype(str)]
            n_miss = sum(r is None for r in self._cache_rows)
            if n_miss:
                log.warning(
                    "cache_partial", n_uncached=n_miss, n=len(self.df), cache=str(cache.dir)
                )

    def __len__(self) -> int:
        return len(self.df)

    @property
    def n_cached(self) -> int:
        return sum(r is not None for r in self._cache_rows)

    def _path(self, i: int) -> Path:
        path = Path(self.df.iloc[i]["path"])
        if self.root is not None and not path.is_absolute():
            path = self.root / path
        return path

    def tensor(self, i: int) -> torch.Tensor:
        """Un-augmented (3, S, S) normalized tensor for row i (cache fast path or decode)."""
        row = self._cache_rows[i]
        if self.cache is not None and row is not None:
            return self.cache.tensor(row)
        arr, meta = load_image(self._path(i))
        return preprocess_frame(arr, meta, self.pre).tensor

    def input_stats(self, i: int) -> dict[str, float]:
        """Per-image drift statistics for row i (from the cache index when available)."""
        row = self._cache_rows[i]
        if self.cache is not None and row is not None:
            return self.cache.stats(row)
        arr, meta = load_image(self._path(i))
        return preprocess_frame(arr, meta, self.pre).stats

    def __getitem__(self, i: int) -> dict[str, Any]:
        row = self.df.iloc[i]
        x = self.tensor(i)
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
