"""Patient-level hashed splits with locked, hashed manifests (D-008)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from secondreader.modalities.cxr.data import load_manifest, manifest_sha256, save_manifest

TIERS = ("train", "val", "test", "external")


def assign_tier(salt: str, source: str, patient_id: str, *, train: int = 80, val: int = 10) -> str:
    h = hashlib.sha256(f"{salt}\x1f{source}\x1f{patient_id}".encode()).hexdigest()
    bucket = int(h[:8], 16) % 100
    if bucket < train:
        return "train"
    if bucket < train + val:
        return "val"
    return "test"


@dataclass
class SplitSummary:
    version: str
    salt: str
    manifest_sha256: str
    tiers: dict[str, dict[str, object]]


def build_splits(
    manifest: pd.DataFrame,
    *,
    out_dir: Path,
    version: str,
    salt: str,
    external_datasets: set[str],
    train_pct: int = 80,
    val_pct: int = 10,
) -> SplitSummary:
    df = manifest.copy()
    is_ext = df["dataset"].isin(external_datasets)
    df.loc[is_ext, "tier"] = "external"
    int_df = df.loc[~is_ext]
    df.loc[~is_ext, "tier"] = [
        assign_tier(salt, d, p, train=train_pct, val=val_pct)
        for d, p in zip(int_df["dataset"], int_df["patient_id"], strict=True)
    ]
    check_no_patient_leakage(df)
    out = Path(out_dir) / version
    out.mkdir(parents=True, exist_ok=True)
    tiers: dict[str, dict[str, object]] = {}
    for t in TIERS:
        part = df[df["tier"] == t]
        p = out / f"{t}.csv"
        sha = save_manifest(part, p)
        tiers[t] = {
            "n": int(len(part)),
            "sha256": sha,
            "path": str(p),
            "datasets": sorted(part["dataset"].unique().tolist()),
        }
    combined = hashlib.sha256("".join(str(tiers[t]["sha256"]) for t in TIERS).encode()).hexdigest()
    (out / "manifest.sha256").write_text(combined + "\n")
    summary = SplitSummary(version=version, salt=salt, manifest_sha256=combined, tiers=tiers)
    (out / "splits.json").write_text(json.dumps(summary.__dict__, indent=2))
    return summary


def check_no_patient_leakage(df: pd.DataFrame) -> None:
    """A (dataset, patient) pair must live in exactly one tier."""
    g = df.groupby(["dataset", "patient_id"])["tier"].nunique()
    bad = g[g > 1]
    if len(bad):
        raise ValueError(f"patient leakage across tiers: {bad.head().to_dict()}")
    ext = set(df.loc[df["tier"] == "external", "dataset"])
    internal = set(df.loc[df["tier"] != "external", "dataset"])
    if ext & internal:
        raise ValueError(f"dataset in both external and internal tiers: {ext & internal}")


def verify_locked(split_dir: Path, expected_sha256: str) -> None:
    got = (Path(split_dir) / "manifest.sha256").read_text().strip()
    if got != expected_sha256:
        raise ValueError(f"split manifest hash changed: {got} != {expected_sha256}")
    for t in TIERS:
        summary = json.loads((Path(split_dir) / "splits.json").read_text())
        if manifest_sha256(Path(split_dir) / f"{t}.csv") != summary["tiers"][t]["sha256"]:
            raise ValueError(f"{t}.csv modified since lock")


def dhash(arr: np.ndarray, size: int = 8) -> int:
    """Difference hash for near-duplicate detection across datasets."""
    from PIL import Image

    im = Image.fromarray(np.asarray(arr, dtype=np.float32)).resize((size + 1, size), Image.BILINEAR)
    a = np.asarray(im, dtype=np.float32)
    diff = a[:, 1:] > a[:, :-1]
    return int("".join("1" if b else "0" for b in diff.flatten()), 2)


def check_image_overlap(
    train_paths: list[Path], external_paths: list[Path], *, max_hamming: int = 2
) -> list[tuple[Path, Path]]:
    """Perceptual-hash overlap between train and external images (NIH re-release trap)."""
    from secondreader.modalities.cxr.data import load_image

    def h(p: Path) -> int:
        arr, _ = load_image(p)
        return dhash(arr)

    train_h = {h(p): p for p in train_paths}
    hits: list[tuple[Path, Path]] = []
    for p in external_paths:
        hv = h(p)
        for th, tp in train_h.items():
            if bin(hv ^ th).count("1") <= max_hamming:
                hits.append((tp, p))
                break
    return hits


def load_tier(split_dir: Path, tier: str) -> pd.DataFrame:
    return load_manifest(Path(split_dir) / f"{tier}.csv")
