"""Patient-level hashed splits with locked, hashed manifests (D-008).

An *official* split (D-029, R1) pins one dataset's published test list to the tier
`official_test` and hashes only the remaining (train/val-list) patients into train/val,
so numbers can be compared with the literature (NIH `test_list.txt`).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from secondreader.modalities.cxr.data import load_manifest, manifest_sha256, save_manifest

TIERS = ("train", "val", "test", "external")
OFFICIAL_TEST_TIER = "official_test"


@dataclass(frozen=True)
class OfficialSplit:
    """A dataset's published image lists. Rows of `dataset` whose image_id is in `test_ids`
    become `official_test`; when `trainval_ids` is given, rows in neither list are dropped
    (reported in splits.json) so train/val contain only the published train/val patients."""

    dataset: str
    test_ids: frozenset[str]
    trainval_ids: frozenset[str] | None = None
    source: str = ""

    @staticmethod
    def _read_list(path: Path) -> frozenset[str]:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        return frozenset(line.strip() for line in lines if line.strip())

    @classmethod
    def nih(cls, test_list: Path, train_val_list: Path | None = None) -> OfficialSplit:
        """NIH ChestX-ray14 `test_list.txt` (+ sibling `train_val_list.txt` when present)."""
        test_list = Path(test_list)
        if train_val_list is None:
            sib = test_list.with_name("train_val_list.txt")
            train_val_list = sib if sib.exists() else None
        return cls(
            dataset="nih_cxr14",
            test_ids=cls._read_list(test_list),
            trainval_ids=cls._read_list(train_val_list) if train_val_list else None,
            source=str(test_list),
        )


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
    official: dict[str, object] | None = field(default=None)


def build_splits(
    manifest: pd.DataFrame,
    *,
    out_dir: Path,
    version: str,
    salt: str,
    external_datasets: set[str],
    train_pct: int = 80,
    val_pct: int = 10,
    official: OfficialSplit | None = None,
) -> SplitSummary:
    """Assign tiers and write `<out_dir>/<version>/{tier}.csv` + hashes.

    Without `official`, the tiers are train/val/test/external (hashed patients, D-008).
    With it, `official_test` is added, filled from the published test list, and the
    hashed buckets only ever produce train/val (pass `train_pct + val_pct == 100`; any
    bucket beyond that would land in `test`, which the caller should not want here).
    """
    df = manifest.copy()
    official_info: dict[str, object] | None = None
    is_off = pd.Series(False, index=df.index)
    if official is not None:
        in_ds = df["dataset"] == official.dataset
        is_off = in_ds & df["image_id"].astype(str).isin(official.test_ids)
        n_dropped = 0
        if official.trainval_ids is not None:
            drop = in_ds & ~is_off & ~df["image_id"].astype(str).isin(official.trainval_ids)
            n_dropped = int(drop.sum())
            df = df.loc[~drop].copy()
            is_off = is_off.loc[df.index]
        df.loc[is_off, "tier"] = OFFICIAL_TEST_TIER
        official_info = {
            "dataset": official.dataset,
            "source": official.source,
            "n_test_ids": len(official.test_ids),
            "n_test_matched": int(is_off.sum()),
            "n_dropped_unlisted": n_dropped,
            "train_pct": train_pct,
            "val_pct": val_pct,
        }
    is_ext = df["dataset"].isin(external_datasets)
    df.loc[is_ext, "tier"] = "external"
    hashed = ~is_ext & ~is_off
    int_df = df.loc[hashed]
    df.loc[hashed, "tier"] = [
        assign_tier(salt, d, p, train=train_pct, val=val_pct)
        for d, p in zip(int_df["dataset"], int_df["patient_id"], strict=True)
    ]
    check_no_patient_leakage(df)
    out = Path(out_dir) / version
    out.mkdir(parents=True, exist_ok=True)
    tier_names = (*TIERS, OFFICIAL_TEST_TIER) if official is not None else TIERS
    tiers: dict[str, dict[str, object]] = {}
    for t in tier_names:
        part = df[df["tier"] == t]
        p = out / f"{t}.csv"
        sha = save_manifest(part, p)
        tiers[t] = {
            "n": int(len(part)),
            "sha256": sha,
            "path": str(p),
            "datasets": sorted(part["dataset"].unique().tolist()),
        }
    combined = hashlib.sha256(
        "".join(str(tiers[t]["sha256"]) for t in tier_names).encode()
    ).hexdigest()
    (out / "manifest.sha256").write_text(combined + "\n")
    summary = SplitSummary(
        version=version, salt=salt, manifest_sha256=combined, tiers=tiers, official=official_info
    )
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
    summary = json.loads((Path(split_dir) / "splits.json").read_text())
    for t, info in summary["tiers"].items():
        if manifest_sha256(Path(split_dir) / f"{t}.csv") != info["sha256"]:
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
