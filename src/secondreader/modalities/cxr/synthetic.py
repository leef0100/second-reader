"""Synthetic labelled chest-radiograph-like dataset for pipeline smoke tests.

NOT a substitute for real data. It exists so the entire train → calibrate → gate →
evaluate → serve path can be exercised end-to-end, with subgroup metadata and a
deliberately shifted "external" source, without downloading anything.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from secondreader.modalities.cxr.data import MANIFEST_COLUMNS, save_manifest


def lung_image(rows: int, cols: int, seed: int = 0, bits: int = 12) -> np.ndarray:
    """Crude frontal-chest-like image: dark lung ellipses on a brighter body."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:rows, 0:cols]
    cy, cx = rows / 2, cols / 2
    body = np.exp(-(((x - cx) / (0.42 * cols)) ** 2 + ((y - cy) / (0.55 * rows)) ** 2))
    left = np.exp(
        -(((x - 0.35 * cols) / (0.14 * cols)) ** 2 + ((y - 0.48 * rows) / (0.3 * rows)) ** 2)
    )
    right = np.exp(
        -(((x - 0.65 * cols) / (0.14 * cols)) ** 2 + ((y - 0.48 * rows) / (0.3 * rows)) ** 2)
    )
    spine = np.exp(-(((x - cx) / (0.03 * cols)) ** 2)) * body
    img = 0.15 + 0.7 * body - 0.45 * (left + right) * body + 0.25 * spine
    img += rng.normal(0, 0.02, size=img.shape)
    img = np.clip(img, 0, 1)
    return (img * (2**bits - 1)).astype(np.uint16)


def _lateral(rows: int, cols: int, rng: np.random.Generator) -> np.ndarray:
    y, x = np.mgrid[0:rows, 0:cols]
    body = np.exp(
        -(((x - cols * 0.5) / (0.3 * cols)) ** 2 + ((y - rows * 0.5) / (0.55 * rows)) ** 2)
    )
    lung = np.exp(
        -(((x - cols * 0.55) / (0.18 * cols)) ** 2 + ((y - rows * 0.5) / (0.32 * rows)) ** 2)
    )
    spine = np.exp(-(((x - cols * 0.3) / (0.03 * cols)) ** 2)) * body
    img = 0.15 + 0.7 * body - 0.5 * lung * body + 0.3 * spine + rng.normal(0, 0.02, (rows, cols))
    return np.clip(img, 0, 1)


def _other(rows: int, cols: int, rng: np.random.Generator) -> np.ndarray:
    base = rng.random((rows // 16, cols // 16))
    img = np.kron(base, np.ones((16, 16)))[:rows, :cols]
    return np.clip(img + rng.normal(0, 0.05, (rows, cols)), 0, 1)


def _draw_findings(img: np.ndarray, labels: dict[str, int], rng: np.random.Generator) -> np.ndarray:
    rows, cols = img.shape
    y, x = np.mgrid[0:rows, 0:cols]
    out = img.copy()
    side = rng.integers(0, 2)
    lung_cx = (0.35 if side == 0 else 0.65) * cols
    if labels["pneumothorax"]:
        # dark apical/lateral crescent: air where lung markings should be
        edge_x = (0.22 if side == 0 else 0.78) * cols
        crescent = np.exp(
            -(((x - edge_x) / (0.06 * cols)) ** 2 + ((y - 0.3 * rows) / (0.18 * rows)) ** 2)
        )
        out -= 0.35 * crescent
    if labels["pleural_effusion"]:
        # bright meniscus filling the lower lung field
        level = 0.62 * rows + rng.uniform(-0.03, 0.03) * rows
        fill = (y > level) & (np.abs(x - lung_cx) < 0.16 * cols) & (y < 0.8 * rows)
        out[fill] += 0.35
    if labels["nodule_mass"]:
        r = rng.uniform(0.02, 0.05) * cols
        cx, cy = lung_cx + rng.uniform(-0.08, 0.08) * cols, rows * rng.uniform(0.3, 0.6)
        blob = np.exp(-(((x - cx) ** 2 + (y - cy) ** 2) / (2 * r * r)))
        out += 0.4 * blob
    return np.clip(out, 0, 1)


def generate(
    out_dir: Path,
    *,
    n_patients: int = 120,
    images_per_patient: int = 2,
    size: int = 256,
    seed: int = 0,
    dataset: str = "synth-a",
    contrast: float = 1.0,
    noise: float = 0.02,
    lateral_fraction: float = 0.1,
    other_fraction: float = 0.05,
    prevalence: tuple[float, float, float] = (0.25, 0.3, 0.3),
) -> pd.DataFrame:
    """Write PNGs + return a manifest DataFrame. Two vendors, ages, sexes, PA/AP."""
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for p in range(n_patients):
        pid = f"{dataset}-P{p:05d}"
        age = int(rng.integers(18, 95))
        sex = "M" if rng.random() < 0.5 else "F"
        vendor = "FauxVendor-1" if rng.random() < 0.6 else "FauxVendor-2"
        for k in range(images_per_patient):
            iid = f"{pid}-{k}"
            u = rng.random()
            view_label = (
                0 if u > lateral_fraction + other_fraction else (1 if u > other_fraction else 2)
            )
            labels = {
                "pneumothorax": int(rng.random() < prevalence[0]),
                "pleural_effusion": int(rng.random() < prevalence[1]),
                "nodule_mass": int(rng.random() < prevalence[2]),
            }
            if view_label == 0:
                img = lung_image(size, size, seed=int(rng.integers(0, 1 << 31)), bits=8) / 255.0
                img = _draw_findings(img, labels, rng)
                view = "PA" if rng.random() < 0.7 else "AP"
                lab = labels
            elif view_label == 1:
                img = _lateral(size, size, rng)
                view, lab = "LL", {c: -1 for c in labels}
            else:
                img = _other(size, size, rng)
                view, lab = "", {c: -1 for c in labels}
            # source-specific shift: contrast/noise differ between synth-a and synth-b
            img = np.clip((img - 0.5) * contrast + 0.5 + rng.normal(0, noise, img.shape), 0, 1)
            if vendor == "FauxVendor-2":
                img = np.clip(img**1.15, 0, 1)
            path = out_dir / "images" / f"{iid}.png"
            Image.fromarray((img * 255).astype(np.uint8)).save(path)
            rows.append(
                {
                    "dataset": dataset,
                    "image_id": iid,
                    "patient_id": pid,
                    "path": str(path),
                    "tier": "",
                    **lab,
                    "age": age,
                    "sex": sex,
                    "view": view,
                    "vendor": vendor,
                    "view_label": view_label,
                }
            )
    df = pd.DataFrame(rows, columns=list(MANIFEST_COLUMNS))
    save_manifest(df, out_dir / "manifest.csv")
    return df
