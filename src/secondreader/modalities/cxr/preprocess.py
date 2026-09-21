"""Chest X-ray preprocessing: DICOM/PNG pixels → model tensor, plus drift statistics.

Steps: rescale slope/intercept → MONOCHROME1 inversion → robust percentile window →
[0,1] → letterbox-resize to a square → 3-channel ImageNet normalization.
`Geometry` records the letterbox so heatmaps can be mapped back to source pixels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from secondreader.core.plugins.protocol import FrameMeta, ImagingInput
from secondreader.modalities.cxr.config import PreprocessConfig

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class Geometry:
    """How the source frame was placed inside the square model input."""

    src_h: int
    src_w: int
    size: int
    scale: float
    pad_top: int
    pad_left: int
    new_h: int
    new_w: int


def to_float_frame(frame: NDArray[Any], meta: FrameMeta | None) -> NDArray[np.float32]:
    a = np.asarray(frame).astype(np.float32)
    if a.ndim == 3:  # RGB → luminance
        a = a[..., :3].mean(axis=-1)
    if meta is not None:
        if meta.rescale_slope is not None or meta.rescale_intercept is not None:
            a = a * float(meta.rescale_slope or 1.0) + float(meta.rescale_intercept or 0.0)
        if (meta.photometric_interpretation or "").upper() == "MONOCHROME1":
            a = a.max() - a  # invert so bone is bright
    return a


def window(a: NDArray[np.float32], lo_pct: float, hi_pct: float) -> NDArray[np.float32]:
    lo, hi = np.percentile(a, [lo_pct, hi_pct])
    if hi <= lo:
        lo, hi = float(a.min()), float(a.max()) + 1e-6
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _resize(a: NDArray[np.float32], new_h: int, new_w: int) -> NDArray[np.float32]:
    t = torch.from_numpy(a)[None, None]
    out = torch.nn.functional.interpolate(
        t, size=(new_h, new_w), mode="bilinear", align_corners=False, antialias=True
    )
    return out[0, 0].numpy()


def letterbox(a: NDArray[np.float32], size: int) -> tuple[NDArray[np.float32], Geometry]:
    h, w = a.shape
    scale = size / max(h, w)
    new_h, new_w = max(1, round(h * scale)), max(1, round(w * scale))
    r = _resize(a, new_h, new_w)
    out = np.zeros((size, size), dtype=np.float32)
    top, left = (size - new_h) // 2, (size - new_w) // 2
    out[top : top + new_h, left : left + new_w] = r
    return out, Geometry(h, w, size, scale, top, left, new_h, new_w)


def unletterbox(map_sq: NDArray[np.float32], g: Geometry) -> NDArray[np.float32]:
    """Map a (size,size) heatmap back to (src_h, src_w)."""
    crop = map_sq[g.pad_top : g.pad_top + g.new_h, g.pad_left : g.pad_left + g.new_w]
    return _resize(np.ascontiguousarray(crop, dtype=np.float32), g.src_h, g.src_w)


def input_statistics(a01: NDArray[np.float32], src_h: int, src_w: int) -> dict[str, float]:
    """Cheap per-image stats logged with every prediction for drift monitoring."""
    hist, _ = np.histogram(a01, bins=32, range=(0.0, 1.0))
    p = hist / max(hist.sum(), 1)
    entropy = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
    return {
        "mean": float(a01.mean()),
        "std": float(a01.std()),
        "p05": float(np.percentile(a01, 5)),
        "p95": float(np.percentile(a01, 95)),
        "entropy": entropy,
        "saturated_fraction": float(((a01 <= 0.0) | (a01 >= 1.0)).mean()),
        "src_h": float(src_h),
        "src_w": float(src_w),
        "aspect": float(max(src_h, src_w) / max(1, min(src_h, src_w))),
    }


@dataclass
class Preprocessed:
    tensor: torch.Tensor  # (3, S, S) normalized
    image01: NDArray[np.float32]  # (S, S) letterboxed, [0,1] — for overlays
    geometry: Geometry
    stats: dict[str, float]


def preprocess_frame(
    frame: NDArray[Any], meta: FrameMeta | None, cfg: PreprocessConfig
) -> Preprocessed:
    a = to_float_frame(frame, meta)
    a01 = window(a, cfg.window_low_pct, cfg.window_high_pct)
    stats = input_statistics(a01, *a01.shape)
    sq, geom = letterbox(a01, cfg.image_size)
    rgb = np.repeat(sq[..., None], 3, axis=-1)
    norm = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(np.ascontiguousarray(norm.transpose(2, 0, 1)))
    return Preprocessed(tensor=tensor, image01=sq, geometry=geom, stats=stats)


def preprocess_input(
    inp: ImagingInput, cfg: PreprocessConfig, frame_index: int = 0
) -> Preprocessed:
    meta = inp.frame_meta[frame_index] if inp.frame_meta else None
    return preprocess_frame(inp.frames[frame_index], meta, cfg)
