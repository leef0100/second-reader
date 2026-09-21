"""Burned-in text (PHI in pixels) risk assessment.

Two layers:
1. Header signals — deterministic and cheap: BurnedInAnnotation, screen-capture
   modalities / SOP classes.
2. A conservative pixel heuristic — looks for bright, high-frequency content in
   the border bands where annotation text is typically burned in. It is a
   heuristic, not OCR; it is tuned to over-call. Any hit means quarantine and
   human review, never silent archiving.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydicom import Dataset

from secondreader.core.dicom.profile import BurnedInConfig


@dataclass(frozen=True)
class BurnedInAssessment:
    quarantine: bool
    reasons: tuple[str, ...] = ()
    scores: dict[str, float] = field(default_factory=dict)


def assess_header(ds: Dataset, cfg: BurnedInConfig) -> list[str]:
    reasons: list[str] = []
    if cfg.quarantine_if_tag_yes and str(ds.get("BurnedInAnnotation", "")).strip().upper() == "YES":
        reasons.append("BurnedInAnnotation=YES")
    modality = str(ds.get("Modality", "")).strip().upper()
    if modality in {m.upper() for m in cfg.quarantine_modalities}:
        reasons.append(f"modality {modality} is a screen-capture/photo type")
    sop = str(ds.get("SOPClassUID", "")).strip()
    if sop in set(cfg.quarantine_sop_classes):
        reasons.append("secondary-capture SOP class")
    return reasons


def _laplacian_var(a: NDArray[np.float64]) -> float:
    if a.shape[0] < 3 or a.shape[1] < 3:
        return 0.0
    lap = -4 * a[1:-1, 1:-1] + a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:]
    return float(lap.var())


def pixel_text_risk(frame: NDArray[Any], cfg: BurnedInConfig) -> tuple[bool, dict[str, float]]:
    """Return (risky, scores) for one 2-D frame."""
    p = cfg.pixel_heuristic
    a = np.asarray(frame, dtype=np.float64)
    if a.ndim == 3:  # colour → luminance
        a = a[..., :3].mean(axis=-1)
    if a.ndim != 2 or min(a.shape) < 32:
        return False, {}
    h, w = a.shape
    band = max(4, int(round(min(h, w) * p.border_fraction)))
    bright_thr = float(np.quantile(a, p.bright_quantile))
    centre = a[band:-band, band:-band]
    centre_lv = _laplacian_var(centre) + 1e-6
    bands = {
        "top": a[:band, :],
        "bottom": a[-band:, :],
        "left": a[:, :band],
        "right": a[:, -band:],
    }
    scores: dict[str, float] = {}
    risky = False
    for name, b in bands.items():
        bright_frac = float((b > bright_thr).mean())
        contrast = _laplacian_var(b) / centre_lv
        scores[f"{name}_bright_fraction"] = bright_frac
        scores[f"{name}_contrast_ratio"] = contrast
        if bright_frac >= p.bright_fraction_threshold and contrast >= p.edge_contrast_ratio:
            risky = True
    return risky, scores


def assess(ds: Dataset, cfg: BurnedInConfig, frames: NDArray[Any] | None) -> BurnedInAssessment:
    reasons = assess_header(ds, cfg)
    scores: dict[str, float] = {}
    if cfg.pixel_heuristic.enabled and frames is not None:
        # Check the first, middle and last frame — cine loops share their overlay.
        n = int(frames.shape[0])
        idx = sorted({0, n // 2, n - 1})
        for i in idx:
            risky, s = pixel_text_risk(frames[i], cfg)
            scores.update({f"f{i}_{k}": v for k, v in s.items()})
            if risky:
                reasons.append(f"pixel heuristic: text-like border content in frame {i}")
                break
    return BurnedInAssessment(quarantine=bool(reasons), reasons=tuple(reasons), scores=scores)
