"""Quality / view / out-of-distribution gate for chest radiographs (principle 2).

Layers, cheapest first; any failure → `unable_to_assess` with named reasons:
  1. header  — modality, body part, view position
  2. pixels  — size, aspect, dynamic range, saturation
  3. learned — (a) view head: frontal vs lateral/other, trained with the model;
               (b) Mahalanobis distance of penultimate features vs the training
                   feature distribution; threshold = validation percentile.
Layers 3a/3b are skipped (and reported as unavailable) when the model bundle
lacks them, so the gate degrades honestly rather than silently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from secondreader.core.plugins.protocol import GateResult, ImagingInput
from secondreader.modalities.cxr.config import GateConfig


@dataclass
class OodModel:
    mean: NDArray[np.float64]  # (D,)
    precision: NDArray[np.float64]  # (D, D)
    threshold: float

    def distance(self, feat: NDArray[np.float64]) -> float:
        d = feat.astype(np.float64) - self.mean
        return float(np.sqrt(max(d @ self.precision @ d, 0.0)))

    @classmethod
    def fit(
        cls, train_feats: NDArray[np.float64], val_feats: NDArray[np.float64], percentile: float
    ) -> OodModel:
        from sklearn.covariance import LedoitWolf

        # Shrinkage covariance stays well-conditioned even when n < D (e.g. smoke runs).
        lw = LedoitWolf().fit(train_feats)
        mean = np.asarray(lw.location_, dtype=np.float64)
        precision = np.asarray(lw.precision_, dtype=np.float64)
        m = cls(mean=mean, precision=precision, threshold=0.0)
        dists = np.array([m.distance(f) for f in val_feats])
        m.threshold = float(np.percentile(dists, percentile)) if len(dists) else float("inf")
        return m


def header_checks(inp: ImagingInput, cfg: GateConfig) -> list[str]:
    reasons: list[str] = []
    mod = (inp.modality or "").upper()
    if mod not in {m.upper() for m in cfg.allowed_modalities}:
        reasons.append(f"modality {mod or '?'} is not a radiograph (expected CR/DX)")
    bp = (inp.body_part_examined or "").upper()
    if bp not in {b.upper() for b in cfg.allowed_body_parts}:
        reasons.append(f"body part '{bp}' is not chest")
    view = (inp.view_position or "").upper()
    if view not in {v.upper() for v in cfg.frontal_views}:
        reasons.append(f"view position '{view}' is not frontal (PA/AP)")
    if inp.n_frames != 1:
        reasons.append(f"expected a single frame, got {inp.n_frames}")
    return reasons


def pixel_checks(
    a01: NDArray[np.float32], src_h: int, src_w: int, cfg: GateConfig
) -> tuple[list[str], dict[str, float]]:
    reasons: list[str] = []
    short, long_ = min(src_h, src_w), max(src_h, src_w)
    aspect = long_ / max(1, short)
    std = float(a01.std())
    sat = float(((a01 <= 0.0) | (a01 >= 1.0)).mean())
    scores = {"short_side": float(short), "aspect": aspect, "std": std, "saturated_fraction": sat}
    if short < cfg.min_short_side:
        reasons.append(f"image too small ({src_h}x{src_w})")
    if aspect > cfg.max_aspect_ratio:
        reasons.append(f"aspect ratio {aspect:.2f} outside chest radiograph range")
    if std < cfg.min_std_fraction:
        reasons.append("image has no usable contrast (blank/uniform)")
    if sat > cfg.max_saturated_fraction:
        reasons.append(f"{sat:.0%} of pixels saturated")
    return reasons, scores


def learned_checks(
    cfg: GateConfig,
    *,
    view_probs: NDArray[np.float64] | None,  # [p_frontal, p_lateral, p_other] or None
    features: NDArray[np.float64] | None,
    ood: OodModel | None,
) -> tuple[list[str], dict[str, float]]:
    reasons: list[str] = []
    scores: dict[str, float] = {}
    if cfg.view_classifier_enabled and view_probs is not None:
        p_front = float(view_probs[0])
        scores["p_frontal"] = p_front
        scores["p_lateral"] = float(view_probs[1])
        scores["p_other"] = float(view_probs[2]) if len(view_probs) > 2 else 0.0
        if p_front < 1.0 - cfg.view_lateral_threshold:
            reasons.append(
                f"view classifier: not a frontal chest radiograph (p_frontal={p_front:.2f})"
            )
    else:
        scores["view_classifier"] = float("nan")  # unavailable — reported, not hidden
    if cfg.ood_enabled and ood is not None and features is not None:
        d = ood.distance(features)
        scores["ood_distance"] = d
        scores["ood_threshold"] = ood.threshold
        if d > ood.threshold:
            reasons.append(
                f"out-of-distribution input (feature distance {d:.1f} > {ood.threshold:.1f})"
            )
    else:
        scores["ood_distance"] = float("nan")
    return reasons, scores


def combine(*parts: tuple[list[str], dict[str, float]]) -> GateResult:
    reasons: list[str] = []
    scores: dict[str, Any] = {}
    for r, s in parts:
        reasons.extend(r)
        scores.update(s)
    return GateResult.unable(*reasons, **scores) if reasons else GateResult.passed(**scores)
