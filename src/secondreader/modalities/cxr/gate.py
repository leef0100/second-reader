"""Quality / view / out-of-distribution gate for chest radiographs (principle 2).

Layers, cheapest first; any failure → `unable_to_assess` with named reasons:
  1. header  — modality, body part, view position
  2. pixels  — size, aspect, dynamic range, saturation
  3. learned — (a) view head: frontal vs lateral/other, trained with the model;
               (b) Mahalanobis distance of penultimate features vs the training
                   feature distribution; threshold = validation percentile.
                   Optionally on L2-normalised features (Mahalanobis++, `gate.ood_normalize`).
               (c) MaxLogit is logged beside it as a secondary score; nothing gates on it.
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


def l2_normalize(feats: NDArray[np.float64]) -> NDArray[np.float64]:
    """Row-wise unit L2 norm (Mahalanobis++); works on one vector or a (N, D) matrix."""
    f = np.asarray(feats, dtype=np.float64)
    norm = np.linalg.norm(f, axis=-1, keepdims=True)
    return np.asarray(f / np.maximum(norm, 1e-12), dtype=np.float64)


@dataclass
class OodModel:
    mean: NDArray[np.float64]  # (D,)
    precision: NDArray[np.float64]  # (D, D)
    threshold: float
    # True when the Gaussian was fit on L2-normalised features (Mahalanobis++). Stored in
    # ood.npz so scoring always matches the fit, whatever the current config says.
    normalize: bool = False

    def distance(self, feat: NDArray[np.float64]) -> float:
        f = feat.astype(np.float64)
        if self.normalize:
            f = l2_normalize(f)
        d = f - self.mean
        return float(np.sqrt(max(d @ self.precision @ d, 0.0)))

    @classmethod
    def fit(
        cls,
        train_feats: NDArray[np.float64],
        val_feats: NDArray[np.float64],
        percentile: float,
        *,
        normalize: bool = False,
    ) -> OodModel:
        from sklearn.covariance import LedoitWolf

        tr = l2_normalize(train_feats) if normalize else np.asarray(train_feats, dtype=np.float64)
        # Shrinkage covariance stays well-conditioned even when n < D (e.g. smoke runs).
        lw = LedoitWolf().fit(tr)
        mean = np.asarray(lw.location_, dtype=np.float64)
        precision = np.asarray(lw.precision_, dtype=np.float64)
        m = cls(mean=mean, precision=precision, threshold=0.0, normalize=normalize)
        dists = np.array([m.distance(f) for f in val_feats])
        m.threshold = float(np.percentile(dists, percentile)) if len(dists) else float("inf")
        return m


def max_logit(logits: NDArray[np.float64]) -> float:
    """MaxLogit (Hendrycks et al. 2022): the largest raw finding logit. Logged as a secondary
    OOD score only — on medical near-OOD it is close to chance (OpenMIBOOD), so nothing gates
    on it (D-019 unchanged). Lower = the model is confident about nothing."""
    return float(np.max(np.asarray(logits, dtype=np.float64)))


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
    if inp.patient_age_years is not None and inp.patient_age_years < cfg.min_patient_age_years:
        reasons.append(
            f"patient age {inp.patient_age_years} y is below the adult-only scope "
            f"({cfg.min_patient_age_years} y)"
        )
    return reasons


def pixel_checks_from_stats(
    stats: dict[str, float], cfg: GateConfig
) -> tuple[list[str], dict[str, float]]:
    """Pixel gate from precomputed `input_statistics` (std, saturated_fraction, src_h/w)."""
    src_h, src_w = int(stats["src_h"]), int(stats["src_w"])
    short, long_ = min(src_h, src_w), max(src_h, src_w)
    aspect = long_ / max(1, short)
    std, sat = float(stats["std"]), float(stats["saturated_fraction"])
    reasons: list[str] = []
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


def pixel_checks(
    a01: NDArray[np.float32], src_h: int, src_w: int, cfg: GateConfig
) -> tuple[list[str], dict[str, float]]:
    stats = {
        "src_h": float(src_h),
        "src_w": float(src_w),
        "std": float(a01.std()),
        "saturated_fraction": float(((a01 <= 0.0) | (a01 >= 1.0)).mean()),
    }
    return pixel_checks_from_stats(stats, cfg)


def learned_checks(
    cfg: GateConfig,
    *,
    view_probs: NDArray[np.float64] | None,  # [p_frontal, p_lateral, p_other] or None
    features: NDArray[np.float64] | None,
    ood: OodModel | None,
    logits: NDArray[np.float64] | None = None,  # raw finding logits → MaxLogit (logged only)
) -> tuple[list[str], dict[str, float]]:
    reasons: list[str] = []
    scores: dict[str, float] = {}
    if logits is not None and len(logits):
        scores["max_logit"] = max_logit(logits)
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
