"""Shared evaluation metrics: discrimination, operating-point, calibration, bootstrap CIs.

Everything here is modality-agnostic and works on plain numpy arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score

FloatArr = NDArray[np.float64]


@dataclass(frozen=True)
class BinaryMetrics:
    n: int
    n_pos: int
    auroc: float | None
    sensitivity: float | None
    specificity: float | None
    ppv: float | None
    npv: float | None
    ece: float | None
    brier: float | None
    threshold: float
    low_n: bool
    ci: dict[str, tuple[float, float]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "n": self.n,
            "n_pos": self.n_pos,
            "auroc": self.auroc,
            "sensitivity": self.sensitivity,
            "specificity": self.specificity,
            "ppv": self.ppv,
            "npv": self.npv,
            "ece": self.ece,
            "brier": self.brier,
            "threshold": self.threshold,
            "low_n": self.low_n,
        }
        d["ci"] = {k: list(v) for k, v in self.ci.items()}
        return d


@dataclass(frozen=True)
class ReliabilityBins:
    bin_edges: FloatArr
    bin_confidence: FloatArr  # mean predicted prob per bin (nan if empty)
    bin_accuracy: FloatArr  # empirical positive rate per bin
    bin_count: NDArray[np.int64]


def expected_calibration_error(
    p: FloatArr, y: FloatArr, n_bins: int = 15
) -> tuple[float, ReliabilityBins]:
    """Equal-width-bin ECE for a binary output (probability of the positive class)."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    conf = np.full(n_bins, np.nan)
    acc = np.full(n_bins, np.nan)
    cnt = np.zeros(n_bins, dtype=np.int64)
    ece = 0.0
    n = len(p)
    for b in range(n_bins):
        m = idx == b
        cnt[b] = int(m.sum())
        if cnt[b] == 0:
            continue
        conf[b] = float(p[m].mean())
        acc[b] = float(y[m].mean())
        ece += (cnt[b] / n) * abs(acc[b] - conf[b])
    return float(ece), ReliabilityBins(edges, conf, acc, cnt)


def _safe_auroc(p: FloatArr, y: FloatArr) -> float | None:
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, p))


def _rates(
    p: FloatArr, y: FloatArr, thr: float
) -> tuple[float | None, float | None, float | None, float | None]:
    pred = p >= thr
    pos, neg = y == 1, y == 0
    tp, fn = int((pred & pos).sum()), int((~pred & pos).sum())
    tn, fp = int((~pred & neg).sum()), int((pred & neg).sum())
    sens = tp / (tp + fn) if tp + fn else None
    spec = tn / (tn + fp) if tn + fp else None
    ppv = tp / (tp + fp) if tp + fp else None
    npv = tn / (tn + fn) if tn + fn else None
    return sens, spec, ppv, npv


def binary_metrics(
    p: FloatArr,
    y: FloatArr,
    threshold: float,
    *,
    n_boot: int = 200,
    seed: int = 0,
    low_n_positives: int = 100,
) -> BinaryMetrics:
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, n_pos = int(len(y)), int((y == 1).sum())
    if n == 0:
        return BinaryMetrics(0, 0, None, None, None, None, None, None, None, threshold, True)
    auroc = _safe_auroc(p, y)
    sens, spec, ppv, npv = _rates(p, y, threshold)
    ece, _ = expected_calibration_error(p, y)
    brier = float(np.mean((p - y) ** 2))
    ci: dict[str, tuple[float, float]] = {}
    if n_boot > 0 and n >= 20:
        rng = np.random.default_rng(seed)
        boots: dict[str, list[float]] = {
            "auroc": [],
            "sensitivity": [],
            "specificity": [],
            "ece": [],
        }
        for _ in range(n_boot):
            i = rng.integers(0, n, n)
            pb, yb = p[i], y[i]
            a = _safe_auroc(pb, yb)
            if a is not None:
                boots["auroc"].append(a)
            s, sp, _, _ = _rates(pb, yb, threshold)
            if s is not None:
                boots["sensitivity"].append(s)
            if sp is not None:
                boots["specificity"].append(sp)
            boots["ece"].append(expected_calibration_error(pb, yb)[0])
        for k, v in boots.items():
            if len(v) >= 10:
                ci[k] = (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return BinaryMetrics(
        n,
        n_pos,
        auroc,
        sens,
        spec,
        ppv,
        npv,
        ece,
        brier,
        threshold,
        low_n=n_pos < low_n_positives,
        ci=ci,
    )


def youden_threshold(p: FloatArr, y: FloatArr) -> float:
    """Threshold maximizing sensitivity + specificity - 1 on (validation) data.

    Candidates are midpoints between consecutive distinct scores, so a perfectly
    separable validation set yields a threshold between the classes rather than
    on top of the lowest positive.
    """
    if len(np.unique(y)) < 2:
        return 0.5
    s = np.unique(p)
    if len(s) < 2:
        return 0.5
    cands = (s[:-1] + s[1:]) / 2
    best_t, best_j = 0.5, -1.0
    for t in cands:
        sens, spec, _, _ = _rates(p, y, float(t))
        j = (sens or 0.0) + (spec or 0.0) - 1.0
        if j > best_j:
            best_j, best_t = j, float(t)
    return float(np.clip(best_t, 1e-4, 1 - 1e-4))


def sensitivity_target_threshold(p: FloatArr, y: FloatArr, target: float = 0.9) -> float:
    """Largest threshold achieving at least `target` sensitivity."""
    pos = np.sort(p[y == 1])
    if len(pos) == 0:
        return 0.5
    k = int(np.floor((1 - target) * len(pos)))
    return float(np.clip(pos[min(k, len(pos) - 1)], 1e-4, 1 - 1e-4))


def fit_temperature(
    logits: FloatArr,
    y: FloatArr,
    *,
    mask: FloatArr | None = None,
    bounds: tuple[float, float] = (0.25, 4.0),
) -> float:
    """1-D temperature scaling by NLL minimization (scipy), per output.

    `bounds` stop the temperature collapsing toward 0 on a perfectly separable
    (usually tiny) validation set, which would produce absurdly confident outputs.
    """
    from scipy.optimize import minimize_scalar

    z = np.asarray(logits, dtype=np.float64)
    t_y = np.asarray(y, dtype=np.float64)
    if mask is not None:
        z, t_y = z[mask.astype(bool)], t_y[mask.astype(bool)]
    if len(np.unique(t_y)) < 2:
        return 1.0

    def nll(log_t: float) -> float:
        s = z / np.exp(log_t)
        # stable BCE with logits
        return float(np.mean(np.maximum(s, 0) - s * t_y + np.log1p(np.exp(-np.abs(s)))))

    res = minimize_scalar(nll, bounds=(np.log(bounds[0]), np.log(bounds[1])), method="bounded")
    return float(np.exp(res.x))


def fit_platt(
    logits: FloatArr,
    y: FloatArr,
    *,
    temperature_bounds: tuple[float, float] = (0.25, 4.0),
    bias_bounds: tuple[float, float] = (-6.0, 6.0),
) -> tuple[float, float]:
    """Platt scaling p = sigmoid(z / T + b) by NLL minimisation. Returns (T, b).

    Temperature alone cannot undo the logit *offset* introduced by positive-class
    weighting (pos_weight); the bias term can. Bounded so a tiny or perfectly
    separable validation set cannot drive T → 0 or |b| → ∞.
    """
    from scipy.optimize import minimize

    z = np.asarray(logits, dtype=np.float64)
    t_y = np.asarray(y, dtype=np.float64)
    if len(np.unique(t_y)) < 2:
        return 1.0, 0.0

    def nll(params: FloatArr) -> float:
        log_t, b = params
        s = z / np.exp(log_t) + b
        return float(np.mean(np.maximum(s, 0) - s * t_y + np.log1p(np.exp(-np.abs(s)))))

    res = minimize(
        nll,
        x0=np.array([0.0, 0.0]),
        bounds=[(np.log(temperature_bounds[0]), np.log(temperature_bounds[1])), bias_bounds],
        method="L-BFGS-B",
    )
    return float(np.exp(res.x[0])), float(res.x[1])


def sigmoid(z: FloatArr) -> FloatArr:
    return np.asarray(1.0 / (1.0 + np.exp(-z)), dtype=np.float64)
