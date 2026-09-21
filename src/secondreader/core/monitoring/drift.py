"""Basic drift monitoring (principle 8).

Reference = distributions captured on the internal validation set at training time
and stored in the model card (`reference`). Live = predictions in a time window.
Population Stability Index (PSI) on the reference's decile bins for each input
statistic and each finding's calibrated probability; plus positive-rate,
gate-out-rate and clinician agreement summaries. PSI > 0.2 is flagged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from secondreader.core.constants import RESEARCH_USE_LABEL
from secondreader.core.db.models import Agreement, Feedback, Prediction

PSI_WARN = 0.1
PSI_ALERT = 0.2


def quantile_edges(x: NDArray[np.floating[Any]], n_bins: int = 10) -> list[float]:
    qs = np.quantile(np.asarray(x, dtype=np.float64), np.linspace(0, 1, n_bins + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    return [float(v) for v in qs]


def psi(
    reference_edges: list[float], ref_counts: list[float], live: NDArray[np.floating[Any]]
) -> float | None:
    if len(live) == 0:
        return None
    edges = np.asarray(reference_edges, dtype=np.float64)
    ref = np.asarray(ref_counts, dtype=np.float64)
    ref = ref / float(max(float(ref.sum()), 1e-9))
    live_hist, _ = np.histogram(np.asarray(live, dtype=np.float64), bins=edges)
    lv = live_hist / float(max(float(live_hist.sum()), 1e-9))
    eps = 1e-4
    ref, lv = np.clip(ref, eps, None), np.clip(lv, eps, None)
    return float(np.sum((lv - ref) * np.log(lv / ref)))


def build_reference(
    input_stats: list[dict[str, float]],
    probs: NDArray[np.floating[Any]],
    codes: list[str],
    gate_out_rate: float,
) -> dict[str, Any]:
    """Compute the reference block stored in the model card at training time."""
    ref: dict[str, Any] = {
        "input_stats": {},
        "outputs": {},
        "gate_out_rate": gate_out_rate,
        "n": int(len(probs)),
    }
    keys = sorted({k for s in input_stats for k in s})
    for k in keys:
        x = np.array([s[k] for s in input_stats if k in s], dtype=np.float64)
        if len(x) == 0:
            continue
        edges = quantile_edges(x)
        counts, _ = np.histogram(x, bins=np.asarray(edges))
        ref["input_stats"][k] = {
            "edges": edges,
            "counts": counts.tolist(),
            "mean": float(x.mean()),
            "std": float(x.std()),
        }
    for j, code in enumerate(codes):
        p = np.asarray(probs[:, j], dtype=np.float64)
        edges = quantile_edges(p)
        counts, _ = np.histogram(p, bins=np.asarray(edges))
        ref["outputs"][code] = {"edges": edges, "counts": counts.tolist(), "mean": float(p.mean())}
    return ref


@dataclass
class DriftReport:
    model_version_id: str
    window_start: datetime
    window_end: datetime
    n_predictions: int
    n_gated_out: int
    gate_out_rate: float | None
    reference_gate_out_rate: float | None
    input_psi: dict[str, float | None] = field(default_factory=dict)
    output_psi: dict[str, float | None] = field(default_factory=dict)
    output_mean: dict[str, float | None] = field(default_factory=dict)
    reference_output_mean: dict[str, float] = field(default_factory=dict)
    positive_rate: dict[str, float | None] = field(default_factory=dict)
    feedback: dict[str, dict[str, int]] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["window_start"], d["window_end"] = (
            self.window_start.isoformat(),
            self.window_end.isoformat(),
        )
        d["research_use"] = RESEARCH_USE_LABEL
        return d


def compute_drift(
    session: Session,
    model_version_id: str,
    reference: dict[str, Any] | None,
    *,
    days: int = 7,
    until: datetime | None = None,
) -> DriftReport:
    until = until or datetime.now(UTC)
    since = until - timedelta(days=days)
    preds = list(
        session.scalars(
            select(Prediction).where(
                Prediction.model_version_id == model_version_id,
                Prediction.created_at >= since,
                Prediction.created_at <= until,
            )
        )
    )
    n = len(preds)
    gated = [p for p in preds if p.gate_status != "ok"]
    ok = [p for p in preds if p.gate_status == "ok"]
    rep = DriftReport(
        model_version_id=model_version_id,
        window_start=since,
        window_end=until,
        n_predictions=n,
        n_gated_out=len(gated),
        gate_out_rate=(len(gated) / n) if n else None,
        reference_gate_out_rate=(reference or {}).get("gate_out_rate"),
    )
    ref = reference or {}
    for k, r in ref.get("input_stats", {}).items():
        live = np.array(
            [p.input_stats[k] for p in preds if k in (p.input_stats or {})], dtype=np.float64
        )
        rep.input_psi[k] = psi(r["edges"], r["counts"], live)
    for code, r in ref.get("outputs", {}).items():
        live = np.array(
            [f["probability"] for p in ok for f in p.findings if f["code"] == code],
            dtype=np.float64,
        )
        rep.output_psi[code] = psi(r["edges"], r["counts"], live)
        rep.output_mean[code] = float(live.mean()) if len(live) else None
        rep.reference_output_mean[code] = float(r["mean"])
        pres = [f["present"] for p in ok for f in p.findings if f["code"] == code]
        rep.positive_rate[code] = (sum(pres) / len(pres)) if pres else None
    if preds:
        ids = [p.id for p in preds]
        rows = session.execute(
            select(Feedback.finding_code, Feedback.agreement, func.count())
            .where(Feedback.prediction_id.in_(ids))
            .group_by(Feedback.finding_code, Feedback.agreement)
        ).all()
        for code, agreement, cnt in rows:
            rep.feedback.setdefault(code, {a.value: 0 for a in Agreement})[
                Agreement(agreement).value
            ] = int(cnt)
    for k, v in {
        **{f"input:{k}": v for k, v in rep.input_psi.items()},
        **{f"output:{k}": v for k, v in rep.output_psi.items()},
    }.items():
        if v is not None and v >= PSI_ALERT:
            rep.flags.append(f"{k} PSI={v:.3f} ≥ {PSI_ALERT} (drift)")
        elif v is not None and v >= PSI_WARN:
            rep.flags.append(f"{k} PSI={v:.3f} ≥ {PSI_WARN} (watch)")
    if (
        rep.gate_out_rate is not None
        and rep.reference_gate_out_rate is not None
        and rep.gate_out_rate > max(0.1, 2 * rep.reference_gate_out_rate)
    ):
        rep.flags.append(
            f"gate-out rate {rep.gate_out_rate:.1%} vs reference {rep.reference_gate_out_rate:.1%}"
        )
    for code, fb in rep.feedback.items():
        tot = sum(fb.values())
        if tot >= 20 and fb.get("disagree", 0) / tot > 0.3:
            rep.flags.append(f"{code}: clinicians disagree with {fb['disagree']}/{tot} findings")
    return rep


def render_drift_markdown(rep: DriftReport) -> str:
    def f(v: float | None) -> str:
        return "—" if v is None else f"{v:.3f}"

    lines = [
        f"# Drift report — model `{rep.model_version_id}`",
        "",
        f"> **{RESEARCH_USE_LABEL}**",
        "",
        f"Window: {rep.window_start:%Y-%m-%d} → {rep.window_end:%Y-%m-%d} · "
        f"predictions: {rep.n_predictions} · gated out: {rep.n_gated_out} "
        f"({f(rep.gate_out_rate)} vs reference {f(rep.reference_gate_out_rate)})",
        "",
    ]
    lines += ["## Flags", ""] + ([f"- ⚠️ {x}" for x in rep.flags] or ["- none"]) + [""]
    lines += [
        "## Input statistics (PSI vs validation reference)",
        "",
        "| statistic | PSI |",
        "|---|---|",
    ]
    lines += [f"| {k} | {f(v)} |" for k, v in rep.input_psi.items()]
    lines += [
        "",
        "## Output distributions",
        "",
        "| finding | PSI | live mean p | reference mean p | positive rate |",
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| {c} | {f(rep.output_psi.get(c))} | {f(rep.output_mean.get(c))} | "
        f"{f(rep.reference_output_mean.get(c))} | {f(rep.positive_rate.get(c))} |"
        for c in rep.output_psi
    ]
    lines += [
        "",
        "## Clinician feedback",
        "",
        "| finding | agree | disagree | unsure |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| {c} | {fb.get('agree', 0)} | {fb.get('disagree', 0)} | {fb.get('unsure', 0)} |"
        for c, fb in rep.feedback.items()
    ] or ["| — | — | — | — |"]
    lines += ["", f"---\n{RESEARCH_USE_LABEL}"]
    return "\n".join(lines)
