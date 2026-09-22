"""Evaluation report schema + markdown/plot writer.

`EvalReport` REQUIRES per-subgroup tables (D-010). A report with a missing
subgroup axis fails validation; an axis whose metadata is absent must still be
present with an explicit "unavailable" row.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from secondreader.core.constants import RESEARCH_USE_LABEL
from secondreader.core.eval.metrics import BinaryMetrics, ReliabilityBins

REQUIRED_SUBGROUP_AXES: tuple[str, ...] = ("source", "vendor", "age_band", "sex", "view")
AGE_BANDS: tuple[tuple[str, int, int], ...] = (
    ("18-39", 18, 39),
    ("40-59", 40, 59),
    ("60-79", 60, 79),
    ("80+", 80, 200),
)


def age_band(age: float | None) -> str:
    if age is None or (isinstance(age, float) and np.isnan(age)):
        return "unavailable"
    for name, lo, hi in AGE_BANDS:
        if lo <= age <= hi:
            return name
    return "<18" if age < 18 else "unavailable"


@dataclass
class SubgroupRow:
    axis: str
    group: str
    metrics: dict[str, BinaryMetrics]  # finding_code → metrics


@dataclass
class EvalReport:
    title: str
    model_version_id: str
    dataset_name: str
    split_manifest_sha256: str
    tier: str
    finding_codes: list[str]
    overall: dict[str, BinaryMetrics]
    subgroups: list[SubgroupRow]
    reliability: dict[str, ReliabilityBins]
    n_gated_out: int = 0
    gate_reasons: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        axes = {r.axis for r in self.subgroups}
        missing = [a for a in REQUIRED_SUBGROUP_AXES if a not in axes]
        if missing:
            raise ValueError(f"EvalReport missing required subgroup axes: {missing}")
        for code in self.finding_codes:
            if code not in self.overall:
                raise ValueError(f"overall metrics missing for {code}")
            if code not in self.reliability:
                raise ValueError(f"reliability data missing for {code}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "research_use": RESEARCH_USE_LABEL,
            "title": self.title,
            "model_version_id": self.model_version_id,
            "dataset_name": self.dataset_name,
            "split_manifest_sha256": self.split_manifest_sha256,
            "tier": self.tier,
            "finding_codes": self.finding_codes,
            "overall": {k: v.as_dict() for k, v in self.overall.items()},
            "subgroups": [
                {
                    "axis": r.axis,
                    "group": r.group,
                    "metrics": {k: v.as_dict() for k, v in r.metrics.items()},
                }
                for r in self.subgroups
            ],
            "n_gated_out": self.n_gated_out,
            "gate_reasons": self.gate_reasons,
            "notes": self.notes,
            "extra": self.extra,
        }


def _fmt(v: float | None, ci: tuple[float, float] | None = None) -> str:
    if v is None or np.isnan(v):
        return "—"
    s = f"{v:.3f}"
    if ci:
        s += f" [{ci[0]:.3f}–{ci[1]:.3f}]"
    return s


def _metrics_table(rows: list[tuple[str, dict[str, BinaryMetrics]]], codes: list[str]) -> str:
    out = [
        "| group | finding | n | n_pos | AUROC [95% CI] | sens [95% CI] | spec [95% CI] | "
        "ECE eq-width | ECE eq-mass | Brier | flag |",
        "|---|---|---:|---:|---|---|---|---|---|---|---|",
    ]
    for name, m in rows:
        for c in codes:
            bm = m.get(c)
            if bm is None:
                out.append(f"| {name} | {c} | — | — | — | — | — | — | — | — | unavailable |")
                continue
            flag = "low-n" if bm.low_n else ""
            out.append(
                f"| {name} | {c} | {bm.n} | {bm.n_pos} | {_fmt(bm.auroc, bm.ci.get('auroc'))} | "
                f"{_fmt(bm.sensitivity, bm.ci.get('sensitivity'))} | "
                f"{_fmt(bm.specificity, bm.ci.get('specificity'))} | {_fmt(bm.ece)} | "
                f"{_fmt(bm.ece_equal_mass)} | {_fmt(bm.brier)} | {flag} |"
            )
    return "\n".join(out)


def write_reliability_plots(report: EvalReport, out_dir: Path) -> dict[str, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for code, rb in report.reliability.items():
        fig, ax = plt.subplots(figsize=(4, 4), dpi=110)
        centers = (rb.bin_edges[:-1] + rb.bin_edges[1:]) / 2
        m = ~np.isnan(rb.bin_accuracy)
        ax.plot([0, 1], [0, 1], "--", color="#888", lw=1, label="perfect")
        ax.bar(
            centers[m],
            rb.bin_accuracy[m],
            width=1 / len(centers),
            alpha=0.6,
            edgecolor="k",
            label="observed",
        )
        ax.plot(rb.bin_confidence[m], rb.bin_accuracy[m], "o-", color="#c0392b", ms=3)
        ax.set_xlabel("predicted probability")
        ax.set_ylabel("observed frequency")
        ece = report.overall[code].ece
        ax.set_title(f"{code} — ECE {ece:.3f}" if ece is not None else code, fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7, loc="upper left")
        fig.text(0.5, 0.005, RESEARCH_USE_LABEL, ha="center", fontsize=6, color="#a00")
        fig.tight_layout()
        p = out_dir / f"reliability_{code}.png"
        fig.savefig(p)
        plt.close(fig)
        paths[code] = p
    return paths


def render_markdown(report: EvalReport, plot_paths: dict[str, Path] | None = None) -> str:
    report.validate()
    codes = report.finding_codes
    lines = [
        f"# {report.title}",
        "",
        f"> **{RESEARCH_USE_LABEL}**",
        "",
        f"- Model version: `{report.model_version_id}`",
        f"- Dataset: `{report.dataset_name}` · tier: `{report.tier}`",
        f"- Split manifest sha256: `{report.split_manifest_sha256}`",
        f"- Gated out (unable to assess): {report.n_gated_out}"
        + (f" — {report.gate_reasons}" if report.gate_reasons else ""),
        "",
        "## Overall",
        "",
        _metrics_table([("all", report.overall)], codes),
        "",
        "Operating thresholds: "
        + ", ".join(
            f"{c}={report.overall[c].threshold:.3f}"
            if not np.isnan(report.overall[c].threshold)
            else f"{c}=per-view (see notes)"
            for c in codes
            if c in report.overall
        ),
        "",
        "ECE eq-width = 15 equal-width bins (literature-comparable; biased at low prevalence). "
        "ECE eq-mass = 15 equal-mass bins (adaptive). Brier = proper scoring rule, binning-free.",
        "",
    ]
    if plot_paths:
        lines.append("## Calibration (reliability diagrams)")
        lines.append("")
        for c in codes:
            if c in plot_paths:
                lines.append(f"![{c}]({plot_paths[c].name})")
        lines.append("")
    lines.append("## Per-subgroup performance")
    lines.append("")
    lines.append(
        "Subgroup metrics are mandatory (D-010). `low-n` = fewer than 100 positives; "
        "treat those estimates as unreliable. `unavailable` = metadata absent in this dataset."
    )
    lines.append("")
    for axis in REQUIRED_SUBGROUP_AXES:
        rows = [(r.group, r.metrics) for r in report.subgroups if r.axis == axis]
        lines.append(f"### by {axis}")
        lines.append("")
        lines.append(_metrics_table(rows, codes) if rows else "_unavailable_")
        lines.append("")
    if report.notes:
        lines.append("## Notes")
        lines.append("")
        lines.extend(f"- {n}" for n in report.notes)
        lines.append("")
    lines.append(f"---\n{RESEARCH_USE_LABEL}")
    return "\n".join(lines)
