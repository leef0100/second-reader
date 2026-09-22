from __future__ import annotations

import numpy as np
import pytest

from secondreader.core.eval import (
    BinaryMetrics,
    EvalReport,
    SubgroupRow,
    adaptive_calibration_error,
    binary_metrics,
    expected_calibration_error,
    fit_temperature,
    render_markdown,
    sensitivity_target_threshold,
    youden_threshold,
)


def test_ece_perfect_and_worst():
    y = np.array([0, 0, 1, 1], dtype=float)
    assert expected_calibration_error(np.array([0.0, 0.0, 1.0, 1.0]), y)[0] == 0.0
    ece, bins = expected_calibration_error(np.array([1.0, 1.0, 0.0, 0.0]), y)
    assert ece == pytest.approx(1.0)
    assert bins.bin_count.sum() == 4


def test_ece_known_value():
    # two bins: 4 items at p=0.2 with 1 positive (gap .05), 4 at p=0.8 with 4 positives (gap .2)
    p = np.array([0.2] * 4 + [0.8] * 4)
    y = np.array([1, 0, 0, 0, 1, 1, 1, 1], dtype=float)
    ece, _ = expected_calibration_error(p, y, n_bins=5)
    assert ece == pytest.approx(0.5 * 0.05 + 0.5 * 0.2)


def test_binary_metrics_and_ci():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400).astype(float)
    p = np.clip(y * 0.6 + rng.normal(0.2, 0.2, 400), 0, 1)
    m = binary_metrics(p, y, 0.5, n_boot=50)
    assert m.auroc is not None and m.auroc > 0.9
    assert m.sensitivity is not None and m.specificity is not None
    assert "auroc" in m.ci and m.ci["auroc"][0] <= m.auroc <= m.ci["auroc"][1]
    assert m.low_n is False and m.n_pos == int(y.sum())
    # R11: equal-mass ECE and Brier travel with the metrics, with CIs and in the JSON
    assert m.ece_equal_mass is not None and m.brier is not None
    assert {"ece_equal_mass", "brier"} <= set(m.ci)
    assert m.as_dict()["ece_equal_mass"] == m.ece_equal_mass


def test_adaptive_ece_known_value_and_tie_handling():
    # two score levels → two equal-mass bins whatever n_bins asks for (ties stay together)
    p = np.array([0.1] * 4 + [0.9] * 4)
    y = np.array([0, 0, 0, 1, 1, 1, 1, 0], dtype=float)
    expected = 0.5 * abs(0.25 - 0.1) + 0.5 * abs(0.75 - 0.9)
    assert adaptive_calibration_error(p, y, n_bins=2) == pytest.approx(expected)
    assert adaptive_calibration_error(p, y, n_bins=15) == pytest.approx(expected)
    assert adaptive_calibration_error(np.array([]), np.array([])) == 0.0
    assert adaptive_calibration_error(np.array([0.0, 1.0]), np.array([0.0, 1.0])) == 0.0


def test_adaptive_ece_sees_miscalibration_equal_width_hides_at_low_prevalence():
    # 5 % prevalence, scores squeezed into [0, 0.06]: equal-width puts everything in bin 0
    # (one gap, mostly cancelled), equal-mass bins resolve the over-confident top decile.
    rng = np.random.default_rng(3)
    n = 20000
    p = rng.uniform(0.0, 0.06, n)
    y = (rng.random(n) < np.where(p > 0.05, 0.5, 0.02)).astype(float)
    ece_w, _ = expected_calibration_error(p, y)
    ece_m = adaptive_calibration_error(p, y)
    assert ece_m > ece_w


def test_binary_metrics_per_row_thresholds():
    p = np.array([0.2, 0.4, 0.6, 0.8, 0.3, 0.7])
    y = np.array([0, 1, 0, 1, 1, 0], dtype=float)
    thr = np.array([0.5, 0.5, 0.5, 0.5, 0.25, 0.75])  # last two rows: "other view"
    m = binary_metrics(p, y, thr, n_boot=0)
    # predictions: F, F, T, T, T, F → tp=2 (0.8, 0.3), fn=1 (0.4), tn=2, fp=1
    assert m.sensitivity == pytest.approx(2 / 3) and m.specificity == pytest.approx(2 / 3)
    assert np.isnan(m.threshold)  # mixed thresholds → no single operating point to report
    uniform = binary_metrics(p, y, np.full(6, 0.5), n_boot=0)
    assert uniform.threshold == 0.5
    assert uniform.sensitivity == binary_metrics(p, y, 0.5, n_boot=0).sensitivity
    with pytest.raises(ValueError, match="per-row thresholds"):
        binary_metrics(p, y, np.array([0.5, 0.5]), n_boot=0)
    # bootstrap resamples thresholds together with the rows
    m_boot = binary_metrics(np.tile(p, 10), np.tile(y, 10), np.tile(thr, 10), n_boot=20)
    assert "sensitivity" in m_boot.ci


def test_binary_metrics_degenerate():
    m = binary_metrics(np.array([0.1, 0.9]), np.array([0.0, 0.0]), 0.5, n_boot=0)
    assert m.auroc is None and m.sensitivity is None and m.specificity == 0.5 and m.low_n


def test_thresholds():
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    y = np.array([0, 0, 0, 1, 1, 1], dtype=float)
    t = youden_threshold(p, y)
    assert 0.3 < t < 0.7  # midpoint, not on top of the lowest positive
    assert sensitivity_target_threshold(p, y, 0.9) <= 0.7


def test_temperature_scaling_reduces_overconfidence_and_is_bounded():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 2000).astype(float)
    z = (y * 2 - 1) * 1.0 + rng.normal(0, 1.0, 2000)
    t_over = fit_temperature(z * 4, y)  # overconfident logits → T > 1
    assert t_over > 1.5
    z_sep = (y * 2 - 1) * 10.0  # perfectly separable → would collapse; bounded instead
    assert fit_temperature(z_sep, y) >= 0.25


def _bm(n=200, n_pos=50):
    return BinaryMetrics(n, n_pos, 0.9, 0.8, 0.85, 0.5, 0.9, 0.05, 0.1, 0.5, low_n=n_pos < 100)


def test_report_requires_all_subgroup_axes():
    from secondreader.core.eval.metrics import expected_calibration_error as ece

    _, rb = ece(np.array([0.2, 0.8]), np.array([0.0, 1.0]))
    base = dict(
        title="t",
        model_version_id="m",
        dataset_name="d",
        split_manifest_sha256="s",
        tier="external",
        finding_codes=["pneumothorax"],
        overall={"pneumothorax": _bm()},
        reliability={"pneumothorax": rb},
    )
    incomplete = EvalReport(subgroups=[SubgroupRow("source", "a", {"pneumothorax": _bm()})], **base)
    with pytest.raises(ValueError, match="subgroup axes"):
        incomplete.validate()
    rows = [
        SubgroupRow(ax, "unavailable" if ax == "vendor" else "g", {"pneumothorax": _bm(n_pos=10)})
        for ax in ("source", "vendor", "age_band", "sex", "view")
    ]
    full = EvalReport(subgroups=rows, **base)
    md = render_markdown(full)
    assert "For research use only. Not for diagnostic use." in md
    assert "### by vendor" in md and "unavailable" in md and "low-n" in md
    assert "ECE eq-width | ECE eq-mass | Brier" in md and "pneumothorax=0.500" in md


def test_report_renders_per_view_threshold_and_missing_equal_mass():
    _, rb = expected_calibration_error(np.array([0.2, 0.8]), np.array([0.0, 1.0]))
    mixed = BinaryMetrics(200, 50, 0.9, 0.8, 0.85, 0.5, 0.9, 0.05, 0.1, float("nan"), low_n=False)
    rows = [SubgroupRow(ax, "g", {"pneumothorax": mixed}) for ax in ("source", "vendor")]
    rows += [SubgroupRow(ax, "g", {"pneumothorax": mixed}) for ax in ("age_band", "sex", "view")]
    rep = EvalReport(
        title="t",
        model_version_id="m",
        dataset_name="d",
        split_manifest_sha256="s",
        tier="test",
        finding_codes=["pneumothorax"],
        overall={"pneumothorax": mixed},
        subgroups=rows,
        reliability={"pneumothorax": rb},
    )
    md = render_markdown(rep)
    assert "pneumothorax=per-view" in md
    # ece_equal_mass defaulted to None (older BinaryMetrics) → rendered as a dash, not "None"
    assert "| 0.050 | — | 0.100 |" in md
