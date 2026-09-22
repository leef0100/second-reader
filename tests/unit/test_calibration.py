from __future__ import annotations

import numpy as np
import pytest

from secondreader.core.eval import expected_calibration_error, fit_platt, fit_temperature
from secondreader.modalities.cxr.model import Calibration, normalize_view
from secondreader.modalities.cxr.train import fit_calibration

LEGACY_KEYS = {"temperatures", "biases", "thresholds", "operating_point", "finding_codes", "method"}


def _shifted_logits(n=6000, shift=2.5, scale=1.7, seed=0):
    """Well-calibrated latent logits, distorted by a scale and an offset (as pos_weight does)."""
    rng = np.random.default_rng(seed)
    z_true = rng.normal(0, 2.0, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-z_true))).astype(float)
    return z_true * scale + shift, y


def test_platt_recovers_scale_and_bias_where_temperature_cannot():
    z, y = _shifted_logits()
    t_only = fit_temperature(z, y)
    p_temp = 1 / (1 + np.exp(-z / t_only))
    t, b = fit_platt(z, y)
    p_platt = 1 / (1 + np.exp(-(z / t + b)))
    ece_temp = expected_calibration_error(p_temp, y)[0]
    ece_platt = expected_calibration_error(p_platt, y)[0]
    assert abs(t - 1.7) < 0.2 and abs(b + 2.5 / 1.7) < 0.2
    assert ece_platt < 0.03
    assert ece_temp > 0.15  # temperature alone cannot remove the offset


def test_platt_degenerate_inputs_are_bounded():
    assert fit_platt(np.array([1.0, 2.0]), np.array([1.0, 1.0])) == (1.0, 0.0)
    y = np.array([0.0] * 50 + [1.0] * 50)
    z = (y * 2 - 1) * 50  # perfectly separable
    t, b = fit_platt(z, y)
    assert 0.25 <= t <= 4.0 and -6 <= b <= 6


def test_calibration_apply_and_roundtrip():
    c = Calibration([2.0, 1.0], [0.5, 0.5], "youden", ["a", "b"], biases=[1.0, 0.0], method="platt")
    p = c.apply(np.array([[0.0, 0.0]]))
    assert p[0, 0] > 0.7 and abs(p[0, 1] - 0.5) < 1e-9
    d = c.as_dict()
    assert Calibration.from_dict(d) == c
    legacy = Calibration.from_dict(
        {
            "temperatures": [1.0],
            "thresholds": [0.5],
            "operating_point": "youden",
            "finding_codes": ["a"],
        }
    )
    assert legacy.biases == [0.0] and legacy.method == "temperature"
    assert legacy.thresholds_by_view == {} and legacy.thresholds_for_view("PA") == [0.5]
    # a calibration without per-view thresholds serialises exactly as before (same hash)
    assert set(c.as_dict()) == LEGACY_KEYS


def test_per_view_thresholds_roundtrip_and_fallback():
    c = Calibration(
        [1.0, 1.0],
        [0.5, 0.4],
        "youden",
        ["a", "b"],
        method="platt",
        thresholds_by_view={"pa": [0.6, 0.3], "AP": [0.2, 0.1]},
    )
    assert set(c.thresholds_by_view) == {"PA", "AP"}  # keys normalised like ViewPosition
    assert c.thresholds_for_view("PA") == [0.6, 0.3] and c.thresholds_for_view("ap") == [0.2, 0.1]
    for unknown in (None, "", "LL", float("nan"), "  "):
        assert c.thresholds_for_view(unknown) == [0.5, 0.4], unknown
    assert c.threshold_for(1, "AP") == 0.1 and c.threshold_for(0, None) == 0.5
    rows = c.row_thresholds(["PA", "AP", None, "LL"])
    assert rows.shape == (4, 2) and rows[0, 0] == 0.6 and rows[2, 1] == 0.4 and rows[3, 0] == 0.5
    assert c.row_thresholds([]).shape == (0, 2)
    d = c.as_dict()
    assert set(d) == LEGACY_KEYS | {"thresholds_by_view"}
    assert Calibration.from_dict(d) == c
    with pytest.raises(ValueError, match="thresholds_by_view"):
        Calibration([1.0], [0.5], "youden", ["a"], thresholds_by_view={"PA": [0.1, 0.2]})
    assert normalize_view(" pa ") == "PA" and normalize_view(None) == ""


def _view_shifted_val(n=4000, seed=0):
    """Validation set where AP positives score lower than PA positives (as NIH does)."""
    rng = np.random.default_rng(seed)
    views = np.where(rng.random(n) < 0.35, "AP", "PA").astype(object)
    y = (rng.random((n, 2)) < 0.2).astype(float)
    z = rng.normal(0, 1.0, (n, 2)) + 2.5 * y
    z[views == "AP"] -= 1.5  # AP films: everything looks less positive
    return z, y, views


def test_fit_calibration_per_view_moves_only_the_threshold():
    z, y, views = _view_shifted_val()
    cal_g, summ_g = fit_calibration(z, y, ["a", "b"], "platt", "youden")
    cal_v, summ_v = fit_calibration(z, y, ["a", "b"], "platt", "youden", views=views)
    assert cal_g.thresholds_by_view == {} and "by_view" not in summ_g["a"]
    assert set(cal_v.thresholds_by_view) == {"AP", "PA"}
    assert cal_v.temperatures == cal_g.temperatures and cal_v.biases == cal_g.biases
    assert cal_v.thresholds == cal_g.thresholds  # global threshold unchanged
    for j in range(2):
        assert cal_v.thresholds_by_view["AP"][j] < cal_v.thresholds_by_view["PA"][j]
    bv = summ_v["a"]["by_view"]["AP"]
    assert bv["fitted"] and bv["n_pos"] >= 20
    # at the AP-specific threshold the AP sensitivity recovers relative to the global one
    assert bv["sensitivity"] > bv["sensitivity_at_global"]
    # a view with too few positives is reported (global threshold, unfitted) but not stored
    views_rare = views.copy()
    views_rare[np.flatnonzero(views == "AP")[:-5]] = "PA"
    cal_r, summ_r = fit_calibration(z, y, ["a", "b"], "platt", "youden", views=views_rare)
    assert set(cal_r.thresholds_by_view) == {"PA"}
    assert summ_r["a"]["by_view"]["AP"]["fitted"] is False
    assert summ_r["a"]["by_view"]["AP"]["threshold"] == cal_r.thresholds[0]
    # unknown/blank views are ignored as keys, and if nothing can be fitted nothing is stored
    cal_none, _ = fit_calibration(
        z, y, ["a", "b"], "platt", "youden", views=np.array(["", None] * (len(z) // 2))
    )
    assert cal_none.thresholds_by_view == {}
