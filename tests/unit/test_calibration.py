from __future__ import annotations

import numpy as np

from secondreader.core.eval import expected_calibration_error, fit_platt, fit_temperature
from secondreader.modalities.cxr.model import Calibration


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
