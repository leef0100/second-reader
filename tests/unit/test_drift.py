from __future__ import annotations

import numpy as np

from secondreader.core.monitoring import build_reference, psi


def test_psi_zero_for_same_distribution_and_large_for_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    r = build_reference([{"x": float(v)} for v in ref], np.zeros((10, 1)), ["f"], 0.0)
    edges, counts = r["input_stats"]["x"]["edges"], r["input_stats"]["x"]["counts"]
    same = psi(edges, counts, rng.normal(0, 1, 5000))
    shifted = psi(edges, counts, rng.normal(1.5, 1, 5000))
    assert same is not None and same < 0.05
    assert shifted is not None and shifted > 0.5
    assert psi(edges, counts, np.array([])) is None


def test_build_reference_shapes():
    probs = np.random.default_rng(1).random((100, 2))
    r = build_reference([{"mean": 0.5, "std": 0.1}] * 100, probs, ["a", "b"], 0.1)
    assert set(r["outputs"]) == {"a", "b"} and len(r["outputs"]["a"]["edges"]) == 11
    assert r["input_stats"]["mean"]["mean"] == 0.5 and r["gate_out_rate"] == 0.1
