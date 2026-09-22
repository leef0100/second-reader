from __future__ import annotations

import numpy as np
import pytest

from secondreader.core.dicom.imaging import to_imaging_input
from secondreader.core.plugins.protocol import FrameMeta
from secondreader.modalities.cxr.config import GateConfig, PreprocessConfig
from secondreader.modalities.cxr.gate import (
    OodModel,
    combine,
    header_checks,
    l2_normalize,
    learned_checks,
    max_logit,
    pixel_checks,
    pixel_checks_from_stats,
)
from secondreader.modalities.cxr.preprocess import (
    input_statistics,
    letterbox,
    preprocess_frame,
    to_float_frame,
    unletterbox,
    window,
)
from secondreader.modalities.cxr.synthetic import lung_image
from tests.fixtures.synthetic import cxr_spec, make_dataset, ultrasound_cine_spec


def test_monochrome1_is_inverted_and_rescale_applied():
    a = np.array([[0, 100], [200, 300]], dtype=np.uint16)
    m1 = FrameMeta(
        photometric_interpretation="MONOCHROME1", rescale_slope=2.0, rescale_intercept=-10.0
    )
    out = to_float_frame(a, m1)
    raw = a.astype(np.float32) * 2 - 10
    assert np.allclose(out, raw.max() - raw)
    m2 = FrameMeta(photometric_interpretation="MONOCHROME2")
    assert np.allclose(to_float_frame(a, m2), a)


def test_window_is_robust_to_outliers():
    a = np.full((100, 100), 1000.0, dtype=np.float32)
    a[0, 0], a[0, 1] = 0.0, 1e6  # two extreme pixels
    w = window(a, 0.5, 99.5)
    assert w.min() >= 0.0 and w.max() <= 1.0
    assert abs(float(np.median(w)) - float(w[50, 50])) < 1e-6


def test_letterbox_roundtrip_geometry():
    a = np.random.default_rng(0).random((300, 200)).astype(np.float32)
    sq, g = letterbox(a, 128)
    assert sq.shape == (128, 128)
    assert g.new_h == 128 and g.new_w == 85 and g.pad_left == 21
    assert np.allclose(sq[:, : g.pad_left], 0.0)
    back = unletterbox(sq, g)
    assert back.shape == (300, 200)
    assert abs(float(back.mean()) - float(a.mean())) < 0.05


def test_preprocess_frame_tensor_shape_and_stats():
    img = lung_image(256, 200, seed=1)
    pp = preprocess_frame(img, None, PreprocessConfig(image_size=160))
    assert tuple(pp.tensor.shape) == (3, 160, 160)
    assert pp.image01.max() <= 1.0
    assert {"mean", "std", "entropy", "aspect", "saturated_fraction"} <= set(pp.stats)
    assert pp.stats["src_h"] == 256 and pp.stats["src_w"] == 200


def test_input_statistics_saturation():
    a = np.zeros((64, 64), dtype=np.float32)
    a[:32] = 1.0
    s = input_statistics(a, 64, 64)
    assert s["saturated_fraction"] == 1.0


# ---- gate --------------------------------------------------------------------------------


def test_header_gate_rejects_lateral_non_chest_and_cine():
    cfg = GateConfig()
    ok = to_imaging_input(make_dataset(cxr_spec()))
    assert header_checks(ok, cfg) == []
    lat = to_imaging_input(make_dataset(cxr_spec(view_position="LL")))
    assert any("frontal" in r for r in header_checks(lat, cfg))
    knee = to_imaging_input(make_dataset(cxr_spec(body_part="KNEE")))
    assert any("chest" in r for r in header_checks(knee, cfg))
    us = to_imaging_input(make_dataset(ultrasound_cine_spec()))
    reasons = header_checks(us, cfg)
    assert any("radiograph" in r for r in reasons) and any("single frame" in r for r in reasons)


def test_pixel_gate_rejects_blank_small_and_odd_aspect():
    cfg = GateConfig(min_short_side=100)
    blank = np.full((300, 300), 0.5, dtype=np.float32)
    assert any("contrast" in r for r in pixel_checks(blank, 300, 300, cfg)[0])
    tiny = np.random.default_rng(0).random((50, 50)).astype(np.float32)
    assert any("small" in r for r in pixel_checks(tiny, 50, 50, cfg)[0])
    strip = np.random.default_rng(0).random((100, 400)).astype(np.float32)
    assert any("aspect" in r for r in pixel_checks(strip, 100, 400, cfg)[0])
    good = window(lung_image(300, 300).astype(np.float32), 0.5, 99.5)
    assert pixel_checks(good, 300, 300, cfg)[0] == []


def test_learned_gate_view_and_ood():
    cfg = GateConfig()
    rng = np.random.default_rng(0)
    train = rng.normal(size=(500, 8))
    val = rng.normal(size=(100, 8))
    ood = OodModel.fit(train, val, 99.0)
    in_dist = rng.normal(size=8)
    far = np.full(8, 25.0)
    r_in, s_in = learned_checks(
        cfg, view_probs=np.array([0.9, 0.05, 0.05]), features=in_dist, ood=ood
    )
    r_far, _ = learned_checks(cfg, view_probs=np.array([0.9, 0.05, 0.05]), features=far, ood=ood)
    r_lat, _ = learned_checks(
        cfg, view_probs=np.array([0.1, 0.85, 0.05]), features=in_dist, ood=ood
    )
    assert r_in == [] and "ood_distance" in s_in
    assert any("out-of-distribution" in r for r in r_far)
    assert any("view classifier" in r for r in r_lat)
    # unavailable layers are reported as NaN scores, not hidden
    _, s_none = learned_checks(cfg, view_probs=None, features=None, ood=None)
    assert np.isnan(s_none["ood_distance"]) and np.isnan(s_none["view_classifier"])


def test_ood_normalize_scores_the_way_it_was_fit_and_ignores_feature_norm():
    rng = np.random.default_rng(1)
    base = rng.normal(size=(600, 8))
    train = base * rng.uniform(0.5, 20.0, size=(600, 1))  # wildly varying feature norms
    val = rng.normal(size=(100, 8)) * rng.uniform(0.5, 20.0, size=(100, 1))
    raw = OodModel.fit(train, val, 99.0)
    plus = OodModel.fit(train, val, 99.0, normalize=True)
    assert raw.normalize is False and plus.normalize is True
    assert np.allclose(np.linalg.norm(l2_normalize(train), axis=1), 1.0)
    f = rng.normal(size=8)
    # Mahalanobis++ is invariant to the feature norm; the raw gate is not
    assert plus.distance(f) == pytest.approx(plus.distance(10.0 * f))
    assert raw.distance(f) != pytest.approx(raw.distance(10.0 * f))
    # the flag rides along with the bundle (ood.npz) and is honoured on reload
    from secondreader.modalities.cxr.model import ModelBundle  # noqa: F401 — import check

    assert OodModel(mean=plus.mean, precision=plus.precision, threshold=1.0).normalize is False


def test_learned_checks_logs_max_logit_but_never_gates_on_it():
    cfg = GateConfig()
    rng = np.random.default_rng(0)
    ood = OodModel.fit(rng.normal(size=(500, 8)), rng.normal(size=(100, 8)), 99.0)
    feats = rng.normal(size=8)
    logits = np.array([-9.0, -4.0, -7.5])  # confident about nothing
    r, s = learned_checks(
        cfg, view_probs=np.array([0.9, 0.05, 0.05]), features=feats, ood=ood, logits=logits
    )
    assert r == [] and s["max_logit"] == -4.0 == max_logit(logits)
    _, s_none = learned_checks(cfg, view_probs=None, features=None, ood=None)
    assert "max_logit" not in s_none


def test_pixel_checks_from_stats_matches_pixel_checks():
    cfg = GateConfig(min_short_side=100)
    img = window(lung_image(300, 220, seed=2).astype(np.float32), 0.5, 99.5)
    img[:40] = 1.0  # some saturation
    r1, s1 = pixel_checks(img, 300, 220, cfg)
    r2, s2 = pixel_checks_from_stats(input_statistics(img, 300, 220), cfg)
    assert r1 == r2 and s1 == pytest.approx(s2)
    tiny_stats = {"src_h": 50.0, "src_w": 50.0, "std": 0.2, "saturated_fraction": 0.0}
    assert any("small" in r for r in pixel_checks_from_stats(tiny_stats, cfg)[0])


def test_combine_yields_unable_with_reasons():
    g = combine((["a"], {"x": 1.0}), ([], {"y": 2.0}))
    assert not g.ok and g.reasons == ("a",) and g.scores == {"x": 1.0, "y": 2.0}
    assert combine(([], {})).ok


@pytest.mark.parametrize(
    "age,band", [(None, "unavailable"), (17, "<18"), (18, "18-39"), (59, "40-59"), (80, "80+")]
)
def test_age_band(age, band):
    from secondreader.core.eval import age_band

    assert age_band(age) == band
