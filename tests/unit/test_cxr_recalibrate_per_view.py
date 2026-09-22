"""`recalibrate --per-view` → new bundle → `evaluate --cache` scores every row at its own
view's threshold (D-029). Fast: random-init resnet18 at 32 px on the synthetic set — checks
the plumbing (per-view thresholds, cache-backed gate, report contents), not accuracy."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from secondreader.modalities.cxr.cache import FrameCache, build_cache
from secondreader.modalities.cxr.config import CxrConfig
from secondreader.modalities.cxr.data import manifest_sha256
from secondreader.modalities.cxr.evaluate import evaluate
from secondreader.modalities.cxr.gate import OodModel
from secondreader.modalities.cxr.model import Calibration, CxrNet, ModelBundle, save_bundle
from secondreader.modalities.cxr.splits import build_splits, load_tier
from secondreader.modalities.cxr.synthetic import generate
from secondreader.modalities.cxr.train import recalibrate


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("perview")
    df = generate(root / "a", n_patients=300, size=32, seed=0, dataset="synth-a")
    # a big validation tier so each view has > 20 positives per finding
    s = build_splits(
        df,
        out_dir=root / "splits",
        version="t",
        salt="s",
        external_datasets=set(),
        train_pct=10,
        val_pct=80,
    )
    cfg = CxrConfig(
        name="unit",
        preprocess={"image_size": 32},
        # random-init net: its OOD/view heads would gate everything, so only header + pixel
        # checks run here (laterals are still gated by the header)
        gate={
            "min_short_side": 16,
            "ood_normalize": True,
            "ood_enabled": False,
            "view_classifier_enabled": False,
        },
        model={"backbone": "resnet18", "pretrained": False},
        train={"num_workers": 0},
    )
    torch.manual_seed(0)
    net = CxrNet(cfg, 3)
    rng = np.random.default_rng(0)
    d = net.backbone.num_features
    ood = OodModel.fit(rng.normal(size=(300, d)), rng.normal(size=(50, d)), 99.0, normalize=True)
    mid = save_bundle(
        root / "models" / "pending",
        net=net,
        cfg=cfg,
        calibration=Calibration([1.0] * 3, [0.5] * 3, "youden", list(cfg.finding_codes)),
        ood=ood,
        split_manifest_sha256=s.manifest_sha256,
        metrics={},
        datasets=[],
    )
    split_dir = root / "splits" / "t"
    val = load_tier(split_dir, "val")
    build_cache(
        val,
        manifest_sha256=manifest_sha256(split_dir / "val.csv"),
        pre=cfg.preprocess,
        out_dir=root / "cache" / "val",
        workers=1,
    )
    return root, split_dir, mid, s.manifest_sha256


def test_recalibrate_per_view_then_evaluate_with_cache(world):
    root, split_dir, mid, sha = world
    val = load_tier(split_dir, "val")
    new_id, summary = recalibrate(
        root / "models" / mid,
        val,
        registry_root=root / "models",
        per_view=True,
        cache_dir=root / "cache" / "val",
        device="cpu",
    )
    assert new_id != mid
    b = ModelBundle.load(root / "models" / new_id, device="cpu")
    cal = json.loads((root / "models" / new_id / "calibration.json").read_text())
    assert set(cal["thresholds_by_view"]) == {"AP", "PA"}
    assert set(b.calibration.thresholds_by_view) == {"AP", "PA"}
    for c in summary.values():
        assert c["by_view"]["PA"]["fitted"] and c["by_view"]["AP"]["fitted"]
        assert c["by_view"]["LL"]["fitted"] is False  # laterals: labels masked → reported only
    card = json.loads((root / "models" / new_id / "model_card.json").read_text())
    assert card["metrics"]["per_view_thresholds"] is True
    assert "per-view" in card["notes"] and card["metrics"]["recalibrated_from"] == mid
    # OOD normalisation flag survives the copy and is what the loaded gate uses
    assert b.ood is not None and b.ood.normalize is True
    assert b.config.model.cam == "hirescam"  # written with the field → kept

    fc = FrameCache.open(
        root / "cache" / "val",
        pre=b.config.preprocess,
        manifest_sha256=manifest_sha256(split_dir / "val.csv"),
    )
    rep = evaluate(
        b,
        val,
        tier="val",
        dataset_name="synth-a",
        split_manifest_sha256=sha,
        out_dir=root / "runs",
        n_boot=0,
        cache=fc,
    )
    tbv = b.calibration.thresholds_by_view
    for row in rep.subgroups:
        if row.axis != "view":
            continue
        for j, code in enumerate(rep.finding_codes):
            if row.group in tbv:
                assert row.metrics[code].threshold == tbv[row.group][j]
    # the overall row mixes PA and AP thresholds → reported as per-view, and the notes list them
    assert all(np.isnan(m.threshold) for m in rep.overall.values())
    md = (root / "runs" / "report.md").read_text()
    assert "per-view" in md and "Per-view thresholds (D-029)" in md and "ECE eq-mass" in md
    assert rep.extra["thresholds_by_view"] == tbv and rep.extra["ood_normalize"] is True
    saved = np.load(root / "runs" / "predictions.npz")
    assert set(saved["view"]) <= {"PA", "AP", "LL", ""}
    assert rep.n_gated_out >= 1  # laterals/others exist in the synthetic set

    # without per-view the same bundle family keeps a single global threshold everywhere
    old = ModelBundle.load(root / "models" / mid, device="cpu")
    rep0 = evaluate(
        old,
        val,
        tier="val",
        dataset_name="synth-a",
        split_manifest_sha256=sha,
        out_dir=root / "runs0",
        n_boot=0,
        cache=fc,
    )
    assert all(m.threshold == 0.5 for m in rep0.overall.values())
