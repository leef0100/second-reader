"""Slow: train on a tiny synthetic set, evaluate, and check the report + plots exist."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from secondreader.modalities.cxr.config import CxrConfig
from secondreader.modalities.cxr.evaluate import evaluate
from secondreader.modalities.cxr.model import ModelBundle
from secondreader.modalities.cxr.splits import build_splits, load_tier
from secondreader.modalities.cxr.synthetic import generate
from secondreader.modalities.cxr.train import train

pytestmark = pytest.mark.slow


def test_train_eval_pipeline(tmp_path):
    a = generate(tmp_path / "a", n_patients=40, size=96, seed=0, dataset="synth-a")
    b = generate(tmp_path / "b", n_patients=12, size=112, seed=1, dataset="synth-b", contrast=0.8)
    s = build_splits(
        pd.concat([a, b]),
        out_dir=tmp_path / "splits",
        version="t",
        salt="s",
        external_datasets={"synth-b"},
    )
    cfg = CxrConfig(
        name="smoke",
        preprocess={"image_size": 96},
        gate={"min_short_side": 64},
        model={"backbone": "resnet18", "pretrained": False},
        train={"epochs": 2, "batch_size": 8, "num_workers": 0, "device": "cpu", "lr": 1e-3},
    )
    mid, metrics = train(
        cfg,
        load_tier(tmp_path / "splits" / "t", "train"),
        load_tier(tmp_path / "splits" / "t", "val"),
        registry_root=tmp_path / "models",
        split_manifest_sha256=s.manifest_sha256,
        datasets_info=[{"name": "synth-a", "license": "none", "commercial_ok": True}],
    )
    assert (tmp_path / "models" / mid / "weights.pt").exists()
    assert set(metrics["val"]) == {"pneumothorax", "pleural_effusion", "nodule_mass"}
    cal = json.loads((tmp_path / "models" / mid / "calibration.json").read_text())
    assert set(cal) == {
        "temperatures",
        "biases",
        "thresholds",
        "operating_point",
        "finding_codes",
        "method",
    }
    assert (tmp_path / "models" / mid / "metrics.jsonl").exists()
    bundle = ModelBundle.load(tmp_path / "models" / mid, device="cpu")
    rep = evaluate(
        bundle,
        load_tier(tmp_path / "splits" / "t", "external"),
        tier="external",
        dataset_name="synth-b",
        split_manifest_sha256=s.manifest_sha256,
        out_dir=tmp_path / "runs",
        n_boot=10,
    )
    md = (tmp_path / "runs" / "report.md").read_text()
    assert "For research use only" in md and "### by vendor" in md and "### by view" in md
    assert (tmp_path / "runs" / "reliability_pneumothorax.png").exists()
    assert rep.n_gated_out >= 1  # laterals/others exist in the synthetic external set
