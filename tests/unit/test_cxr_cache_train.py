"""Preprocessing cache, resumable training, epoch metrics, and positive-class weighting."""

from __future__ import annotations

import json
import math
import pickle

import numpy as np
import pandas as pd
import pytest
import torch

from secondreader.modalities.cxr import train as train_mod
from secondreader.modalities.cxr.cache import FrameCache, StaleCacheError, build_cache, cache_key
from secondreader.modalities.cxr.config import CxrConfig, PreprocessConfig
from secondreader.modalities.cxr.data import ManifestDataset, load_manifest, manifest_sha256
from secondreader.modalities.cxr.splits import build_splits, load_tier
from secondreader.modalities.cxr.synthetic import generate
from secondreader.modalities.cxr.train import (
    ResumableSampler,
    compute_pos_weight,
    masked_bce,
    resolve_pos_weight,
    train,
)

PRE = PreprocessConfig(image_size=32)
# The cache stores the letterboxed [0,1] frame as uint8: the only permitted difference from the
# decode path is 8-bit quantization (≤ 0.5/255 in [0,1], divided by the smallest ImageNet std).
QUANT_TOL = 0.5 / 255 / 0.225 * 1.01  # +1% for float32 rounding of the normalization
CALIBRATION_KEYS = {"temperatures", "thresholds", "operating_point", "finding_codes"}


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    root = tmp_path_factory.mktemp("synth")
    generate(root / "a", n_patients=40, size=64, seed=0, dataset="synth-a")
    csv = root / "a" / "manifest.csv"
    return root, load_manifest(csv), manifest_sha256(csv)


@pytest.fixture(scope="module")
def splits(synth):
    root, df, _ = synth
    s = build_splits(df, out_dir=root / "splits", version="t", salt="s", external_datasets=set())
    d = root / "splits" / "t"
    return d, load_tier(d, "train"), load_tier(d, "val"), s.manifest_sha256


# ---- cache -----------------------------------------------------------------------------------


def test_cache_matches_decode_path(synth, tmp_path):
    _, df, sha = synth
    meta = build_cache(df, manifest_sha256=sha, pre=PRE, out_dir=tmp_path / "t", workers=1)
    assert meta["n"] == len(df) and meta["n_failed"] == 0
    assert meta["key"] == cache_key(sha, PRE)
    mm = np.load(tmp_path / "t" / "images.npy", mmap_mode="r")
    assert mm.dtype == np.uint8 and mm.shape == (len(df), 32, 32)
    idx = pd.read_csv(tmp_path / "t" / "index.csv", dtype={"image_id": str})
    assert list(idx["image_id"]) == list(df["image_id"].astype(str))
    assert {"mean", "std", "entropy", "src_h", "src_w", "aspect"} <= set(idx.columns)

    fc = FrameCache.open(tmp_path / "t", pre=PRE, manifest_sha256=sha)
    assert len(fc) == len(df)
    ds_c = ManifestDataset(df, PRE, cache=fc)
    ds_d = ManifestDataset(df, PRE)
    assert ds_c.n_cached == len(df) and ds_d.n_cached == 0
    for i in (0, 7, len(df) - 1):
        a, b = ds_c[i], ds_d[i]
        assert tuple(a["x"].shape) == (3, 32, 32)
        assert torch.allclose(a["x"], b["x"], atol=QUANT_TOL)
        assert torch.equal(a["y"], b["y"]) and int(a["view"]) == int(b["view"])
        sc, sd = ds_c.input_stats(i), ds_d.input_stats(i)
        assert set(sc) == set(sd)
        assert all(abs(sc[k] - sd[k]) < 1e-5 for k in sd)
    # augmentation still runs on the cached tensor
    ds_a = ManifestDataset(df, PRE, cache=fc, augment=True)
    assert tuple(ds_a[0]["x"].shape) == (3, 32, 32)


def test_cache_multiprocess_build_and_worker_pickling(synth, tmp_path):
    _, df, sha = synth
    build_cache(df, manifest_sha256=sha, pre=PRE, out_dir=tmp_path / "one", workers=1)
    build_cache(df, manifest_sha256=sha, pre=PRE, out_dir=tmp_path / "two", workers=2, chunk_size=8)
    a = np.load(tmp_path / "one" / "images.npy", mmap_mode="r")
    b = np.load(tmp_path / "two" / "images.npy", mmap_mode="r")
    assert np.array_equal(a, b)
    fc = FrameCache.open(tmp_path / "two", pre=PRE, manifest_sha256=sha)
    x = fc.tensor(3)
    fc2 = pickle.loads(pickle.dumps(fc))  # what DataLoader workers receive
    assert fc2._mm is None and torch.equal(fc2.tensor(3), x)


def test_stale_or_mismatched_cache_is_refused(synth, tmp_path):
    _, df, sha = synth
    build_cache(df, manifest_sha256=sha, pre=PRE, out_dir=tmp_path / "t", workers=1)
    with pytest.raises(StaleCacheError, match="preprocess"):
        FrameCache.open(tmp_path / "t", pre=PreprocessConfig(image_size=32, window_low_pct=1.0))
    with pytest.raises(StaleCacheError, match="manifest"):
        FrameCache.open(tmp_path / "t", pre=PRE, manifest_sha256="0" * 64)
    with pytest.raises(FileNotFoundError):
        FrameCache.open(tmp_path / "missing", pre=PRE)
    fc = FrameCache.open(tmp_path / "t", pre=PRE, manifest_sha256=sha)
    with pytest.raises(ValueError, match="cache size"):
        ManifestDataset(df, PreprocessConfig(image_size=48), cache=fc)


def test_limited_cache_falls_back_to_decoding(synth, tmp_path):
    _, df, sha = synth
    meta = build_cache(df, manifest_sha256=sha, pre=PRE, out_dir=tmp_path / "t", workers=1, limit=5)
    assert meta["n"] == 5 and meta["limit"] == 5 and meta["n_manifest"] == len(df)
    fc = FrameCache.open(tmp_path / "t", pre=PRE, manifest_sha256=sha)
    ds = ManifestDataset(df, PRE, cache=fc)
    assert ds.n_cached == 5 and len(ds) == len(df)
    plain = ManifestDataset(df, PRE)
    assert torch.equal(ds[10]["x"], plain[10]["x"])  # decoded, bit-identical
    assert torch.allclose(ds[2]["x"], plain[2]["x"], atol=QUANT_TOL)  # cached


# ---- class imbalance -------------------------------------------------------------------------


def test_compute_and_resolve_pos_weight():
    df = pd.DataFrame(
        {
            "pneumothorax": [1, 0, 0, 0, -1, np.nan],
            "pleural_effusion": [1, 1, 0, 0, 0, 0],
            "nodule_mass": [0, 0, 0, 0, 0, 0],
        }
    )
    assert compute_pos_weight(df, ("pneumothorax", "pleural_effusion", "nodule_mass")) == [
        3.0,
        2.0,
        1.0,
    ]
    assert resolve_pos_weight(CxrConfig(), df) is None
    assert resolve_pos_weight(CxrConfig(train={"pos_weight": "auto"}), df) == [3.0, 2.0, 1.0]
    assert resolve_pos_weight(CxrConfig(train={"pos_weight": [1, 2, 3]}), df) == [1.0, 2.0, 3.0]
    with pytest.raises(ValueError, match="pos_weight"):
        resolve_pos_weight(CxrConfig(train={"pos_weight": [1.0]}), df)


def test_masked_bce_pos_weight_scales_positives_only():
    logits = torch.zeros(2, 3)
    y = torch.tensor([[1.0, 0.0, -1.0], [0.0, 1.0, 1.0]])
    ln2 = math.log(2.0)
    assert masked_bce(logits, y).item() == pytest.approx(ln2)  # 5 labelled entries, all ln2
    w = torch.tensor([3.0, 1.0, 1.0])
    # only the (0,0) positive gets weight 3; the uncertain (0,2) stays masked (D-017)
    assert masked_bce(logits, y, w).item() == pytest.approx((3 + 4) * ln2 / 5)
    assert masked_bce(logits, y, torch.ones(3)).item() == pytest.approx(ln2)


# ---- resumable training ----------------------------------------------------------------------


def test_resumable_sampler_skips_consumed_items():
    s = ResumableSampler(10, seed=0)
    s.set_epoch(1)
    full = list(s)
    assert sorted(full) == list(range(10))
    s.set_epoch(1, skip_items=4)
    assert list(s) == full[4:] and len(s) == 6
    s.set_epoch(2)
    assert list(s) != full


def _cfg(**train_over):
    return CxrConfig(
        name="t",
        preprocess={"image_size": 32},
        model={"backbone": "resnet18", "pretrained": False},
        train={
            "epochs": 3,
            "batch_size": 8,
            "num_workers": 0,
            "device": "cpu",
            "lr": 1e-3,
            "early_stop_patience": 5,
            "max_steps_per_epoch": 2,
            **train_over,
        },
    )


def test_interrupt_checkpoint_resume_and_metrics_jsonl(splits, tmp_path, monkeypatch):
    _, tr, va, sha = splits
    reg = tmp_path / "models"
    kw = dict(registry_root=reg, split_manifest_sha256=sha, datasets_info=[])
    real = train_mod.masked_bce
    calls = {"n": 0}

    def interrupt_on_4th_step(*a, **k):
        calls["n"] += 1
        if calls["n"] == 4:  # epoch 0 = steps 1-2, epoch 1 = step 3 done, step 4 → Ctrl-C
            raise KeyboardInterrupt
        return real(*a, **k)

    monkeypatch.setattr(train_mod, "masked_bce", interrupt_on_4th_step)
    with pytest.raises(KeyboardInterrupt):
        train(_cfg(), tr, va, **kw)
    ck = torch.load(reg / "pending" / "checkpoint.pt", weights_only=True)
    assert ck["epoch"] == 1 and ck["step_in_epoch"] == 1 and len(ck["history"]) == 1
    assert set(ck) >= {"net", "opt", "sched", "best_state", "best_auc", "bad_epochs"}
    lines = (reg / "pending" / "metrics.jsonl").read_text().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["n_steps"] == 2

    monkeypatch.setattr(train_mod, "masked_bce", real)
    with pytest.raises(ValueError, match="different config"):
        train(_cfg(lr=5e-4), tr, va, resume=True, **kw)
    mid, metrics = train(_cfg(), tr, va, resume=True, **kw)
    hist = metrics["history"]
    assert [h["epoch"] for h in hist] == [0, 1, 2]
    # epoch 1 resumed after its first step: 1 more batch (8 images) completes the 2-step epoch
    assert [h["n_steps"] for h in hist] == [2, 2, 2]
    assert [h["n_images"] for h in hist] == [16, 8, 16]
    assert all({"images_per_s", "train_loss", "val_mean_auroc", "lr"} <= set(h) for h in hist)
    bundle = reg / mid
    assert not (reg / "pending" / "checkpoint.pt").exists()
    assert len((bundle / "metrics.jsonl").read_text().splitlines()) == 3
    assert set(json.loads((bundle / "calibration.json").read_text())) == CALIBRATION_KEYS


def test_train_with_cache_pos_weight_and_max_steps(splits, tmp_path):
    d, tr, va, sha = splits
    caches = {}
    for tier, df in (("train", tr), ("val", va)):
        build_cache(
            df,
            manifest_sha256=manifest_sha256(d / f"{tier}.csv"),
            pre=PRE,
            out_dir=tmp_path / "cache" / tier,
            workers=1,
        )
        caches[tier] = FrameCache.open(
            tmp_path / "cache" / tier, pre=PRE, manifest_sha256=manifest_sha256(d / f"{tier}.csv")
        )
    cfg = _cfg(epochs=1, pos_weight="auto", max_steps_per_epoch=1)
    mid, metrics = train(
        cfg,
        tr,
        va,
        registry_root=tmp_path / "models",
        split_manifest_sha256=sha,
        datasets_info=[],
        train_cache=caches["train"],
        val_cache=caches["val"],
    )
    assert metrics["history"][0]["n_steps"] == 1
    assert len(metrics["pos_weight"]) == 3 and all(w >= 1.0 for w in metrics["pos_weight"])
    card = json.loads((tmp_path / "models" / mid / "model_card.json").read_text())
    assert card["metrics"]["pos_weight"] == metrics["pos_weight"]
    assert card["init"] == "random" and card["reference"]  # drift reference built from the cache
    assert set(json.loads((tmp_path / "models" / mid / "calibration.json").read_text())) == (
        CALIBRATION_KEYS
    )
