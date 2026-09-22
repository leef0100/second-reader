"""Localisation metrics on synthetic heat-maps/boxes, plus the CLI on two synthetic rows."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from typer.testing import CliRunner

from secondreader.core.plugins.protocol import BoundingBox
from secondreader.modalities.cxr.config import CxrConfig
from secondreader.modalities.cxr.gate import OodModel
from secondreader.modalities.cxr.localization_eval import (
    LocMetrics,
    RowResult,
    box_iou,
    centre_prior_heatmap,
    load_nih_boxes,
    mass_inside,
    pixel_box_to_normalized,
    pointing_game_hit,
    score_centre_prior,
    score_random_baseline,
    summarize,
)
from secondreader.modalities.cxr.model import Calibration, CxrNet, save_bundle


def _heat(h: int, w: int, peak_yx: tuple[int, int]) -> np.ndarray:
    heat = np.zeros((h, w), dtype=np.float32)
    heat[peak_yx] = 1.0
    return heat


def test_pixel_box_to_normalized_clips_to_frame():
    b = pixel_box_to_normalized(100, 200, 300, 900, src_h=1000, src_w=1000)
    assert (b.x_min, b.y_min, b.x_max, b.y_max) == (0.1, 0.2, 0.4, 1.0)


def test_pointing_game_inside_and_outside():
    gt = BoundingBox(0.25, 0.25, 0.5, 0.5)  # pixels x,y in [25,50) of a 100×100 map
    assert pointing_game_hit(_heat(100, 100, (30, 40)), gt)
    assert pointing_game_hit(_heat(100, 100, (25, 25)), gt)  # inclusive lower edge
    assert not pointing_game_hit(_heat(100, 100, (50, 30)), gt)  # exclusive upper edge
    assert not pointing_game_hit(_heat(100, 100, (80, 80)), gt)
    assert not pointing_game_hit(np.zeros((100, 100), dtype=np.float32), gt)


def test_iou_known_values():
    a = BoundingBox(0.0, 0.0, 0.5, 0.5)
    assert box_iou(a, a) == pytest.approx(1.0)
    assert box_iou(a, BoundingBox(0.5, 0.5, 1.0, 1.0)) == 0.0
    # half overlap: a=[0,.5]², b=[.25,.75]×[0,.5] → inter .125, union .25+.25-.125
    assert box_iou(a, BoundingBox(0.25, 0.0, 0.75, 0.5)) == pytest.approx(0.125 / 0.375)
    assert box_iou(None, a) == 0.0


def test_mass_inside_known_values():
    heat = np.zeros((10, 10), dtype=np.float32)
    heat[0:5, 0:5] = 1.0  # 25 units in the top-left quadrant
    heat[9, 9] = 3.0  # 3 units outside
    gt = BoundingBox(0.0, 0.0, 0.5, 0.5)
    assert mass_inside(heat, gt) == pytest.approx(25 / 28)
    assert mass_inside(heat, BoundingBox(0.5, 0.5, 1.0, 1.0)) == pytest.approx(3 / 28)
    assert mass_inside(np.zeros((10, 10), dtype=np.float32), gt) == 0.0


def test_centre_prior_points_at_centre():
    h = centre_prior_heatmap(64, 64)
    assert h.shape == (64, 64) and h.max() == pytest.approx(1.0)
    assert pointing_game_hit(h, BoundingBox(0.4, 0.4, 0.6, 0.6))
    assert not pointing_game_hit(h, BoundingBox(0.0, 0.0, 0.2, 0.2))
    m = score_centre_prior(BoundingBox(0.3, 0.3, 0.7, 0.7), 64, 64)
    assert m.pointing_hit == 1.0 and m.iou > 0.3


def test_random_baseline_pointing_matches_area_fraction():
    gt = BoundingBox(0.2, 0.2, 0.6, 0.6)  # area 0.16
    rng = np.random.default_rng(0)
    m = score_random_baseline(BoundingBox(0.0, 0.0, 0.4, 0.4), gt, 200, 200, rng, n_draws=4000)
    assert m.pointing_hit == pytest.approx(0.16, abs=0.03)
    assert m.mass_inside == pytest.approx(0.16)
    assert 0.0 < m.iou < 1.0


def _row(code: str, tier: str, present: bool, hit: float, iou: float) -> RowResult:
    m = LocMetrics(hit, iou, float(iou >= 0.3), float(iou >= 0.5), iou)
    z = LocMetrics(0.0, 0.0, 0.0, 0.0, 0.1)
    return RowResult(
        image_id=f"{code}_{tier}_{hit}_{iou}.png",
        code=code,
        nih_label=code,
        tier=tier,
        src_h=10,
        src_w=10,
        gt=(0.0, 0.0, 0.5, 0.5),
        gt_area_frac=0.25,
        probability=0.9 if present else 0.01,
        threshold=0.5,
        present=present,
        gate_ok=True,
        gate_reasons=(),
        pred_box=(0.0, 0.0, 0.5, 0.5),
        model=m,
        random=z,
        centre=z,
    )


def test_summarize_groups_and_strata():
    rows = [
        _row("pneumothorax", "test", True, 1.0, 0.6),
        _row("pneumothorax", "test", False, 0.0, 0.1),
        _row("pleural_effusion", "train", True, 1.0, 0.4),
    ]
    s = summarize(rows, n_boot=20, seed=0)
    assert s["overall"]["all"]["model"]["n"] == 3
    assert s["overall"]["pneumothorax"]["model"]["pointing_hit"]["mean"] == pytest.approx(0.5)
    assert s["overall"]["pneumothorax"]["model"]["hit_iou_50"]["mean"] == pytest.approx(0.5)
    assert s["overall"]["all"]["model"]["hit_iou_30"]["mean"] == pytest.approx(2 / 3)
    assert s["by_present"]["pneumothorax"]["present"]["model"]["iou"]["mean"] == 0.6
    assert s["by_tier"]["all"]["train"]["model"]["n"] == 1
    assert "nodule_mass" not in s["overall"]
    assert s["overall"]["all"]["random"]["mass_inside"]["mean"] == pytest.approx(0.1)
    assert s["n_present"] == 2 and s["n_gated_out"] == 0


def test_load_nih_boxes_maps_and_ignores(tmp_path: Path):
    csv = tmp_path / "BBox_List_2017.csv"
    csv.write_text(
        "Image Index,Finding Label,Bbox [x,y,w,h],,,\n"
        "a.png,Atelectasis,1,2,3,4\n"
        "b.png,Pneumothorax,10,20,30,40\n"
        "c.png,Nodule,1,1,5,5\n"
        "c.png,Mass,2,2,6,6\n"
        "d.png,Effusion,0,0,9,9\n"
    )
    df = load_nih_boxes(csv)
    assert list(df["code"]) == ["pneumothorax", "nodule_mass", "nodule_mass", "pleural_effusion"]
    assert float(df.loc[0, "w"]) == 30.0


@pytest.fixture(scope="module")
def tiny_registry(tmp_path_factory) -> tuple[Path, str]:
    cfg = CxrConfig(
        name="unit-loc",
        preprocess={"image_size": 64},
        gate={"min_short_side": 32},
        model={"backbone": "resnet18", "pretrained": False},
        train={"num_workers": 0},
    )
    torch.manual_seed(0)
    net = CxrNet(cfg, 3)
    d = net.backbone.num_features
    rng = np.random.default_rng(0)
    ood = OodModel.fit(rng.normal(size=(300, d)), rng.normal(size=(50, d)), 99.0)
    root = tmp_path_factory.mktemp("registry")
    mid = save_bundle(
        root / "pending",
        net=net,
        cfg=cfg,
        calibration=Calibration(
            [1.0, 1.0, 1.0], [0.5, 0.5, 0.5], "youden", list(cfg.finding_codes)
        ),
        ood=ood,
        split_manifest_sha256="abc",
        metrics={},
        datasets=[],
        notes="unit",
    )
    return root, mid


def test_cli_eval_localization_on_two_synthetic_rows(tmp_path: Path, tiny_registry):
    from secondreader.cli.main import app

    registry, mid = tiny_registry
    data = tmp_path / "nih"
    (data / "images").mkdir(parents=True)
    rng = np.random.default_rng(1)
    for name in ("00000001_000.png", "00000002_000.png"):
        img = (rng.random((96, 96)) * 200 + 20).astype(np.uint8)
        Image.fromarray(img).save(data / "images" / name)
    (data / "BBox_List_2017.csv").write_text(
        "Image Index,Finding Label,Bbox [x,y,w,h],,,\n"
        "00000001_000.png,Pneumothorax,10,10,40,40\n"
        "00000002_000.png,Effusion,20,50,60,30\n"
        "00000002_000.png,Cardiomegaly,0,0,50,50\n"
    )
    (data / "Data_Entry_2017_v2020.csv").write_text(
        "Image Index,Finding Labels,Follow-up #,Patient ID,Patient Age,Patient Sex,"
        "View Position,OriginalImage[Width,Height],OriginalImagePixelSpacing[x,y]\n"
        "00000001_000.png,Pneumothorax,0,1,50,M,PA,96,96,0.1,0.1\n"
        "00000002_000.png,Effusion,0,2,61,F,AP,96,96,0.1,0.1\n"
    )
    splits = tmp_path / "splits"
    splits.mkdir()
    (splits / "test.csv").write_text("image_id\n00000001_000.png\n")
    out = tmp_path / "loc"
    res = CliRunner().invoke(
        app,
        [
            "cxr",
            "eval-localization",
            mid,
            "--boxes",
            str(data / "BBox_List_2017.csv"),
            "--splits",
            str(splits),
            "--registry",
            str(registry),
            "--out",
            str(out),
            "--n-boot",
            "10",
            "--n-examples",
            "1",
            "--device",
            "cpu",
        ],
    )
    assert res.exit_code == 0, res.output
    assert "2 boxes on 2 images" in res.output
    md = (out / "localization_report.md").read_text()
    assert "For research use only" in md and "| pneumothorax | model |" in md
    js = json.loads((out / "localization_report.json").read_text())
    assert js["n_rows"] == 2 and js["model_version_id"] == mid
    tiers = {r["image_id"]: r["tier"] for r in js["rows"]}
    assert tiers == {"00000001_000.png": "test", "00000002_000.png": "unknown"}
    for r in js["rows"]:
        assert 0.0 <= r["model"]["iou"] <= 1.0 and 0.0 <= r["model"]["mass_inside"] <= 1.0
        assert r["model"]["pointing_hit"] in (0.0, 1.0)
    pngs = sorted((out / "examples").glob("*.png"))
    assert 1 <= len(pngs) <= 2 and all(p.name.startswith(("good_", "bad_")) for p in pngs)
    assert js["examples"] and js["examples"][0]["png"].startswith("examples/")
