"""Plugin end-to-end on a tiny randomly-initialised bundle (fast) — checks the contract,
not accuracy. Accuracy is exercised by the slow smoke test."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from secondreader.core.dicom import Deidentifier
from secondreader.core.dicom.imaging import to_imaging_input
from secondreader.core.plugins import ModalityPlugin, PluginRegistry
from secondreader.modalities.cxr.config import CxrConfig
from secondreader.modalities.cxr.gate import OodModel
from secondreader.modalities.cxr.model import Calibration, CxrNet, ModelBundle, save_bundle
from secondreader.modalities.cxr.plugin import CxrPlugin
from secondreader.modalities.cxr.saliency import box_from_heatmap, grad_cam
from tests.fixtures.synthetic import cxr_spec, make_dataset, ultrasound_cine_spec


@pytest.fixture(scope="module")
def bundle_dir(tmp_path_factory) -> Path:
    cfg = CxrConfig(
        name="unit",
        preprocess={"image_size": 96},
        gate={"min_short_side": 64},
        model={"backbone": "resnet18", "pretrained": False},
        train={"num_workers": 0},
    )
    torch.manual_seed(0)
    net = CxrNet(cfg, 3)
    d = net.backbone.num_features
    rng = np.random.default_rng(0)
    feats = rng.normal(size=(300, d))
    ood = OodModel.fit(feats, rng.normal(size=(50, d)), 99.0)
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
    return root / mid


def test_bundle_roundtrip_and_ids(bundle_dir):
    b = ModelBundle.load(bundle_dir, device="cpu")
    card = json.loads((bundle_dir / "model_card.json").read_text())
    assert b.model_version_id == bundle_dir.name == card["model_version_id"]
    assert len(card["model_version_id"]) == 16 and card["init"] == "random"
    assert b.ood is not None and b.calibration.finding_codes == [
        "pneumothorax",
        "pleural_effusion",
        "nodule_mass",
    ]
    assert "research use only" in card["research_use"].lower()


def test_plugin_satisfies_protocol_and_routes(bundle_dir):
    p = CxrPlugin(ModelBundle.load(bundle_dir, device="cpu"))
    assert isinstance(p, ModalityPlugin)
    reg = PluginRegistry()
    reg.register(p)
    assert reg.route({"Modality": "CR", "BodyPartExamined": "CHEST"}) is p
    assert reg.route({"Modality": "DX", "BodyPartExamined": ""}) is p
    assert reg.route({"Modality": "US", "BodyPartExamined": "ABDOMEN"}) is None
    assert reg.route({"Modality": "CR", "BodyPartExamined": "KNEE"}) is None
    assert p.info.intended_use_doc.exists()


def test_gate_and_predict_contract(bundle_dir, deidentifier: Deidentifier):
    p = CxrPlugin(ModelBundle.load(bundle_dir, device="cpu"))
    ds = deidentifier.deidentify(make_dataset(cxr_spec(rows=128, cols=128))).dataset
    inp = to_imaging_input(ds, input_sha256="x" * 64)
    gate = p.quality_gate(inp)
    # random net → view head / OOD may fire; we only require structure + honest reporting
    assert gate.status in ("ok", "unable_to_assess")
    assert "ood_distance" in gate.scores and "p_frontal" in gate.scores
    findings = p.predict(inp)
    assert [f.code for f in findings] == ["pneumothorax", "pleural_effusion", "nodule_mass"]
    for f in findings:
        assert 0.0 <= f.probability <= 1.0
        assert f.localization is not None and f.localization.heatmap is not None
        assert f.localization.heatmap.shape == (128, 128)  # mapped back to source pixels
        assert f.display_name and f.snomed_code
    stats = p.input_stats(inp)
    assert stats["src_h"] == 128


def test_gate_refuses_lateral_and_cine(bundle_dir, deidentifier):
    p = CxrPlugin(ModelBundle.load(bundle_dir, device="cpu"))
    lat = to_imaging_input(
        deidentifier.deidentify(
            make_dataset(cxr_spec(view_position="LL", rows=128, cols=128))
        ).dataset
    )
    g = p.quality_gate(lat)
    assert not g.ok and any("frontal" in r for r in g.reasons)
    us = to_imaging_input(deidentifier.deidentify(make_dataset(ultrasound_cine_spec())).dataset)
    g = p.quality_gate(us)
    assert not g.ok and any("single frame" in r for r in g.reasons)


def test_gradcam_and_box(bundle_dir):
    b = ModelBundle.load(bundle_dir, device="cpu")
    x = torch.randn(1, 3, 96, 96)
    cam = grad_cam(b.net, x, 0)
    assert cam.shape == (96, 96) and cam.min() >= 0 and cam.max() <= 1.0
    heat = np.zeros((50, 80), dtype=np.float32)
    heat[10:20, 40:60] = 1.0
    box = box_from_heatmap(heat)
    assert box is not None
    assert box.x_min == pytest.approx(0.5) and box.x_max == pytest.approx(0.75)
    assert box.y_min == pytest.approx(0.2) and box.y_max == pytest.approx(0.4)
    assert box_from_heatmap(np.zeros((4, 4), dtype=np.float32)) is None


def test_finding_requires_localization_when_present():
    from secondreader.core.plugins.protocol import Finding

    with pytest.raises(ValueError):
        Finding(code="x", display_name="x", probability=0.9, threshold=0.5, localization=None)
    Finding(
        code="x", display_name="x", probability=0.1, threshold=0.5, localization=None
    )  # absent is fine
