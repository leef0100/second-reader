"""Baseline multi-label CXR classifier with a view head, temperature scaling,
per-finding thresholds, and an OOD feature model — packaged as a versioned bundle.

Bundle layout (`<registry>/<model_version_id>/`):
  weights.pt        state_dict
  config.yaml       CxrConfig used for training
  calibration.json  temperatures, biases, thresholds (+ optional per-view thresholds)
  ood.npz           mean / precision / threshold (+ normalize flag) for Mahalanobis OOD
  model_card.json   ids, hashes, split manifest hash, dataset licenses, timestamps
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import timm
import torch
import yaml
from numpy.typing import NDArray
from torch import nn

from secondreader.core.registry import model_version_id, sha256_file, sha256_json
from secondreader.modalities.cxr.config import CxrConfig
from secondreader.modalities.cxr.gate import OodModel

VIEW_CLASSES = ("frontal", "lateral", "other")


class CxrNet(nn.Module):
    def __init__(self, cfg: CxrConfig, n_findings: int) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            cfg.model.backbone,
            pretrained=cfg.model.pretrained,
            num_classes=0,
            drop_rate=cfg.model.drop_rate,
            in_chans=cfg.model.in_chans,
        )
        d = int(self.backbone.num_features)
        self.findings = nn.Linear(d, n_findings)
        self.view = nn.Linear(d, len(VIEW_CLASSES))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)  # (B, D) pooled

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f = self.features(x)
        return self.findings(f), self.view(f), f

    def feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Last conv feature map (B, C, h, w) for Grad-CAM."""
        return self.backbone.forward_features(x)

    def head_from_maps(self, fmap: torch.Tensor) -> torch.Tensor:
        pooled = self.backbone.forward_head(fmap, pre_logits=True)
        return self.findings(pooled)


def pick_device(pref: str = "auto") -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def normalize_view(view: str | None) -> str:
    """DICOM ViewPosition / manifest `view` → upper-case key; '' when unknown."""
    return str(view or "").strip().upper() if view is not None and view == view else ""


@dataclass
class Calibration:
    """p = sigmoid(z / T + b) per finding. `biases` all zero == plain temperature scaling.

    `thresholds` is the global operating point per finding. `thresholds_by_view` (D-029)
    optionally overrides it per view position, e.g. {"PA": [...], "AP": [...]}; any view
    not listed (including unknown) falls back to the global threshold. Scale and bias are
    never per view — only where the line is drawn.
    """

    temperatures: list[float]
    thresholds: list[float]
    operating_point: str
    finding_codes: list[str]
    biases: list[float] = field(default_factory=list)
    method: str = "temperature"
    thresholds_by_view: dict[str, list[float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.biases:
            self.biases = [0.0] * len(self.temperatures)
        k = len(self.thresholds)
        self.thresholds_by_view = {
            normalize_view(v): [float(x) for x in t] for v, t in self.thresholds_by_view.items()
        }
        for v, t in self.thresholds_by_view.items():
            if not v or len(t) != k:
                raise ValueError(f"thresholds_by_view[{v!r}] must have {k} entries, got {len(t)}")

    def apply(self, logits: NDArray[np.float64]) -> NDArray[np.float64]:
        z = np.asarray(logits, dtype=np.float64) / np.asarray(self.temperatures) + np.asarray(
            self.biases
        )
        return np.asarray(1.0 / (1.0 + np.exp(-z)))

    def thresholds_for_view(self, view: str | None) -> list[float]:
        """Per-finding thresholds for one view position (global when unknown / unlisted)."""
        return list(self.thresholds_by_view.get(normalize_view(view), self.thresholds))

    def threshold_for(self, index: int, view: str | None) -> float:
        return float(self.thresholds_for_view(view)[index])

    def row_thresholds(self, views: Iterable[str | None]) -> NDArray[np.float64]:
        """(n, n_findings) thresholds for a sequence of per-row view positions."""
        rows = [self.thresholds_for_view(v) for v in views]
        return np.asarray(rows, dtype=np.float64).reshape(-1, len(self.thresholds))

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "temperatures": self.temperatures,
            "biases": self.biases,
            "thresholds": self.thresholds,
            "operating_point": self.operating_point,
            "finding_codes": self.finding_codes,
            "method": self.method,
        }
        if self.thresholds_by_view:  # absent when unused → older bundles hash identically
            d["thresholds_by_view"] = {v: list(t) for v, t in self.thresholds_by_view.items()}
        return d

    @classmethod
    def from_dict(cls, cal: dict[str, Any]) -> Calibration:
        return cls(
            temperatures=list(cal["temperatures"]),
            thresholds=list(cal["thresholds"]),
            operating_point=str(cal["operating_point"]),
            finding_codes=list(cal["finding_codes"]),
            biases=list(cal.get("biases") or []),
            method=str(cal.get("method", "temperature")),
            thresholds_by_view={
                str(v): list(t) for v, t in (cal.get("thresholds_by_view") or {}).items()
            },
        )


@dataclass
class ModelBundle:
    model_version_id: str
    config: CxrConfig
    net: CxrNet
    calibration: Calibration
    ood: OodModel | None
    card: dict[str, Any]
    device: torch.device

    @classmethod
    def load(cls, path: Path, device: str = "auto") -> ModelBundle:
        path = Path(path)
        cfg = CxrConfig.load(path / "config.yaml")
        cfg.model.pretrained = False  # never download at inference time
        raw = yaml.safe_load((path / "config.yaml").read_text(encoding="utf-8")) or {}
        if "cam" not in (raw.get("model") or {}):
            cfg.model.cam = "gradcam"  # bundle predates the switch: keep its heatmaps (D-029)
        net = CxrNet(cfg, len(cfg.finding_codes))
        state = torch.load(path / "weights.pt", map_location="cpu", weights_only=True)
        net.load_state_dict(state)
        dev = pick_device(device)
        net.to(dev).eval()
        calibration = Calibration.from_dict(json.loads((path / "calibration.json").read_text()))
        ood: OodModel | None = None
        if (path / "ood.npz").exists():
            z = np.load(path / "ood.npz")
            ood = OodModel(
                mean=z["mean"],
                precision=z["precision"],
                threshold=float(z["threshold"]),
                # older bundles were fit on raw features; scoring must match the fit
                normalize=bool(z["normalize"]) if "normalize" in z.files else False,
            )
        card = json.loads((path / "model_card.json").read_text())
        return cls(card["model_version_id"], cfg, net, calibration, ood, card, dev)


def save_bundle(
    out_dir: Path,
    *,
    net: CxrNet,
    cfg: CxrConfig,
    calibration: Calibration,
    ood: OodModel | None,
    split_manifest_sha256: str | None,
    metrics: dict[str, Any],
    datasets: list[dict[str, Any]],
    notes: str = "",
    reference: dict[str, Any] | None = None,
) -> str:
    """Write the bundle; returns the content-hashed model_version_id (D-006)."""
    from datetime import UTC, datetime

    out_dir = Path(out_dir)
    tmp = out_dir.with_name(out_dir.name + ".tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), tmp / "weights.pt")
    (tmp / "config.yaml").write_text(
        yaml.safe_dump(cfg.model_dump(exclude={"content_hash"}), sort_keys=True)
    )
    (tmp / "calibration.json").write_text(json.dumps(calibration.as_dict(), indent=2))
    if ood is not None:
        np.savez(
            tmp / "ood.npz",
            mean=ood.mean,
            precision=ood.precision,
            threshold=ood.threshold,
            normalize=np.asarray(ood.normalize),
        )
    w_sha = sha256_file(tmp / "weights.pt")
    c_sha = cfg.canonical_hash()
    cal_sha = sha256_json(json.loads((tmp / "calibration.json").read_text()))
    mid = model_version_id(w_sha, c_sha, split_manifest_sha256, cal_sha)
    card = {
        "model_version_id": mid,
        "modality_id": "cxr",
        "name": cfg.name,
        "weights_sha256": w_sha,
        "config_sha256": c_sha,
        "calibration_sha256": cal_sha,
        "split_manifest_sha256": split_manifest_sha256,
        "finding_codes": cfg.finding_codes,
        "backbone": cfg.model.backbone,
        "init": "imagenet" if cfg.model.pretrained else "random",
        "datasets": datasets,
        "metrics": metrics,
        "created_at": datetime.now(UTC).isoformat(),
        "notes": notes,
        "reference": reference or {},
        "research_use": "For research use only. Not for diagnostic use.",
    }
    (tmp / "model_card.json").write_text(json.dumps(card, indent=2))
    final = out_dir.parent / mid
    if final.exists():
        import shutil

        shutil.rmtree(final)
    tmp.rename(final)
    return mid
