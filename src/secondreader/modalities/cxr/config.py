"""Chest X-ray plugin configuration (validated YAML → pydantic)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

FINDING_CODES: tuple[str, ...] = ("pneumothorax", "pleural_effusion", "nodule_mass")

FINDING_DISPLAY: dict[str, str] = {
    "pneumothorax": "Pneumothorax",
    "pleural_effusion": "Pleural effusion",
    "nodule_mass": "Pulmonary nodule/mass",
}

# SNOMED CT codes used in the DICOM SR output.
FINDING_SNOMED: dict[str, tuple[str, str]] = {
    "pneumothorax": ("36118008", "Pneumothorax"),
    "pleural_effusion": ("60046008", "Pleural effusion"),
    "nodule_mass": ("427359005", "Solitary nodule of lung"),
}


class PreprocessConfig(BaseModel):
    image_size: int = 320
    window_low_pct: float = 0.5
    window_high_pct: float = 99.5
    letterbox: bool = True


class GateConfig(BaseModel):
    allowed_modalities: list[str] = ["CR", "DX"]
    allowed_body_parts: list[str] = ["CHEST", "THORAX", "LUNG", "PECTUS", "TORAX", ""]
    frontal_views: list[str] = ["PA", "AP", ""]
    min_short_side: int = 256
    max_aspect_ratio: float = 1.8  # long/short
    min_std_fraction: float = 0.03  # of dynamic range — below → blank/uniform
    max_saturated_fraction: float = 0.35
    ood_enabled: bool = True
    ood_percentile: float = 99.0  # validation Mahalanobis percentile → threshold
    # Mahalanobis++ (Mueller & Hein 2025): L2-normalise penultimate features before fitting the
    # Gaussian and before scoring. Recorded in ood.npz, so a bundle always scores the way it
    # was fit regardless of this flag (D-029). Opt-in: unverified on CXR, A/B it.
    ood_normalize: bool = False
    view_classifier_enabled: bool = True
    view_lateral_threshold: float = 0.5


class ModelConfig(BaseModel):
    backbone: str = "densenet121"
    pretrained: bool = True  # ImageNet init only (D-007)
    drop_rate: float = 0.2
    in_chans: int = 3
    # Saliency variant shown to the clinician. HiResCAM (Draelos & Carin 2020) is the
    # element-wise gradient × activation product — provably the class-score decomposition for
    # conv → GAP → linear heads. Bundles written before this field existed are pinned to
    # "gradcam" on load (D-029), so an existing model's heatmaps never change silently.
    cam: Literal["gradcam", "hirescam"] = "hirescam"


class TrainConfig(BaseModel):
    epochs: int = 10
    batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 1e-4
    early_stop_patience: int = 3
    num_workers: int = 2
    augment: bool = True
    seed: int = 0
    operating_point: Literal["youden", "sens90"] = "youden"
    # "platt" fits scale + bias (needed whenever pos_weight shifts the logits, D-027);
    # "temperature" fits scale only.
    calibration: Literal["platt", "temperature"] = "platt"
    max_train_samples: int | None = None  # for smoke runs
    max_steps_per_epoch: int | None = None  # for smoke runs: cap optimizer steps per epoch
    device: str = "auto"
    # Positive-class weighting for the masked BCE (class imbalance, e.g. NIH pneumothorax ~5%).
    # "auto" = n_neg / n_pos per finding from the training manifest; a list = explicit weights;
    # null = unweighted (default).
    pos_weight: Literal["auto"] | list[float] | None = None
    prefetch_factor: int = 4  # DataLoader batches prefetched per worker
    # Choose the operating threshold separately per view (PA / AP) on validation (D-029).
    # Platt scale/bias stay global; only the threshold moves. Unknown views use the global one.
    per_view_thresholds: bool = False


class CxrConfig(BaseModel):
    name: str = "cxr-baseline"
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    gate: GateConfig = Field(default_factory=GateConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    finding_codes: list[str] = list(FINDING_CODES)
    content_hash: str = ""

    @classmethod
    def load(cls, path: Path | str) -> CxrConfig:
        text = Path(path).read_text(encoding="utf-8")
        data: dict[str, Any] = yaml.safe_load(text) or {}
        cfg = cls.model_validate(data)
        cfg.content_hash = hashlib.sha256(text.encode()).hexdigest()
        return cfg

    def canonical_hash(self) -> str:
        from secondreader.core.registry import sha256_json

        return sha256_json(self.model_dump(exclude={"content_hash"}))
