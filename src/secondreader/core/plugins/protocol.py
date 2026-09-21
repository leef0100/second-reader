"""The modality plugin contract.

Everything modality-specific (chest X-ray today, ultrasound later) lives behind
this protocol. `core` only ever talks to a `ModalityPlugin`.

Design constraints baked in here (see DECISIONS.md D-003):
* `ImagingInput.frames` always has a frame axis — a radiograph is one frame, a
  cine loop is many. Nothing in core assumes `frames == 1`.
* The quality gate runs *before* inference for every modality and may return
  `unable_to_assess`; core never bypasses it.
* Every finding carries a calibrated probability and a localization; the
  results writer refuses findings without both.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from secondreader.core.constants import GATE_OK

GateStatus = Literal["ok", "unable_to_assess"]


@dataclass(frozen=True)
class FrameMeta:
    """Per-frame acquisition metadata a plugin may need (all optional)."""

    photometric_interpretation: str | None = None
    rescale_slope: float | None = None
    rescale_intercept: float | None = None
    window_center: float | None = None
    window_width: float | None = None
    pixel_spacing_mm: tuple[float, float] | None = None
    frame_time_ms: float | None = None  # cine loops


@dataclass(frozen=True)
class ImagingInput:
    """A de-identified imaging object ready for a plugin.

    `frames` shape is (F, H, W) for grayscale or (F, H, W, C) for colour.
    """

    frames: NDArray[np.floating[Any]] | NDArray[np.integer[Any]]
    frame_meta: tuple[FrameMeta, ...]
    modality: str  # DICOM Modality tag, e.g. "CR", "DX", "US"
    sop_class_uid: str
    body_part_examined: str | None
    view_position: str | None
    patient_sex: str | None
    patient_age_years: int | None
    manufacturer: str | None
    manufacturer_model: str | None
    source_tags: dict[str, Any] = field(default_factory=dict)
    input_sha256: str = ""

    @property
    def n_frames(self) -> int:
        return int(self.frames.shape[0])


@dataclass(frozen=True)
class GateResult:
    status: GateStatus
    reasons: tuple[str, ...] = ()
    scores: dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == GATE_OK

    @classmethod
    def passed(cls, **scores: float) -> GateResult:
        return cls(status="ok", scores=dict(scores))

    @classmethod
    def unable(cls, *reasons: str, **scores: float) -> GateResult:
        return cls(status="unable_to_assess", reasons=tuple(reasons), scores=dict(scores))


@dataclass(frozen=True)
class BoundingBox:
    """Normalized [0,1] box in frame coordinates."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float
    frame_index: int = 0


@dataclass(frozen=True)
class Localization:
    """Where the model is looking. At least one of heatmap / boxes must be set."""

    heatmap: NDArray[np.floating[Any]] | None = None  # (H, W) in [0,1], frame-aligned
    boxes: tuple[BoundingBox, ...] = ()
    frame_index: int = 0

    def __post_init__(self) -> None:
        if self.heatmap is None and not self.boxes:
            raise ValueError("Localization requires a heatmap or at least one box")


@dataclass(frozen=True)
class Finding:
    """A single finding with a calibrated probability and localization."""

    code: str  # plugin-scoped id, e.g. "pneumothorax"
    display_name: str
    probability: float  # calibrated P(present)
    threshold: float  # operating point used to call it "present"
    localization: Localization | None  # required when present; None allowed when absent
    snomed_code: str | None = None
    snomed_meaning: str | None = None

    @property
    def present(self) -> bool:
        return self.probability >= self.threshold

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(f"probability out of range: {self.probability}")
        if self.present and self.localization is None:
            raise ValueError(f"finding {self.code} is present but has no localization")


@dataclass(frozen=True)
class PluginInfo:
    modality_id: str
    version: str
    display_name: str
    intended_use_doc: Path
    finding_codes: tuple[str, ...]


@runtime_checkable
class ModalityPlugin(Protocol):
    """Implemented once per modality. See `modalities/cxr` for the reference."""

    @property
    def info(self) -> PluginInfo: ...

    def accepts(self, header: dict[str, Any]) -> bool:
        """Cheap header-only routing: SOP class, Modality, BodyPartExamined."""
        ...

    def quality_gate(self, inp: ImagingInput) -> GateResult:
        """Runs before inference. Returns `unable_to_assess` rather than guessing."""
        ...

    def predict(self, inp: ImagingInput) -> list[Finding]:
        """Calibrated findings with localization. Only called when the gate passed."""
        ...

    @property
    def model_version_id(self) -> str:
        """Content hash of the loaded model (see registry). Recorded on every prediction."""
        ...
