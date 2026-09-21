"""The chest-radiograph `ModalityPlugin`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from secondreader.core.plugins.protocol import (
    Finding,
    GateResult,
    ImagingInput,
    Localization,
    PluginInfo,
)
from secondreader.modalities.cxr.config import FINDING_DISPLAY, FINDING_SNOMED
from secondreader.modalities.cxr.gate import combine, header_checks, learned_checks, pixel_checks
from secondreader.modalities.cxr.model import ModelBundle
from secondreader.modalities.cxr.preprocess import Preprocessed, preprocess_input
from secondreader.modalities.cxr.saliency import box_from_heatmap, grad_cam, to_source_coords

CXR_SOP_CLASSES = {
    "1.2.840.10008.5.1.4.1.1.1",  # CR Image Storage
    "1.2.840.10008.5.1.4.1.1.1.1",  # DX Image Storage – For Presentation
    "1.2.840.10008.5.1.4.1.1.1.1.1",  # DX Image Storage – For Processing
}

PLUGIN_VERSION = "0.1.0"


class _Cache:
    """Per-input forward pass shared between gate and predict."""

    def __init__(self) -> None:
        self.key: str | None = None
        self.pre: Preprocessed | None = None
        self.logits: np.ndarray | None = None
        self.view_probs: np.ndarray | None = None
        self.features: np.ndarray | None = None
        self.x: torch.Tensor | None = None


class CxrPlugin:
    def __init__(self, bundle: ModelBundle) -> None:
        self.bundle = bundle
        self._cache = _Cache()
        self._info = PluginInfo(
            modality_id="cxr",
            version=PLUGIN_VERSION,
            display_name="Chest radiograph (frontal)",
            intended_use_doc=Path(__file__).with_name("INTENDED_USE.md"),
            finding_codes=tuple(bundle.calibration.finding_codes),
        )

    @classmethod
    def from_registry(cls, registry_root: Path, model_id: str, device: str = "auto") -> CxrPlugin:
        return cls(ModelBundle.load(Path(registry_root) / model_id, device=device))

    @property
    def info(self) -> PluginInfo:
        return self._info

    @property
    def model_version_id(self) -> str:
        return self.bundle.model_version_id

    def accepts(self, header: dict[str, Any]) -> bool:
        mod = str(header.get("Modality", "")).upper()
        sop = str(header.get("SOPClassUID", ""))
        bp = str(header.get("BodyPartExamined", "")).upper()
        cfg = self.bundle.config.gate
        return (mod in {m.upper() for m in cfg.allowed_modalities} or sop in CXR_SOP_CLASSES) and (
            bp in {b.upper() for b in cfg.allowed_body_parts}
        )

    # ---- internals ----

    def _forward(self, inp: ImagingInput) -> _Cache:
        key = inp.input_sha256 or str(id(inp))
        c = self._cache
        if c.key == key and c.pre is not None:
            return c
        pre = preprocess_input(inp, self.bundle.config.preprocess)
        x = pre.tensor[None].to(self.bundle.device)
        with torch.no_grad():
            lo, vw, f = self.bundle.net(x)
        c.key, c.pre, c.x = key, pre, x
        c.logits = lo[0].float().cpu().numpy()
        c.view_probs = torch.softmax(vw[0].float(), dim=0).cpu().numpy()
        c.features = f[0].float().cpu().numpy()
        return c

    # ---- protocol ----

    def quality_gate(self, inp: ImagingInput) -> GateResult:
        hdr = header_checks(inp, self.bundle.config.gate)
        if inp.n_frames != 1:
            return GateResult.unable(*hdr)
        c = self._forward(inp)
        assert c.pre is not None and c.features is not None
        px = pixel_checks(
            c.pre.image01[
                c.pre.geometry.pad_top : c.pre.geometry.pad_top + c.pre.geometry.new_h,
                c.pre.geometry.pad_left : c.pre.geometry.pad_left + c.pre.geometry.new_w,
            ],
            c.pre.geometry.src_h,
            c.pre.geometry.src_w,
            self.bundle.config.gate,
        )
        lr = learned_checks(
            self.bundle.config.gate,
            view_probs=c.view_probs,
            features=c.features,
            ood=self.bundle.ood,
        )
        return combine((hdr, {}), px, lr)

    def predict(self, inp: ImagingInput) -> list[Finding]:
        c = self._forward(inp)
        assert c.logits is not None and c.pre is not None and c.x is not None
        probs = self.bundle.calibration.apply(c.logits[None])[0]
        findings: list[Finding] = []
        for j, code in enumerate(self.bundle.calibration.finding_codes):
            cam = grad_cam(self.bundle.net, c.x, j)
            heat = to_source_coords(cam, c.pre.geometry)
            box = box_from_heatmap(heat)
            snomed = FINDING_SNOMED.get(code)
            findings.append(
                Finding(
                    code=code,
                    display_name=FINDING_DISPLAY.get(code, code),
                    probability=float(np.clip(probs[j], 0.0, 1.0)),
                    threshold=float(self.bundle.calibration.thresholds[j]),
                    localization=Localization(heatmap=heat, boxes=(box,) if box else ()),
                    snomed_code=snomed[0] if snomed else None,
                    snomed_meaning=snomed[1] if snomed else None,
                )
            )
        return findings

    def input_stats(self, inp: ImagingInput) -> dict[str, float]:
        c = self._forward(inp)
        assert c.pre is not None
        return dict(c.pre.stats)
