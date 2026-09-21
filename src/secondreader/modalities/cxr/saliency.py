"""Grad-CAM saliency per finding, mapped back to source-image coordinates, plus a
coarse bounding box derived from the heatmap (for the DICOM SR)."""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray
from scipy import ndimage

from secondreader.core.plugins.protocol import BoundingBox
from secondreader.modalities.cxr.model import CxrNet
from secondreader.modalities.cxr.preprocess import Geometry, unletterbox


def grad_cam(net: CxrNet, x: torch.Tensor, class_index: int) -> NDArray[np.float32]:
    """Heatmap (S,S) in [0,1] for one class. `x` is (1,3,S,S) on the model's device."""
    net.eval()
    x = x.detach().requires_grad_(False)
    with torch.enable_grad():
        fmap = net.feature_maps(x)
        fmap.retain_grad()
        logits = net.head_from_maps(fmap)
        score = logits[0, class_index]
        net.zero_grad(set_to_none=True)
        score.backward()
        grads = fmap.grad
        assert grads is not None
        weights = grads.mean(dim=(2, 3), keepdim=True)  # (1,C,1,1)
        cam = torch.relu((weights * fmap).sum(dim=1, keepdim=True))  # (1,1,h,w)
        cam = torch.nn.functional.interpolate(
            cam, size=x.shape[-2:], mode="bilinear", align_corners=False
        )
    cam_np = cam[0, 0].detach().float().cpu().numpy()
    mx = float(cam_np.max())
    if mx > 0:
        cam_np = cam_np / mx
    return cam_np.astype(np.float32)


def to_source_coords(cam_sq: NDArray[np.float32], geom: Geometry) -> NDArray[np.float32]:
    out = unletterbox(cam_sq, geom)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def box_from_heatmap(heat: NDArray[np.float32], rel_threshold: float = 0.5) -> BoundingBox | None:
    """Largest connected region above `rel_threshold * max`, as a normalized box."""
    if heat.size == 0 or float(heat.max()) <= 0:
        return None
    mask = heat >= rel_threshold * float(heat.max())
    labels, n = ndimage.label(mask)
    if n == 0:
        return None
    sizes = ndimage.sum(mask, labels, index=range(1, n + 1))
    biggest = int(np.argmax(sizes)) + 1
    ys, xs = np.where(labels == biggest)
    h, w = heat.shape
    return BoundingBox(
        x_min=float(xs.min() / w),
        y_min=float(ys.min() / h),
        x_max=float((xs.max() + 1) / w),
        y_max=float((ys.max() + 1) / h),
    )
