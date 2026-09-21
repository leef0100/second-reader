"""Throughput micro-benchmarks: DataLoader images/s and model forward/backward images/s.

Numbers for the development Mac live in docs/reports/throughput_m1pro.md together with the
exact `secondreader cxr bench ...` commands that produced them.
"""

from __future__ import annotations

import time
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from secondreader.modalities.cxr.config import CxrConfig, ModelConfig
from secondreader.modalities.cxr.data import ManifestDataset
from secondreader.modalities.cxr.model import CxrNet, pick_device
from secondreader.modalities.cxr.train import autocast_context, masked_bce, to_device, view_ce


def bench_loader(
    ds: ManifestDataset,
    *,
    batch_size: int,
    num_workers: int,
    steps: int = 30,
    warmup: int = 3,
    prefetch_factor: int = 4,
) -> dict[str, Any]:
    """Images/s of a shuffled DataLoader over `ds` (nothing is sent to the accelerator)."""
    kw: dict[str, Any] = {"prefetch_factor": prefetch_factor} if num_workers > 0 else {}
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, **kw)
    it = iter(dl)
    n, t0 = 0, 0.0
    for i in range(warmup + steps):
        try:
            b = next(it)
        except StopIteration:
            it = iter(dl)
            b = next(it)
        if i == warmup:
            n, t0 = 0, time.perf_counter()
        n += int(b["x"].shape[0])
    dt = time.perf_counter() - t0
    return {
        "kind": "loader",
        "n_items": len(ds),
        "n_cached": ds.n_cached,
        "size": ds.pre.image_size,
        "augment": ds.augment,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "steps": steps,
        "images": n,
        "seconds": round(dt, 3),
        "images_per_s": round(n / dt, 1),
    }


def bench_model(
    *,
    size: int = 320,
    batch_size: int = 16,
    device: str = "auto",
    channels_last: bool = False,
    autocast: bool = False,
    steps: int = 20,
    warmup: int = 5,
    backbone: str = "densenet121",
    n_findings: int = 3,
) -> dict[str, Any]:
    """Forward+backward+AdamW step images/s (train) and forward-only images/s (infer)."""
    dev = pick_device(device)
    cfg = CxrConfig(model=ModelConfig(backbone=backbone, pretrained=False))
    net = CxrNet(cfg, n_findings).to(dev)
    if channels_last:
        net = net.to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
    x = torch.randn(batch_size, 3, size, size)
    y = (torch.rand(batch_size, n_findings) > 0.8).float()
    v = torch.zeros(batch_size, dtype=torch.long)
    x, y, v = to_device(x, dev, channels_last), y.to(dev), v.to(dev)

    def sync() -> None:
        if dev.type == "mps":
            torch.mps.synchronize()
        elif dev.type == "cuda":
            torch.cuda.synchronize()

    net.train()
    for i in range(warmup + steps):
        if i == warmup:
            sync()
            t0 = time.perf_counter()
        with autocast_context(dev, autocast):
            lo, vw, _ = net(x)
        loss = masked_bce(lo.float(), y) + 0.3 * view_ce(vw.float(), v)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
    sync()
    train_dt = time.perf_counter() - t0

    net.eval()
    with torch.no_grad():
        for i in range(warmup + steps):
            if i == warmup:
                sync()
                t1 = time.perf_counter()
            with autocast_context(dev, autocast):
                net(x)
        sync()
    infer_dt = time.perf_counter() - t1
    return {
        "kind": "model",
        "backbone": backbone,
        "device": str(dev),
        "size": size,
        "batch_size": batch_size,
        "channels_last": channels_last,
        "autocast": autocast,
        "steps": steps,
        "train_images_per_s": round(steps * batch_size / train_dt, 1),
        "infer_images_per_s": round(steps * batch_size / infer_dt, 1),
        "last_loss": round(float(loss.item()), 4),
    }


__all__ = ["bench_loader", "bench_model"]
