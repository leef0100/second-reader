"""Throughput micro-benchmarks: DataLoader images/s and model forward/backward images/s.

Numbers for the development Mac live in docs/reports/throughput_m1pro.md together with the
exact `secondreader cxr bench ...` commands that produced them. `channels_last` and fp16
`autocast` are bench-only knobs: on the M1 Pro both were slower than plain fp32 (and
channels_last changed the training loss), so the trainer does not expose them.
"""

from __future__ import annotations

import time
from contextlib import AbstractContextManager, nullcontext
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from secondreader.modalities.cxr.config import CxrConfig, ModelConfig
from secondreader.modalities.cxr.data import ManifestDataset
from secondreader.modalities.cxr.model import CxrNet, pick_device
from secondreader.modalities.cxr.train import masked_bce, view_ce


def _autocast(device: torch.device, enabled: bool) -> AbstractContextManager[Any]:
    if not enabled or device.type == "cpu":
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.float16)


def _to_device(x: torch.Tensor, device: torch.device, channels_last: bool) -> torch.Tensor:
    x = x.to(device, non_blocking=True)
    if channels_last and x.ndim == 4:
        x = x.contiguous(memory_format=torch.channels_last)
    return x


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
    torch.manual_seed(0)  # same init + inputs for every flag combination, so losses compare
    cfg = CxrConfig(model=ModelConfig(backbone=backbone, pretrained=False))
    net = CxrNet(cfg, n_findings).to(dev)
    if channels_last:
        net = net.to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
    x = torch.randn(batch_size, 3, size, size)
    y = (torch.rand(batch_size, n_findings) > 0.8).float()
    v = torch.zeros(batch_size, dtype=torch.long)
    x, y, v = _to_device(x, dev, channels_last), y.to(dev), v.to(dev)

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
        with _autocast(dev, autocast):
            lo, vw, _ = net(x)
        loss = masked_bce(lo.float(), y) + 0.3 * view_ce(vw.float(), v)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
    sync()
    train_dt = time.perf_counter() - t0
    mem_mb: int | None = None  # accelerator memory held after training (unified on MPS)
    if dev.type == "mps":
        mem_mb = round(torch.mps.driver_allocated_memory() / 2**20)
    elif dev.type == "cuda":
        mem_mb = round(torch.cuda.max_memory_allocated() / 2**20)

    net.eval()
    with torch.no_grad():
        for i in range(warmup + steps):
            if i == warmup:
                sync()
                t1 = time.perf_counter()
            with _autocast(dev, autocast):
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
        "device_mem_mb_after_train": mem_mb,
    }


__all__ = ["bench_loader", "bench_model"]
