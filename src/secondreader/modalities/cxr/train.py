"""Train the baseline, calibrate it, fit the OOD model, and write a versioned bundle."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog
import torch
from numpy.typing import NDArray
from torch import nn
from torch.utils.data import DataLoader

from secondreader.core.eval import (
    binary_metrics,
    fit_temperature,
    sensitivity_target_threshold,
    youden_threshold,
)
from secondreader.modalities.cxr.config import FINDING_CODES, CxrConfig
from secondreader.modalities.cxr.data import ManifestDataset
from secondreader.modalities.cxr.gate import OodModel
from secondreader.modalities.cxr.model import Calibration, CxrNet, pick_device, save_bundle

log = structlog.get_logger(__name__)


def masked_bce(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mask = (y >= 0).float()
    loss = nn.functional.binary_cross_entropy_with_logits(logits, y.clamp(min=0), reduction="none")
    return (loss * mask).sum() / mask.sum().clamp(min=1.0)


def view_ce(logits: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    m = v >= 0
    if not bool(m.any()):
        return logits.sum() * 0.0
    return nn.functional.cross_entropy(logits[m], v[m])


@torch.no_grad()
def run_inference(
    net: CxrNet, loader: DataLoader[dict[str, Any]], device: torch.device
) -> dict[str, NDArray[Any]]:
    net.eval()
    logits, views, feats, ys, vs = [], [], [], [], []
    for batch in loader:
        x = batch["x"].to(device)
        lo, vw, f = net(x)
        logits.append(lo.float().cpu().numpy())
        views.append(torch.softmax(vw.float(), dim=1).cpu().numpy())
        feats.append(f.float().cpu().numpy())
        ys.append(batch["y"].numpy())
        vs.append(batch["view"].numpy())
    return {
        "logits": np.concatenate(logits),
        "view_probs": np.concatenate(views),
        "features": np.concatenate(feats),
        "y": np.concatenate(ys),
        "view": np.concatenate(vs),
    }


def _mean_auroc(logits: NDArray[Any], y: NDArray[Any]) -> float:
    from sklearn.metrics import roc_auc_score

    vals = []
    for j in range(y.shape[1]):
        m = y[:, j] >= 0
        if m.sum() and len(np.unique(y[m, j])) == 2:
            vals.append(roc_auc_score(y[m, j], logits[m, j]))
    return float(np.mean(vals)) if vals else float("nan")


def train(
    cfg: CxrConfig,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    registry_root: Path,
    split_manifest_sha256: str | None,
    datasets_info: list[dict[str, Any]],
    notes: str = "",
) -> tuple[str, dict[str, Any]]:
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    device = pick_device(cfg.train.device)
    if cfg.train.max_train_samples:
        train_df = train_df.sample(
            n=min(cfg.train.max_train_samples, len(train_df)), random_state=cfg.train.seed
        )
    log.info(
        "train_start",
        device=str(device),
        n_train=len(train_df),
        n_val=len(val_df),
        backbone=cfg.model.backbone,
        size=cfg.preprocess.image_size,
    )

    ds_tr = ManifestDataset(train_df, cfg.preprocess, augment=cfg.train.augment)
    ds_va = ManifestDataset(val_df, cfg.preprocess, augment=False)
    nw = cfg.train.num_workers
    dl_tr = DataLoader(
        ds_tr,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=nw,
        drop_last=len(ds_tr) > cfg.train.batch_size,
        persistent_workers=nw > 0,
    )
    dl_va = DataLoader(ds_va, batch_size=cfg.train.batch_size, shuffle=False, num_workers=nw)
    dl_tr_eval = DataLoader(ds_tr, batch_size=cfg.train.batch_size, shuffle=False, num_workers=nw)

    net = CxrNet(cfg, len(cfg.finding_codes)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    steps = max(1, len(dl_tr)) * cfg.train.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.train.lr, total_steps=steps, pct_start=0.1
    )

    best_auc, best_state, bad_epochs, history = -1.0, None, 0, []
    for epoch in range(cfg.train.epochs):
        net.train()
        t0, tot, n = time.time(), 0.0, 0
        for batch in dl_tr:
            x, y, v = batch["x"].to(device), batch["y"].to(device), batch["view"].to(device)
            lo, vw, _ = net(x)
            loss = masked_bce(lo, y) + 0.3 * view_ce(vw, v)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            sched.step()
            tot += float(loss.item()) * len(x)
            n += len(x)
        val = run_inference(net, dl_va, device)
        auc = _mean_auroc(val["logits"], val["y"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": tot / max(n, 1),
                "val_mean_auroc": auc,
                "seconds": round(time.time() - t0, 1),
            }
        )
        log.info("epoch", **history[-1])
        if auc > best_auc:
            best_auc, bad_epochs = auc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.train.early_stop_patience:
                log.info("early_stop", epoch=epoch)
                break
    assert best_state is not None
    net.load_state_dict(best_state)
    net.to(device)

    # ---- calibration on validation (never on test/external) ----
    val = run_inference(net, dl_va, device)
    temps, thrs, val_summary = [], [], {}
    for j, code in enumerate(cfg.finding_codes):
        m = val["y"][:, j] >= 0
        z, y = val["logits"][m, j].astype(np.float64), val["y"][m, j].astype(np.float64)
        t = fit_temperature(z, y)
        p = 1 / (1 + np.exp(-z / t))
        thr = (
            youden_threshold(p, y)
            if cfg.train.operating_point == "youden"
            else sensitivity_target_threshold(p, y, 0.9)
        )
        temps.append(float(t))
        thrs.append(float(thr))
        bm = binary_metrics(p, y, thr, n_boot=0)
        val_summary[code] = {
            "auroc": bm.auroc,
            "sensitivity": bm.sensitivity,
            "specificity": bm.specificity,
            "ece": bm.ece,
            "temperature": t,
            "threshold": thr,
            "n": bm.n,
            "n_pos": bm.n_pos,
        }
    calibration = Calibration(temps, thrs, cfg.train.operating_point, list(cfg.finding_codes))

    # ---- OOD model on frontal training features; threshold from validation ----
    ood: OodModel | None = None
    if cfg.gate.ood_enabled:
        tr = run_inference(net, dl_tr_eval, device)
        f_tr = tr["features"][tr["view"] == 0]
        f_va = val["features"][val["view"] == 0]
        if len(f_tr) >= 32 and len(f_va) >= 5:
            ood = OodModel.fit(
                f_tr.astype(np.float64), f_va.astype(np.float64), cfg.gate.ood_percentile
            )
        else:
            log.warning("ood_skipped", n_train_feats=len(f_tr), n_val_feats=len(f_va))

    metrics = {"val": val_summary, "best_val_mean_auroc": best_auc, "history": history}
    out = Path(registry_root) / "pending"
    mid = save_bundle(
        out,
        net=net.cpu(),
        cfg=cfg,
        calibration=calibration,
        ood=ood,
        split_manifest_sha256=split_manifest_sha256,
        metrics=metrics,
        datasets=datasets_info,
        notes=notes,
    )
    (Path(registry_root) / mid / "train_history.json").write_text(json.dumps(history, indent=2))
    log.info("train_done", model_version_id=mid, best_val_mean_auroc=best_auc)
    return mid, metrics


__all__ = ["FINDING_CODES", "train", "run_inference", "masked_bce"]
