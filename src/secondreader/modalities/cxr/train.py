"""Train the baseline, calibrate it, fit the OOD model, and write a versioned bundle.

Long-run robustness (D-025): a checkpoint (`<registry>/pending/checkpoint.pt`) with model,
optimizer, scheduler and early-stopping state is written after every epoch and on
KeyboardInterrupt (mid-epoch, with the step so the epoch continues where it stopped);
`resume=True` picks it up. Epoch metrics are appended to `<registry>/pending/metrics.jsonl`
and copied into the bundle.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog
import torch
from numpy.typing import NDArray
from torch import nn
from torch.utils.data import DataLoader, Sampler

from secondreader.core.eval import (
    binary_metrics,
    fit_platt,
    fit_temperature,
    sensitivity_target_threshold,
    youden_threshold,
)
from secondreader.modalities.cxr.cache import FrameCache
from secondreader.modalities.cxr.config import FINDING_CODES, CxrConfig
from secondreader.modalities.cxr.data import ManifestDataset
from secondreader.modalities.cxr.gate import OodModel
from secondreader.modalities.cxr.model import (
    Calibration,
    CxrNet,
    normalize_view,
    pick_device,
    save_bundle,
)

log = structlog.get_logger(__name__)

CHECKPOINT_NAME = "checkpoint.pt"
METRICS_NAME = "metrics.jsonl"


def masked_bce(
    logits: torch.Tensor, y: torch.Tensor, pos_weight: torch.Tensor | None = None
) -> torch.Tensor:
    """BCE over labelled entries only (D-017); optional per-finding positive weighting."""
    mask = (y >= 0).float()
    loss = nn.functional.binary_cross_entropy_with_logits(
        logits, y.clamp(min=0), reduction="none", pos_weight=pos_weight
    )
    return (loss * mask).sum() / mask.sum().clamp(min=1.0)


def compute_pos_weight(df: pd.DataFrame, codes: list[str] | tuple[str, ...]) -> list[float]:
    """n_neg / n_pos per finding over labelled (0/1) rows; 1.0 when a finding has no positives."""
    out: list[float] = []
    for c in codes:
        v = pd.to_numeric(df[c], errors="coerce") if c in df.columns else pd.Series(dtype=float)
        n_pos, n_neg = int((v == 1).sum()), int((v == 0).sum())
        out.append(float(n_neg) / n_pos if n_pos else 1.0)
    return out


def resolve_pos_weight(cfg: CxrConfig, train_df: pd.DataFrame) -> list[float] | None:
    pw = cfg.train.pos_weight
    if pw is None:
        return None
    if pw == "auto":
        return compute_pos_weight(train_df, cfg.finding_codes)
    if len(pw) != len(cfg.finding_codes):
        raise ValueError(
            f"train.pos_weight has {len(pw)} entries, expected {len(cfg.finding_codes)}"
        )
    return [float(w) for w in pw]


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
        lo, vw, f = net(batch["x"].to(device, non_blocking=True))
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


def _pick_threshold(p: NDArray[np.float64], y: NDArray[np.float64], operating_point: str) -> float:
    if operating_point == "youden":
        return float(youden_threshold(p, y))
    return float(sensitivity_target_threshold(p, y, 0.9))


def fit_calibration(
    logits: NDArray[Any],
    y: NDArray[Any],
    codes: list[str],
    method: str,
    operating_point: str,
    views: NDArray[Any] | None = None,
    *,
    min_per_view: int = 20,
) -> tuple[Calibration, dict[str, Any]]:
    """Per-finding Platt/temperature fit + operating threshold on VALIDATION outputs only.

    With `views` (one view position per row, e.g. the manifest `view` column) the operating
    threshold is additionally chosen per view (D-029). Scale and bias stay global. A view
    needs at least `min_per_view` positives and negatives for a finding; otherwise that
    finding keeps the global threshold in that view's list.
    """
    temps, biases, thrs, summary = [], [], [], {}
    views_u: NDArray[Any] | None = None
    view_keys: list[str] = []
    if views is not None:
        views_u = np.asarray([normalize_view(v) for v in views], dtype=object)
        view_keys = sorted(v for v in set(views_u.tolist()) if v)
    by_view: dict[str, list[float]] = {v: [] for v in view_keys}
    fitted_views: set[str] = set()  # views with ≥ 1 finding fitted; others are not stored
    for j, code in enumerate(codes):
        m = y[:, j] >= 0
        z, t_y = logits[m, j].astype(np.float64), y[m, j].astype(np.float64)
        if method == "platt":
            t, b = fit_platt(z, t_y)
        else:
            t, b = fit_temperature(z, t_y), 0.0
        p = 1 / (1 + np.exp(-(z / t + b)))
        thr = _pick_threshold(p, t_y, operating_point)
        temps.append(float(t))
        biases.append(float(b))
        thrs.append(float(thr))
        bm = binary_metrics(p, t_y, thr, n_boot=0)
        summary[code] = {
            "auroc": bm.auroc,
            "sensitivity": bm.sensitivity,
            "specificity": bm.specificity,
            "ece": bm.ece,
            "ece_equal_mass": bm.ece_equal_mass,
            "brier": bm.brier,
            "temperature": t,
            "bias": b,
            "threshold": thr,
            "n": bm.n,
            "n_pos": bm.n_pos,
        }
        if views_u is None:
            continue
        vm = views_u[m]
        per_view: dict[str, Any] = {}
        for v in view_keys:
            sel = vm == v
            p_v, y_v = p[sel], t_y[sel]
            n_pos_v, n_neg_v = int((y_v == 1).sum()), int((y_v == 0).sum())
            if n_pos_v >= min_per_view and n_neg_v >= min_per_view:
                thr_v = _pick_threshold(p_v, y_v, operating_point)
                fitted_views.add(v)
            else:
                thr_v = float(thr)  # too little data in this view → global threshold
            by_view[v].append(float(thr_v))
            at_view = binary_metrics(p_v, y_v, thr_v, n_boot=0)
            at_global = binary_metrics(p_v, y_v, thr, n_boot=0)
            per_view[v] = {
                "threshold": float(thr_v),
                "n": at_view.n,
                "n_pos": at_view.n_pos,
                "auroc": at_view.auroc,
                "sensitivity": at_view.sensitivity,
                "specificity": at_view.specificity,
                "sensitivity_at_global": at_global.sensitivity,
                "specificity_at_global": at_global.specificity,
                "fitted": n_pos_v >= min_per_view and n_neg_v >= min_per_view,
            }
        summary[code]["by_view"] = per_view
    cal = Calibration(
        temps,
        thrs,
        operating_point,
        codes,
        biases=biases,
        method=method,
        thresholds_by_view={v: t for v, t in by_view.items() if v in fitted_views},
    )
    return cal, summary


def _mean_auroc(logits: NDArray[Any], y: NDArray[Any]) -> float:
    from sklearn.metrics import roc_auc_score

    vals = []
    for j in range(y.shape[1]):
        m = y[:, j] >= 0
        if m.sum() and len(np.unique(y[m, j])) == 2:
            vals.append(roc_auc_score(y[m, j], logits[m, j]))
    return float(np.mean(vals)) if vals else float("nan")


class ResumableSampler(Sampler[int]):
    """Seeded per-epoch permutation that can skip the items a checkpoint already consumed."""

    def __init__(self, n: int, seed: int) -> None:
        self.n, self.seed, self.epoch, self.skip = n, seed, 0, 0

    def set_epoch(self, epoch: int, skip_items: int = 0) -> None:
        self.epoch, self.skip = epoch, min(skip_items, self.n)

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator().manual_seed(self.seed * 100_003 + self.epoch)
        perm: list[int] = torch.randperm(self.n, generator=g).tolist()
        return iter(perm[self.skip :])

    def __len__(self) -> int:
        return self.n - self.skip


def _loader(
    ds: ManifestDataset,
    cfg: CxrConfig,
    *,
    sampler: Sampler[int] | None = None,
    drop_last: bool = False,
) -> DataLoader[dict[str, Any]]:
    nw = cfg.train.num_workers
    kw: dict[str, Any] = {}
    if nw > 0:
        kw = {"persistent_workers": True, "prefetch_factor": cfg.train.prefetch_factor}
    return DataLoader(
        ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=nw,
        drop_last=drop_last,
        **kw,
    )


def _save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=float) + "\n")


def train(
    cfg: CxrConfig,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    registry_root: Path,
    split_manifest_sha256: str | None,
    datasets_info: list[dict[str, Any]],
    notes: str = "",
    train_cache: FrameCache | None = None,
    val_cache: FrameCache | None = None,
    resume: bool = False,
) -> tuple[str, dict[str, Any]]:
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    device = pick_device(cfg.train.device)
    if cfg.train.max_train_samples:
        train_df = train_df.sample(
            n=min(cfg.train.max_train_samples, len(train_df)), random_state=cfg.train.seed
        )
    pos_weight = resolve_pos_weight(cfg, train_df)
    pw_t = torch.tensor(pos_weight, dtype=torch.float32, device=device) if pos_weight else None
    pending = Path(registry_root) / "pending"
    ckpt_path, metrics_path = pending / CHECKPOINT_NAME, pending / METRICS_NAME
    log.info(
        "train_start",
        device=str(device),
        n_train=len(train_df),
        n_val=len(val_df),
        backbone=cfg.model.backbone,
        size=cfg.preprocess.image_size,
        pos_weight=pos_weight,
        cache=train_cache is not None,
    )

    ds_tr = ManifestDataset(train_df, cfg.preprocess, augment=cfg.train.augment, cache=train_cache)
    ds_va = ManifestDataset(val_df, cfg.preprocess, augment=False, cache=val_cache)
    bs = cfg.train.batch_size
    sampler = ResumableSampler(len(ds_tr), cfg.train.seed)
    drop_last = len(ds_tr) > bs
    dl_tr = _loader(ds_tr, cfg, sampler=sampler, drop_last=drop_last)
    dl_va = _loader(ds_va, cfg)
    dl_tr_eval = _loader(ManifestDataset(train_df, cfg.preprocess, cache=train_cache), cfg)

    full_steps = len(ds_tr) // bs if drop_last else max(1, -(-len(ds_tr) // bs))
    steps_per_epoch = min(full_steps, cfg.train.max_steps_per_epoch or full_steps)
    net = CxrNet(cfg, len(cfg.finding_codes)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=cfg.train.lr,
        total_steps=max(1, steps_per_epoch) * cfg.train.epochs,
        pct_start=0.1,
    )

    best_auc, best_state, bad_epochs = -1.0, None, 0
    history: list[dict[str, Any]] = []
    start_epoch, skip_steps = 0, 0
    cfg_hash = cfg.canonical_hash()
    if resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if ck["config_hash"] != cfg_hash:
            raise ValueError(f"{ckpt_path} was written with a different config; refusing to resume")
        net.load_state_dict(ck["net"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        best_auc, bad_epochs = float(ck["best_auc"]), int(ck["bad_epochs"])
        best_state = ck["best_state"]
        history = list(ck["history"])
        start_epoch, skip_steps = int(ck["epoch"]), int(ck["step_in_epoch"])
        metrics_path.unlink(missing_ok=True)
        for rec in history:
            _append_jsonl(metrics_path, rec)
        log.info("resume", checkpoint=str(ckpt_path), epoch=start_epoch, step=skip_steps)
    else:
        if ckpt_path.exists():
            log.warning("checkpoint_ignored", checkpoint=str(ckpt_path), hint="pass --resume")
        pending.mkdir(parents=True, exist_ok=True)
        metrics_path.unlink(missing_ok=True)

    def checkpoint(epoch: int, step: int) -> None:
        _save_checkpoint(
            ckpt_path,
            {
                "config_hash": cfg_hash,
                "epoch": epoch,
                "step_in_epoch": step,
                "net": {k: v.detach().cpu() for k, v in net.state_dict().items()},
                "opt": opt.state_dict(),
                "sched": sched.state_dict(),
                "best_auc": best_auc,
                "bad_epochs": bad_epochs,
                "best_state": best_state,
                "history": history,
            },
        )

    for epoch in range(start_epoch, cfg.train.epochs):
        step = skip_steps
        sampler.set_epoch(epoch, skip_steps * bs)
        skip_steps = 0
        t0, tot, n = time.time(), 0.0, 0
        try:
            net.train()
            for batch in dl_tr:
                if step >= steps_per_epoch:
                    break
                x = batch["x"].to(device, non_blocking=True)
                y, v = batch["y"].to(device), batch["view"].to(device)
                lo, vw, _ = net(x)
                loss = masked_bce(lo, y, pw_t) + 0.3 * view_ce(vw, v)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                opt.step()
                sched.step()
                tot += float(loss.item()) * len(x)
                n += len(x)
                step += 1
            train_seconds = time.time() - t0
            val = run_inference(net, dl_va, device)
        except KeyboardInterrupt:
            checkpoint(epoch, step)
            log.warning("interrupted", checkpoint=str(ckpt_path), epoch=epoch, step=step)
            raise
        auc = _mean_auroc(val["logits"], val["y"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": tot / max(n, 1),
                "val_mean_auroc": auc,
                "seconds": round(time.time() - t0, 1),
                "train_seconds": round(train_seconds, 1),
                "images_per_s": round(n / max(train_seconds, 1e-9), 1),
                "n_steps": step,
                "n_images": n,
                "lr": float(opt.param_groups[0]["lr"]),
                "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
        )
        _append_jsonl(metrics_path, history[-1])
        log.info("epoch", **history[-1])
        if auc > best_auc:
            best_auc, bad_epochs = auc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            bad_epochs += 1
        checkpoint(epoch + 1, 0)
        if bad_epochs >= cfg.train.early_stop_patience:
            log.info("early_stop", epoch=epoch)
            break
    assert best_state is not None
    net.load_state_dict(best_state)
    net.to(device)

    # ---- calibration on validation (never on test/external) ----
    val = run_inference(net, dl_va, device)
    calibration, val_summary = fit_calibration(
        val["logits"],
        val["y"],
        list(cfg.finding_codes),
        cfg.train.calibration,
        cfg.train.operating_point,
        views=ds_va.df["view"].to_numpy() if cfg.train.per_view_thresholds else None,
    )

    # ---- OOD model on frontal training features; threshold from validation ----
    ood: OodModel | None = None
    if cfg.gate.ood_enabled:
        tr = run_inference(net, dl_tr_eval, device)
        f_tr = tr["features"][tr["view"] == 0]
        f_va = val["features"][val["view"] == 0]
        if len(f_tr) >= 32 and len(f_va) >= 5:
            ood = OodModel.fit(
                f_tr.astype(np.float64),
                f_va.astype(np.float64),
                cfg.gate.ood_percentile,
                normalize=cfg.gate.ood_normalize,
            )
        else:
            log.warning("ood_skipped", n_train_feats=len(f_tr), n_val_feats=len(f_va))

    # ---- drift reference: validation input statistics + calibrated output distributions ----
    from secondreader.core.monitoring import build_reference

    frontal_idx = np.flatnonzero(val["view"] == 0)
    stats = [ds_va.input_stats(int(i)) for i in frontal_idx]
    probs_val = calibration.apply(val["logits"][val["view"] == 0])
    gate_out_rate = float((val["view"] != 0).mean()) if len(val["view"]) else 0.0
    reference = build_reference(stats, probs_val, list(cfg.finding_codes), gate_out_rate)

    metrics = {
        "val": val_summary,
        "best_val_mean_auroc": best_auc,
        "history": history,
        "pos_weight": pos_weight,
    }
    net = net.cpu()
    mid = save_bundle(
        pending,
        net=net,
        cfg=cfg,
        calibration=calibration,
        ood=ood,
        split_manifest_sha256=split_manifest_sha256,
        metrics=metrics,
        datasets=datasets_info,
        notes=notes,
        reference=reference,
    )
    final = Path(registry_root) / mid
    (final / "train_history.json").write_text(json.dumps(history, indent=2))
    if metrics_path.exists():
        metrics_path.replace(final / METRICS_NAME)
    ckpt_path.unlink(missing_ok=True)
    log.info("train_done", model_version_id=mid, best_val_mean_auroc=best_auc)
    return mid, metrics


__all__ = [
    "CHECKPOINT_NAME",
    "FINDING_CODES",
    "METRICS_NAME",
    "ResumableSampler",
    "compute_pos_weight",
    "fit_calibration",
    "masked_bce",
    "recalibrate",
    "resolve_pos_weight",
    "run_inference",
    "train",
]


def recalibrate(
    bundle_dir: Path,
    val_df: pd.DataFrame,
    *,
    registry_root: Path,
    method: str = "platt",
    operating_point: str | None = None,
    cache_dir: Path | None = None,
    batch_size: int = 32,
    device: str = "auto",
    notes: str = "",
    per_view: bool = False,
) -> tuple[str, dict[str, Any]]:
    """Refit calibration (and thresholds) for an existing bundle on validation data.

    Weights, config, split and OOD model are copied unchanged; the result is a NEW
    model_version_id because calibration is part of the version (D-020). `per_view`
    additionally chooses a threshold per view position (D-029).
    """
    import json as _json

    from secondreader.modalities.cxr.cache import FrameCache
    from secondreader.modalities.cxr.model import ModelBundle

    b = ModelBundle.load(bundle_dir, device=device)
    # ModelBundle.load sets model.pretrained=False to avoid downloads; the bundle we write must
    # keep the ORIGINAL training config (init provenance is part of the model card, D-007).
    cfg = CxrConfig.load(Path(bundle_dir) / "config.yaml")
    fc = None
    if cache_dir is not None and (Path(cache_dir) / "cache.json").exists():
        fc = FrameCache.open(Path(cache_dir), pre=b.config.preprocess, manifest_sha256=None)
    ds = ManifestDataset(val_df, b.config.preprocess, augment=False, cache=fc)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    val = run_inference(b.net, dl, b.device)
    op = operating_point or b.calibration.operating_point
    calibration, summary = fit_calibration(
        val["logits"],
        val["y"],
        list(b.calibration.finding_codes),
        method,
        op,
        views=ds.df["view"].to_numpy() if per_view else None,
    )
    card = _json.loads((Path(bundle_dir) / "model_card.json").read_text())
    metrics = dict(card.get("metrics", {}))
    metrics["val"] = summary
    metrics["recalibrated_from"] = b.model_version_id
    metrics["per_view_thresholds"] = bool(calibration.thresholds_by_view)
    if per_view:
        notes = notes or (f"recalibrated ({method}, per-view thresholds) from {b.model_version_id}")
    mid = save_bundle(
        Path(registry_root) / "pending",
        net=b.net.cpu(),
        cfg=cfg,
        calibration=calibration,
        ood=b.ood,
        split_manifest_sha256=card.get("split_manifest_sha256"),
        metrics=metrics,
        datasets=list(card.get("datasets", [])),
        notes=(notes or f"recalibrated ({method}) from {b.model_version_id}"),
        reference=card.get("reference"),
    )
    log.info("recalibrated", from_model=b.model_version_id, model_version_id=mid, method=method)
    return mid, summary
