"""Evaluate a bundle on a manifest tier → EvalReport (markdown + plots + JSON).

The gate runs first: gated-out images are counted and excluded from the metrics,
because in deployment they would never produce a finding. Operating thresholds are
looked up per row from the manifest `view` column (D-029), exactly as the plugin does
from `ImagingInput.view_position`.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from secondreader.core.eval import (
    EvalReport,
    SubgroupRow,
    age_band,
    binary_metrics,
    expected_calibration_error,
    render_markdown,
    write_reliability_plots,
)
from secondreader.modalities.cxr.data import ManifestDataset, load_image
from secondreader.modalities.cxr.gate import learned_checks, pixel_checks, pixel_checks_from_stats
from secondreader.modalities.cxr.model import ModelBundle
from secondreader.modalities.cxr.preprocess import to_float_frame, window
from secondreader.modalities.cxr.train import run_inference

if TYPE_CHECKING:
    from secondreader.modalities.cxr.cache import FrameCache


def _gate_rows(
    bundle: ModelBundle, df: pd.DataFrame, out: dict[str, Any], ds: ManifestDataset
) -> tuple[np.ndarray, Counter[str]]:
    """Apply header/pixel/learned gates per row. Returns keep-mask and reason counts.

    Pixel checks use the cache's per-image statistics when the row is cached (same
    numbers `preprocess_frame` computes), otherwise the image is decoded again.
    """
    cfg = bundle.config.gate
    keep = np.ones(len(df), dtype=bool)
    reasons: Counter[str] = Counter()
    for i, (_, row) in enumerate(df.iterrows()):
        rs: list[str] = []
        view = str(row.get("view") or "").upper()
        if view not in {v.upper() for v in cfg.frontal_views}:
            rs.append("header: non-frontal view")
        if ds.cache is not None and ds._cache_rows[i] is not None:
            pr, _ = pixel_checks_from_stats(ds.input_stats(i), cfg)
        else:
            arr, meta = load_image(row["path"])
            a01 = window(
                to_float_frame(arr, meta),
                bundle.config.preprocess.window_low_pct,
                bundle.config.preprocess.window_high_pct,
            )
            pr, _ = pixel_checks(a01, *a01.shape, cfg)
        rs += [f"pixels: {r}" for r in pr]
        lr, _ = learned_checks(
            cfg,
            view_probs=out["view_probs"][i],
            features=out["features"][i],
            ood=bundle.ood,
            logits=out["logits"][i],
        )
        rs += [
            "learned: out-of-distribution"
            if "out-of-distribution" in r
            else "learned: view classifier"
            for r in lr
        ]
        if rs:
            keep[i] = False
            reasons.update(rs)
    return keep, reasons


def evaluate(
    bundle: ModelBundle,
    df: pd.DataFrame,
    *,
    tier: str,
    dataset_name: str,
    split_manifest_sha256: str,
    out_dir: Path,
    n_boot: int = 200,
    batch_size: int = 16,
    apply_gate: bool = True,
    cache: FrameCache | None = None,
) -> EvalReport:
    codes = list(bundle.calibration.finding_codes)
    ds = ManifestDataset(df, bundle.config.preprocess, augment=False, cache=cache)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    out = run_inference(bundle.net, dl, bundle.device)
    probs = bundle.calibration.apply(out["logits"])

    if apply_gate:
        keep, reasons = _gate_rows(bundle, df, out, ds)
    else:
        keep, reasons = np.ones(len(df), dtype=bool), Counter()
    kept = df.reset_index(drop=True)[keep].reset_index(drop=True)
    p_all, y_all = probs[keep], out["y"][keep]
    # one threshold per row: the view's operating point when calibrated per view (D-029)
    thr_all = bundle.calibration.row_thresholds(kept["view"].tolist())

    def metrics_for(mask: np.ndarray) -> dict[str, Any]:
        res: dict[str, Any] = {}
        for j, code in enumerate(codes):
            m = mask & (y_all[:, j] >= 0)
            if m.sum() == 0:
                continue
            res[code] = binary_metrics(
                p_all[m, j], y_all[m, j], thr_all[m, j], n_boot=n_boot, seed=j
            )
        return res

    all_mask = np.ones(len(kept), dtype=bool)
    overall = metrics_for(all_mask)
    axes = {
        "source": kept["dataset"].fillna("unavailable").astype(str),
        "vendor": kept["vendor"]
        .fillna("unavailable")
        .astype(str)
        .replace({"": "unavailable", "nan": "unavailable"}),
        "age_band": kept["age"].map(lambda a: age_band(None if pd.isna(a) else float(a))),
        "sex": kept["sex"]
        .fillna("unavailable")
        .astype(str)
        .replace({"": "unavailable", "nan": "unavailable"}),
        "view": kept["view"]
        .fillna("unavailable")
        .astype(str)
        .replace({"": "unavailable", "nan": "unavailable"}),
    }
    subgroups: list[SubgroupRow] = []
    for axis, series in axes.items():
        groups = sorted(series.unique().tolist()) if len(kept) else ["unavailable"]
        for g in groups:
            m = (series == g).to_numpy() if len(kept) else all_mask
            subgroups.append(SubgroupRow(axis=axis, group=str(g), metrics=metrics_for(m)))

    reliability = {}
    for j, code in enumerate(codes):
        m = y_all[:, j] >= 0
        if m.sum():
            reliability[code] = expected_calibration_error(p_all[m, j], y_all[m, j])[1]
    report = EvalReport(
        title=f"Evaluation — {bundle.config.name} on {dataset_name} ({tier})",
        model_version_id=bundle.model_version_id,
        dataset_name=dataset_name,
        split_manifest_sha256=split_manifest_sha256,
        tier=tier,
        finding_codes=codes,
        overall=overall,
        subgroups=subgroups,
        reliability=reliability,
        n_gated_out=int((~keep).sum()),
        gate_reasons=dict(reasons),
        notes=[
            f"Operating point: {bundle.calibration.operating_point} "
            "(chosen on internal validation).",
            "Global thresholds: "
            + str(
                dict(zip(codes, [round(t, 4) for t in bundle.calibration.thresholds], strict=True))
            ),
            *(
                [
                    "Per-view thresholds (D-029), used for rows whose `view` matches; other views "
                    "use the global threshold: "
                    + str(
                        {
                            v: dict(zip(codes, [round(t, 4) for t in ts], strict=True))
                            for v, ts in bundle.calibration.thresholds_by_view.items()
                        }
                    )
                ]
                if bundle.calibration.thresholds_by_view
                else []
            ),
            "Temperatures: "
            + str(
                dict(
                    zip(codes, [round(t, 3) for t in bundle.calibration.temperatures], strict=True)
                )
            ),
            f"Bootstrap 95% CIs (n={n_boot}). Internal and external metrics are never averaged.",
            f"Saliency method configured for this bundle: {bundle.config.model.cam}.",
        ],
        extra={
            "thresholds": list(bundle.calibration.thresholds),
            "thresholds_by_view": dict(bundle.calibration.thresholds_by_view),
            "cam": bundle.config.model.cam,
            "ood_normalize": bool(bundle.ood.normalize) if bundle.ood is not None else None,
        },
    )
    report.validate()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots = write_reliability_plots(report, out_dir)
    (out_dir / "report.md").write_text(render_markdown(report, plots))
    (out_dir / "report.json").write_text(json.dumps(report.as_dict(), indent=2, default=float))
    np.savez(
        out_dir / "predictions.npz",
        probs=probs,
        y=out["y"],
        keep=keep,
        image_id=df["image_id"].to_numpy().astype(str),
        view=df["view"].fillna("").to_numpy().astype(str),
    )
    return report


__all__ = ["evaluate", "torch"]
