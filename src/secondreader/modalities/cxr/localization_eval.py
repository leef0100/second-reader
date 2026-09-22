"""Does the heat-map / box point at the finding? Localisation evaluation against
radiologist boxes (NIH ChestX-ray14 `BBox_List_2017.csv`).

The pitch must not claim more than we can show (CLAUDE.md principle 3): every finding is
rendered with a heat-map and a box, so this module measures how often that box actually
covers the radiologist's box — per finding, against a random-box and a centre-prior
baseline so the numbers have context, and stratified by whether the model called the
finding present and by which split tier the image came from (most NIH box images fall in
our *training* tier and must not be mistaken for held-out evidence).

Metrics per (image, finding) row, all in source-pixel coordinates:
* pointing game — the heat-map argmax lies inside the ground-truth (GT) box;
* IoU between our derived box (`saliency.box_from_heatmap`) and the GT box, plus
  hit@IoU ≥ 0.3 and ≥ 0.5;
* fraction of heat-map mass inside the GT box.

Baselines (same metrics):
* random — a box of the same size as ours at a uniformly random position, a uniformly
  random pointing location, and a uniform heat-map (mass inside = GT area fraction);
* centre prior — an isotropic Gaussian at the image centre (σ = 0.2 · short side),
  boxed with the same `box_from_heatmap` rule.

NIH-derived boxes are evaluation only, never a training signal (D-008, R9).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from secondreader.core.constants import RESEARCH_USE_LABEL
from secondreader.core.plugins.protocol import BoundingBox, FrameMeta, ImagingInput
from secondreader.core.results.writer import render_overlay_rgb
from secondreader.modalities.cxr.config import FINDING_CODES
from secondreader.modalities.cxr.data import load_image
from secondreader.modalities.cxr.model import ModelBundle
from secondreader.modalities.cxr.plugin import CxrPlugin
from secondreader.modalities.cxr.saliency import box_from_heatmap

FloatArr = NDArray[np.float32]

#: NIH `Finding Label` → our finding code. Labels absent here are ignored.
NIH_LABEL_MAP: dict[str, str] = {
    "Pneumothorax": "pneumothorax",
    "Effusion": "pleural_effusion",
    "Nodule": "nodule_mass",
    "Mass": "nodule_mass",
}

IOU_HIT_LEVELS: tuple[float, ...] = (0.3, 0.5)
CENTRE_PRIOR_SIGMA_FRAC = 0.2
CR_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.1"
GT_BOX_COLOUR = (0, 255, 255)  # cyan; our box is yellow (render_overlay_rgb)


# --------------------------------------------------------------------------- metrics


def pixel_box_to_normalized(
    x: float, y: float, w: float, h: float, src_h: int, src_w: int
) -> BoundingBox:
    """`[x, y, w, h]` in source pixels → normalized [0,1] `BoundingBox`, clipped to the frame."""
    return BoundingBox(
        x_min=float(np.clip(x / src_w, 0.0, 1.0)),
        y_min=float(np.clip(y / src_h, 0.0, 1.0)),
        x_max=float(np.clip((x + w) / src_w, 0.0, 1.0)),
        y_max=float(np.clip((y + h) / src_h, 0.0, 1.0)),
    )


def box_area(b: BoundingBox) -> float:
    return max(0.0, b.x_max - b.x_min) * max(0.0, b.y_max - b.y_min)


def box_iou(a: BoundingBox | None, b: BoundingBox) -> float:
    """Intersection over union of two normalized boxes; 0 when `a` is None."""
    if a is None:
        return 0.0
    ix = max(0.0, min(a.x_max, b.x_max) - max(a.x_min, b.x_min))
    iy = max(0.0, min(a.y_max, b.y_max) - max(a.y_min, b.y_min))
    inter = ix * iy
    union = box_area(a) + box_area(b) - inter
    return float(inter / union) if union > 0 else 0.0


def _pixel_bounds(gt: BoundingBox, h: int, w: int) -> tuple[int, int, int, int]:
    """GT box as half-open pixel bounds (y0, y1, x0, x1), at least one pixel wide."""
    y0 = int(np.floor(gt.y_min * h))
    x0 = int(np.floor(gt.x_min * w))
    y1 = max(y0 + 1, int(np.ceil(gt.y_max * h)))
    x1 = max(x0 + 1, int(np.ceil(gt.x_max * w)))
    return max(0, y0), min(h, y1), max(0, x0), min(w, x1)


def point_in_box(y: int, x: int, gt: BoundingBox, h: int, w: int) -> bool:
    y0, y1, x0, x1 = _pixel_bounds(gt, h, w)
    return y0 <= y < y1 and x0 <= x < x1


def pointing_game_hit(heat: FloatArr, gt: BoundingBox) -> bool:
    """Is the heat-map argmax (first maximum in row-major order) inside the GT box?"""
    if heat.size == 0 or float(heat.max()) <= 0:
        return False
    y, x = np.unravel_index(int(np.argmax(heat)), heat.shape)
    return point_in_box(int(y), int(x), gt, heat.shape[0], heat.shape[1])


def mass_inside(heat: FloatArr, gt: BoundingBox) -> float:
    """Fraction of total heat-map mass that lies inside the GT box (0 when the map is empty)."""
    total = float(heat.sum())
    if heat.size == 0 or total <= 0:
        return 0.0
    y0, y1, x0, x1 = _pixel_bounds(gt, heat.shape[0], heat.shape[1])
    return float(heat[y0:y1, x0:x1].sum() / total)


def centre_prior_heatmap(h: int, w: int, sigma_frac: float = CENTRE_PRIOR_SIGMA_FRAC) -> FloatArr:
    """Isotropic Gaussian at the frame centre, max-normalised to 1 — a 'no-image' prior."""
    sigma = max(1.0, sigma_frac * min(h, w))
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    g = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma**2))
    return (g / g.max()).astype(np.float32)


def random_box_like(pred: BoundingBox | None, rng: np.random.Generator) -> BoundingBox:
    """A box with `pred`'s width/height (or a 0.25×0.25 box if None) at a uniform random spot."""
    bw = (pred.x_max - pred.x_min) if pred is not None else 0.25
    bh = (pred.y_max - pred.y_min) if pred is not None else 0.25
    bw, bh = float(np.clip(bw, 1e-6, 1.0)), float(np.clip(bh, 1e-6, 1.0))
    x0 = float(rng.uniform(0.0, 1.0 - bw))
    y0 = float(rng.uniform(0.0, 1.0 - bh))
    return BoundingBox(x_min=x0, y_min=y0, x_max=x0 + bw, y_max=y0 + bh)


@dataclass(frozen=True)
class LocMetrics:
    """The per-row metric set, for the model or for a baseline."""

    pointing_hit: float  # 0/1 for the model; mean over draws for the random baseline
    iou: float
    hit_iou_30: float
    hit_iou_50: float
    mass_inside: float


def score_heatmap(heat: FloatArr, box: BoundingBox | None, gt: BoundingBox) -> LocMetrics:
    iou = box_iou(box, gt)
    return LocMetrics(
        pointing_hit=float(pointing_game_hit(heat, gt)),
        iou=iou,
        hit_iou_30=float(iou >= 0.3),
        hit_iou_50=float(iou >= 0.5),
        mass_inside=mass_inside(heat, gt),
    )


def score_random_baseline(
    pred_box: BoundingBox | None,
    gt: BoundingBox,
    h: int,
    w: int,
    rng: np.random.Generator,
    n_draws: int = 32,
) -> LocMetrics:
    """Random box of our size, random pointing, uniform heat — averaged over `n_draws`."""
    hits, ious = [], []
    for _ in range(n_draws):
        rb = random_box_like(pred_box, rng)
        ious.append(box_iou(rb, gt))
        py, px = int(rng.integers(0, h)), int(rng.integers(0, w))
        hits.append(float(point_in_box(py, px, gt, h, w)))
    iou_arr = np.asarray(ious)
    return LocMetrics(
        pointing_hit=float(np.mean(hits)),
        iou=float(iou_arr.mean()),
        hit_iou_30=float((iou_arr >= 0.3).mean()),
        hit_iou_50=float((iou_arr >= 0.5).mean()),
        mass_inside=box_area(gt),  # uniform heat: mass inside == area fraction
    )


def score_centre_prior(gt: BoundingBox, h: int, w: int) -> LocMetrics:
    heat = centre_prior_heatmap(h, w)
    return score_heatmap(heat, box_from_heatmap(heat), gt)


# --------------------------------------------------------------------------- rows


@dataclass
class RowResult:
    image_id: str
    code: str
    nih_label: str
    tier: str  # train / val / test / unknown
    src_h: int
    src_w: int
    gt: tuple[float, float, float, float]  # normalized x_min, y_min, x_max, y_max
    gt_area_frac: float
    probability: float
    threshold: float
    present: bool
    gate_ok: bool
    gate_reasons: tuple[str, ...]
    pred_box: tuple[float, float, float, float] | None
    model: LocMetrics
    random: LocMetrics
    centre: LocMetrics

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["gate_reasons"] = list(self.gate_reasons)
        return d


def load_nih_boxes(csv_path: Path | str) -> pd.DataFrame:
    """`BBox_List_2017.csv` → rows for our findings only: image_id, nih_label, code, x, y, w, h."""
    raw = pd.read_csv(csv_path)
    raw = raw.iloc[:, :6].copy()
    raw.columns = ["image_id", "nih_label", "x", "y", "w", "h"]
    raw["code"] = raw["nih_label"].astype(str).str.strip().map(NIH_LABEL_MAP)
    df = raw.dropna(subset=["code"]).copy()
    for c in ("x", "y", "w", "h"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["x", "y", "w", "h"])
    df["image_id"] = df["image_id"].astype(str)
    return df.reset_index(drop=True)


def load_nih_entry(images_root: Path | str) -> pd.DataFrame | None:
    """Per-image view/sex/age from NIH's Data_Entry CSV (either release), if present."""
    root = Path(images_root)
    for name in ("Data_Entry_2017_v2020.csv", "Data_Entry_2017.csv"):
        p = root / name
        if p.exists():
            df = pd.read_csv(p)
            sex_col = "Patient Sex" if "Patient Sex" in df.columns else "Patient Gender"
            age = pd.to_numeric(
                df["Patient Age"].astype(str).str.extract(r"(\d+)")[0], errors="coerce"
            )
            return pd.DataFrame(
                {
                    "image_id": df["Image Index"].astype(str),
                    "view": df["View Position"].astype(str).str.upper(),
                    "sex": df[sex_col].astype(str).str.upper().str[0],
                    "age": age.where(age < 120),
                }
            ).set_index("image_id")
    return None


def load_tiers(splits_dir: Path | str | None) -> dict[str, str]:
    """image_id → tier from `<splits>/{train,val,test,external}.csv` (empty when absent)."""
    out: dict[str, str] = {}
    if splits_dir is None:
        return out
    for tier in ("train", "val", "test", "external"):
        p = Path(splits_dir) / f"{tier}.csv"
        if p.exists():
            ids = pd.read_csv(p, usecols=["image_id"])["image_id"].astype(str)
            out.update(dict.fromkeys(ids, tier))
    return out


def imaging_input_from_png(path: Path, meta_row: pd.Series | None, image_id: str) -> ImagingInput:
    arr, fmeta = load_image(path)
    view = str(meta_row["view"]) if meta_row is not None else ""
    sex = str(meta_row["sex"]) if meta_row is not None else None
    age = meta_row["age"] if meta_row is not None else None
    return ImagingInput(
        frames=np.asarray(arr)[None],
        frame_meta=(fmeta or FrameMeta(),),
        modality="CR",
        sop_class_uid=CR_SOP_CLASS,
        body_part_examined="CHEST",
        view_position=view if view in ("PA", "AP") else "",
        patient_sex=sex,
        patient_age_years=int(age) if age is not None and not pd.isna(age) else None,
        manufacturer=None,
        manufacturer_model=None,
        source_tags={"image_id": image_id},
        input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


@dataclass
class ExampleOverlay:
    image_id: str
    code: str
    iou: float
    kind: str  # good / bad
    png: str


@dataclass
class LocalizationRun:
    model_version_id: str
    rows: list[RowResult]
    n_images: int
    n_ignored_labels: dict[str, int]
    examples: list[ExampleOverlay] = field(default_factory=list)


def _bootstrap_ci(
    values: NDArray[np.float64], rng: np.random.Generator, n_boot: int
) -> tuple[float, float]:
    if len(values) == 0:
        return (float("nan"), float("nan"))
    if len(values) == 1 or n_boot <= 0:
        return (float(values.mean()), float(values.mean()))
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def _summ(
    rows: Sequence[RowResult], which: str, rng: np.random.Generator, n_boot: int
) -> dict[str, Any]:
    out: dict[str, Any] = {"n": len(rows)}
    for m in ("pointing_hit", "iou", "hit_iou_30", "hit_iou_50", "mass_inside"):
        v = np.asarray([getattr(getattr(r, which), m) for r in rows], dtype=np.float64)
        lo, hi = _bootstrap_ci(v, rng, n_boot)
        out[m] = {
            "mean": float(v.mean()) if len(v) else float("nan"),
            "ci_low": lo,
            "ci_high": hi,
        }
    return out


def summarize(rows: Sequence[RowResult], n_boot: int = 1000, seed: int = 0) -> dict[str, Any]:
    """Per finding (+ all) × {model, random, centre}, overall and stratified by
    `present` (model called it at threshold) and by split tier. Bootstrap 95% CIs."""
    rng = np.random.default_rng(seed)
    codes = [c for c in FINDING_CODES if any(r.code == c for r in rows)]
    groups: dict[str, list[RowResult]] = {"all": list(rows)}
    groups.update({c: [r for r in rows if r.code == c] for c in codes})

    def table(sel: Iterable[RowResult]) -> dict[str, Any]:
        s = list(sel)
        return {w: _summ(s, w, rng, n_boot) for w in ("model", "random", "centre")}

    summary: dict[str, Any] = {"overall": {}, "by_present": {}, "by_tier": {}}
    for g, rs in groups.items():
        summary["overall"][g] = table(rs)
        summary["by_present"][g] = {
            "present": table(r for r in rs if r.present),
            "absent": table(r for r in rs if not r.present),
        }
        summary["by_tier"][g] = {
            t: table(r for r in rs if r.tier == t)
            for t in ("train", "val", "test", "external", "unknown")
            if any(r.tier == t for r in rs)
        }
    summary["gt_area_frac_mean"] = {
        g: float(np.mean([r.gt_area_frac for r in rs])) if rs else float("nan")
        for g, rs in groups.items()
    }
    summary["n_gated_out"] = int(sum(not r.gate_ok for r in rows))
    summary["n_present"] = int(sum(r.present for r in rows))
    return summary


# --------------------------------------------------------------------------- run


def _draw_gt_box(rgb: NDArray[np.uint8], gt: BoundingBox, max_side: int = 512) -> Image.Image:
    im = Image.fromarray(rgb)
    d = ImageDraw.Draw(im)
    h, w = im.height, im.width
    d.rectangle(
        [gt.x_min * w, gt.y_min * h, gt.x_max * w, gt.y_max * h],
        outline=GT_BOX_COLOUR,
        width=max(2, w // 200),
    )
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.Resampling.BILINEAR)
    return im


def evaluate_localization(
    bundle: ModelBundle,
    boxes: pd.DataFrame,
    images_dir: Path | str,
    *,
    entry: pd.DataFrame | None = None,
    tiers: dict[str, str] | None = None,
    out_dir: Path | str | None = None,
    n_examples: int = 3,
    seed: int = 0,
    limit: int | None = None,
    progress: Any = None,
) -> LocalizationRun:
    """Run the plugin path on every boxed image and score the localisation.

    `boxes` is `load_nih_boxes(...)`; one row per (image, finding). Images with several
    of our findings are predicted once. When `out_dir` is given, `n_examples` best and
    worst overlays (by IoU) are written to `<out_dir>/examples/`.
    """
    images_dir = Path(images_dir)
    tiers = tiers or {}
    plugin = CxrPlugin(bundle)
    rng = np.random.default_rng(seed)
    rows: list[RowResult] = []
    codes = list(bundle.calibration.finding_codes)
    image_ids = list(dict.fromkeys(boxes["image_id"].tolist()))
    if limit is not None:
        image_ids = image_ids[:limit]
    # per-row material for the example overlays, kept only for candidates
    overlay_pool: list[tuple[float, RowResult, FloatArr, FloatArr, BoundingBox]] = []

    for k, image_id in enumerate(image_ids):
        path = images_dir / image_id
        if not path.exists():
            continue
        meta_row = entry.loc[image_id] if entry is not None and image_id in entry.index else None
        inp = imaging_input_from_png(path, meta_row, image_id)
        gate = plugin.quality_gate(inp)
        findings = {f.code: f for f in plugin.predict(inp)}
        display = plugin.display_image()
        src_h, src_w = int(inp.frames.shape[1]), int(inp.frames.shape[2])
        for _, b in boxes[boxes["image_id"] == image_id].iterrows():
            code = str(b["code"])
            if code not in codes:
                continue
            f = findings[code]
            assert f.localization is not None and f.localization.heatmap is not None
            heat = np.asarray(f.localization.heatmap, dtype=np.float32)
            gt = pixel_box_to_normalized(
                float(b["x"]), float(b["y"]), float(b["w"]), float(b["h"]), src_h, src_w
            )
            pred_box = f.localization.boxes[0] if f.localization.boxes else None
            row = RowResult(
                image_id=image_id,
                code=code,
                nih_label=str(b["nih_label"]),
                tier=tiers.get(image_id, "unknown"),
                src_h=src_h,
                src_w=src_w,
                gt=(gt.x_min, gt.y_min, gt.x_max, gt.y_max),
                gt_area_frac=box_area(gt),
                probability=f.probability,
                threshold=f.threshold,
                present=f.present,
                gate_ok=gate.ok,
                gate_reasons=gate.reasons,
                pred_box=(
                    (pred_box.x_min, pred_box.y_min, pred_box.x_max, pred_box.y_max)
                    if pred_box
                    else None
                ),
                model=score_heatmap(heat, pred_box, gt),
                random=score_random_baseline(pred_box, gt, src_h, src_w, rng),
                centre=score_centre_prior(gt, src_h, src_w),
            )
            rows.append(row)
            if out_dir is not None and display is not None:
                overlay_pool.append((row.model.iou, row, display.astype(np.float32), heat, gt))
        if progress is not None:
            progress(k + 1, len(image_ids))

    ignored = boxes["nih_label"].value_counts().to_dict() if "nih_label" in boxes else {}
    ignored = {k: int(v) for k, v in ignored.items() if k not in NIH_LABEL_MAP}
    run = LocalizationRun(
        model_version_id=bundle.model_version_id,
        rows=rows,
        n_images=len(set(r.image_id for r in rows)),
        n_ignored_labels=ignored,
    )
    if out_dir is not None and overlay_pool:
        run.examples = _write_examples(overlay_pool, Path(out_dir), n_examples)
    return run


def _write_examples(
    pool: list[tuple[float, RowResult, FloatArr, FloatArr, BoundingBox]],
    out_dir: Path,
    n: int,
) -> list[ExampleOverlay]:
    ex_dir = out_dir / "examples"
    ex_dir.mkdir(parents=True, exist_ok=True)
    # one worst and one best per finding when there are enough findings, else global extremes
    codes = [c for c in FINDING_CODES if any(t[1].code == c for t in pool)]
    picks: list[tuple[str, tuple[float, RowResult, FloatArr, FloatArr, BoundingBox]]] = []
    if len(codes) >= n:
        for c in codes[:n]:
            ordered = sorted((t for t in pool if t[1].code == c), key=lambda t: t[0])
            picks.append(("bad", ordered[0]))
            picks.append(("good", ordered[-1]))
    else:
        ordered = sorted(pool, key=lambda t: t[0])
        picks = [("bad", t) for t in ordered[:n]] + [("good", t) for t in reversed(ordered[-n:])]
    seen: set[str] = set()
    out: list[ExampleOverlay] = []
    for kind, (iou, row, display, heat, gt) in picks:
        name = f"{kind}_{row.code}_{Path(row.image_id).stem}.png"
        if name in seen:
            continue
        seen.add(name)
        label = (
            f"{row.code} p={row.probability:.2f} thr={row.threshold:.2f} "
            f"IoU={iou:.2f} PG={'hit' if row.model.pointing_hit else 'miss'} "
            f"(yellow=model, cyan=radiologist)"
        )
        rgb = render_overlay_rgb(display, heat, label, box=row.pred_box)
        _draw_gt_box(rgb, gt).save(ex_dir / name)
        out.append(
            ExampleOverlay(
                image_id=row.image_id, code=row.code, iou=iou, kind=kind, png=f"examples/{name}"
            )
        )
    return out


# --------------------------------------------------------------------------- report


def _fmt(m: dict[str, Any], pct: bool = True) -> str:
    if not np.isfinite(m["mean"]):
        return "—"
    if pct:
        return f"{100 * m['mean']:.0f}% [{100 * m['ci_low']:.0f}–{100 * m['ci_high']:.0f}]"
    return f"{m['mean']:.3f} [{m['ci_low']:.3f}–{m['ci_high']:.3f}]"


def _table(block: dict[str, Any], groups: list[tuple[str, str]]) -> list[str]:
    """One markdown table; `groups` = [(display label, key into block)]."""
    lines = [
        "| group | finding | source | n | pointing game | mean IoU | hit@IoU≥0.3 | hit@IoU≥0.5 "
        "| heat mass in GT box |",
        "|---|---|---|---:|---|---|---|---|---|",
    ]
    for label, key in groups:
        for code, tbl in block.items():
            sub = tbl.get(key) if key else tbl
            if not sub or sub["model"]["n"] == 0:
                continue
            for src in ("model", "random", "centre"):
                s = sub[src]
                lines.append(
                    f"| {label} | {code} | {src} | {s['n']} | {_fmt(s['pointing_hit'])} "
                    f"| {_fmt(s['iou'], pct=False)} | {_fmt(s['hit_iou_30'])} "
                    f"| {_fmt(s['hit_iou_50'])} | {_fmt(s['mass_inside'])} |"
                )
    return lines


def render_markdown(run: LocalizationRun, summary: dict[str, Any], extra: dict[str, Any]) -> str:
    codes = [c for c in FINDING_CODES if c in summary["overall"]]
    n_by_code = {c: summary["overall"][c]["model"]["n"] for c in codes}
    tier_counts: dict[str, int] = {}
    for r in run.rows:
        tier_counts[r.tier] = tier_counts.get(r.tier, 0) + 1
    lines: list[str] = [
        f"# Localisation evaluation — {extra.get('model_name', run.model_version_id)} "
        f"vs NIH radiologist boxes",
        "",
        f"> **{RESEARCH_USE_LABEL}**",
        "",
        f"- Model version: `{run.model_version_id}`",
        f"- Boxes: `{extra.get('boxes_path', '')}` (NIH ChestX-ray14 `BBox_List_2017.csv`; "
        "NIH-derived, evaluation only, never a training signal)",
        f"- Rows evaluated (image × finding): {len(run.rows)} on {run.n_images} images — "
        + ", ".join(f"{c}={n}" for c, n in n_by_code.items()),
        "- Ignored NIH labels (not our findings): "
        + (", ".join(f"{k}={v}" for k, v in run.n_ignored_labels.items()) or "none"),
        f"- Split tier of the boxed images ({extra.get('splits', 'no splits given')}): "
        + ", ".join(f"{t}={n}" for t, n in sorted(tier_counts.items())),
        f"- Gate: {summary['n_gated_out']} rows would have been `unable_to_assess` "
        "(scored anyway; the box is what a bench user would see)",
        f"- Called present at threshold: {summary['n_present']} / {len(run.rows)} rows",
        "- Mean GT box area fraction: "
        + ", ".join(f"{c}={100 * summary['gt_area_frac_mean'][c]:.1f}%" for c in codes),
        "",
        "## How to read this",
        "",
        "- **pointing game** — heat-map argmax lies inside the radiologist's box.",
        "- **IoU / hit@IoU** — our derived box (largest component ≥ 0.5·max of the Grad-CAM) "
        "vs the radiologist's box.",
        "- **heat mass in GT box** — fraction of heat-map mass inside the radiologist's box.",
        "- **random** — a box of our size at a uniform random position, a random pointing "
        "location, a uniform heat-map (32 draws/row). **centre** — Gaussian centre prior "
        f"(σ = {CENTRE_PRIOR_SIGMA_FRAC:.1f}·short side), boxed by the same rule.",
        "- 95% CIs are percentile bootstrap over rows. Most NIH box images are in this "
        "model's **training** tier; only the `val`/`test` rows are held-out evidence and "
        "they are low-n.",
        "",
        "## Overall",
        "",
        *_table(summary["overall"], [("all rows", "")]),
        "",
        "## By model call at threshold",
        "",
        *_table(
            summary["by_present"], [("called present", "present"), ("called absent", "absent")]
        ),
        "",
        "## By split tier (train = seen in training; val/test = held-out patients)",
        "",
        *_table(
            summary["by_tier"],
            [(t, t) for t in ("train", "val", "test", "external", "unknown")],
        ),
        "",
    ]
    if run.examples:
        lines += ["## Examples (yellow = model box, cyan = radiologist box)", ""]
        for e in run.examples:
            lines.append(f"**{e.kind}** — {e.code} on `{e.image_id}` (IoU {e.iou:.2f})")
            lines.append("")
            lines.append(f"![{e.kind} {e.code}]({e.png})")
            lines.append("")
    lines += [
        "## Caveats",
        "",
        "- NIH boxes are one radiologist's box per finding on ≈ 880 images; there is no "
        "inter-reader agreement to bound the ceiling.",
        "- Grad-CAM resolution is the last DenseNet block (10×10 at 320 px) upsampled, so "
        "boxes are coarse by construction; a small nodule cannot be tightly boxed.",
        "- The derived box is the *largest* high-heat component; a correct secondary peak "
        "on a multi-focal finding is not credited.",
        "",
        f"_{RESEARCH_USE_LABEL}_",
        "",
    ]
    return "\n".join(lines)


def write_report(
    run: LocalizationRun, summary: dict[str, Any], out_dir: Path, extra: dict[str, Any]
) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md = out_dir / "localization_report.md"
    js = out_dir / "localization_report.json"
    md.write_text(render_markdown(run, summary, extra), encoding="utf-8")
    payload = {
        "research_use": RESEARCH_USE_LABEL,
        "model_version_id": run.model_version_id,
        "n_rows": len(run.rows),
        "n_images": run.n_images,
        "n_ignored_labels": run.n_ignored_labels,
        "extra": extra,
        "summary": summary,
        "examples": [asdict(e) for e in run.examples],
        "rows": [r.as_dict() for r in run.rows],
    }
    js.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    return md, js
