"""`secondreader cxr ...` commands."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import typer

from secondreader.core.constants import RESEARCH_USE_LABEL

if TYPE_CHECKING:
    from secondreader.modalities.cxr.cache import FrameCache
    from secondreader.modalities.cxr.config import PreprocessConfig

app = typer.Typer(help=f"Chest X-ray plugin — {RESEARCH_USE_LABEL}", no_args_is_help=True)


@app.command("synth")
def synth(
    out: Path = Path("data/synthetic"), n_patients: int = 150, size: int = 256, seed: int = 0
) -> None:
    """Generate the synthetic smoke-test dataset (train source A + shifted external source B)."""
    from secondreader.modalities.cxr.synthetic import generate

    a = generate(out / "synth-a", n_patients=n_patients, size=size, seed=seed, dataset="synth-a")
    b = generate(
        out / "synth-b",
        n_patients=max(20, n_patients // 3),
        size=size + 64,
        seed=seed + 1,
        dataset="synth-b",
        contrast=0.8,
        noise=0.04,
    )
    typer.echo(f"synth-a: {len(a)} images, synth-b (external): {len(b)} images → {out}")


@app.command("split")
def split(
    manifests: list[Path],
    out: Path = Path("datasets/splits"),
    version: str = "v1",
    salt: str = "second-reader-v1",
    external: str = "vindr_cxr,padchest,brax,synth-b",
) -> None:
    """Build patient-level hashed splits from one or more manifests."""
    import pandas as pd

    from secondreader.modalities.cxr.data import load_manifest
    from secondreader.modalities.cxr.splits import build_splits

    df = pd.concat([load_manifest(m) for m in manifests], ignore_index=True)
    s = build_splits(
        df, out_dir=out, version=version, salt=salt, external_datasets=set(external.split(","))
    )
    typer.echo(f"splits {version}: " + ", ".join(f"{t}={v['n']}" for t, v in s.tiers.items()))
    typer.echo(f"manifest sha256: {s.manifest_sha256}  (record in DECISIONS.md when locking)")


def _open_caches(
    cache: Path | None, splits: Path, pre: PreprocessConfig, tiers: list[str]
) -> dict[str, FrameCache]:
    """Open `<cache>/<tier>` for each tier, refusing caches built from other manifests/configs."""
    from secondreader.modalities.cxr.cache import FrameCache
    from secondreader.modalities.cxr.data import manifest_sha256

    out: dict[str, FrameCache] = {}
    if cache is None:
        return out
    for t in tiers:
        fc = FrameCache.open(
            cache / t, pre=pre, manifest_sha256=manifest_sha256(splits / f"{t}.csv")
        )
        typer.echo(f"cache {t}: {len(fc)} images at {fc.size}px ({fc.dir})")
        out[t] = fc
    return out


@app.command("train")
def train_cmd(
    config: Path = Path("configs/experiments/cxr_baseline.yaml"),
    splits: Path = Path("datasets/splits/v1"),
    registry: Path = Path("models"),
    notes: str = "",
    cache: Path | None = typer.Option(
        None, help="preprocessing cache root (from `cxr cache`) holding train/ and val/"
    ),
    resume: bool = typer.Option(False, help="resume from <registry>/pending/checkpoint.pt"),
) -> None:
    """Train, calibrate, fit OOD, and write a versioned model bundle."""
    import json

    import yaml

    from secondreader.modalities.cxr.config import CxrConfig
    from secondreader.modalities.cxr.splits import load_tier
    from secondreader.modalities.cxr.train import train

    cfg = CxrConfig.load(config)
    tr, va = load_tier(splits, "train"), load_tier(splits, "val")
    sha = (splits / "manifest.sha256").read_text().strip()
    caches = _open_caches(cache, splits, cfg.preprocess, ["train", "val"])
    reg = yaml.safe_load(Path("datasets/registry.yaml").read_text())["datasets"]
    used = sorted(set(tr["dataset"]) | set(va["dataset"]))
    info = [
        {
            "name": d,
            "license": reg.get(d, {}).get("license", "synthetic/none"),
            "commercial_ok": reg.get(d, {}).get("commercial_ok", d.startswith("synth")),
        }
        for d in used
    ]
    try:
        mid, metrics = train(
            cfg,
            tr,
            va,
            registry_root=registry,
            split_manifest_sha256=sha,
            datasets_info=info,
            notes=notes,
            train_cache=caches.get("train"),
            val_cache=caches.get("val"),
            resume=resume,
        )
    except KeyboardInterrupt:
        typer.echo(
            f"interrupted — checkpoint saved to {registry / 'pending' / 'checkpoint.pt'}; "
            "rerun the same command with --resume",
            err=True,
        )
        raise typer.Exit(code=130) from None
    typer.echo(f"model_version_id: {mid}")
    typer.echo(json.dumps(metrics["val"], indent=2))


@app.command("eval")
def eval_cmd(
    model_id: str,
    tier: str = "external",
    splits: Path = Path("datasets/splits/v1"),
    registry: Path = Path("models"),
    out: Path = Path("runs"),
    n_boot: int = 200,
    no_gate: bool = False,
) -> None:
    """Evaluate a bundle on a split tier (default: the locked external test set)."""
    from secondreader.modalities.cxr.evaluate import evaluate
    from secondreader.modalities.cxr.model import ModelBundle
    from secondreader.modalities.cxr.splits import load_tier

    bundle = ModelBundle.load(registry / model_id)
    df = load_tier(splits, tier)
    sha = (splits / "manifest.sha256").read_text().strip()
    out_dir = out / model_id / tier
    rep = evaluate(
        bundle,
        df,
        tier=tier,
        dataset_name=",".join(sorted(df["dataset"].unique())),
        split_manifest_sha256=sha,
        out_dir=out_dir,
        n_boot=n_boot,
        apply_gate=not no_gate,
    )
    for code, m in rep.overall.items():
        typer.echo(
            f"{code:18s} AUROC={m.auroc} sens={m.sensitivity} spec={m.specificity} ECE={m.ece}"
        )
    typer.echo(f"report: {out_dir / 'report.md'}  (gated out: {rep.n_gated_out})")


_OUT_OPTION = typer.Option(None, "--out", help="default: datasets/manifests/<dataset id>.csv")


@app.command("manifest")
def manifest_cmd(
    dataset: str,
    root: Path,
    out: Path | None = _OUT_OPTION,
    csv_name: str = "",
    split: str = "train",
) -> None:
    """Build a unified manifest from a dataset's native CSVs (nih|chexpert|vindr|padchest)."""
    from secondreader.modalities.cxr import adapters
    from secondreader.modalities.cxr.data import save_manifest

    if dataset == "nih":
        df = adapters.nih_manifest(root)
    elif dataset == "chexpert":
        df = adapters.chexpert_manifest(root, csv_name or "train.csv")
    elif dataset == "vindr":
        df = adapters.vindr_manifest(root, split)
    elif dataset == "padchest":
        df = adapters.padchest_manifest(
            root, csv_name or "PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv"
        )
    else:
        raise typer.BadParameter("dataset must be nih|chexpert|vindr|padchest")
    if out is None:
        out = Path("datasets/manifests") / f"{df['dataset'].iloc[0]}.csv"
    sha = save_manifest(df, out)
    typer.echo(f"{len(df)} rows → {out} (sha256 {sha[:12]})")
    from secondreader.modalities.cxr.config import FINDING_CODES

    for code in FINDING_CODES:
        pos = int((df[code] == 1).sum())
        typer.echo(f"  {code:18s} positives={pos:7d}  prevalence={pos / max(len(df), 1):.4f}")


@app.command("cache")
def cache_cmd(
    splits: Path = Path("datasets/splits/v1"),
    out: Path = Path("data/cache/v1"),
    config: Path | None = typer.Option(
        None, help="experiment YAML; its `preprocess` section replaces --size/--window-*"
    ),
    size: int = 320,
    window_low: float = 0.5,
    window_high: float = 99.5,
    tiers: str = "train,val,test,external",
    workers: int = typer.Option(0, help="decode processes (0 = cpu_count - 2, max 8)"),
    limit: int | None = typer.Option(None, help="only the first N rows per tier (smoke runs)"),
    force: bool = False,
) -> None:
    """Decode + window + letterbox every image of each split tier once into a uint8 memmap.

    The cache is keyed on (tier manifest sha256, preprocess config); `train --cache` refuses a
    stale one. Pass --config so the preprocess section is taken from the experiment YAML, or
    use the same --size/--window-* values as that section.
    """
    import os

    from secondreader.modalities.cxr.cache import FrameCache, StaleCacheError, build_cache
    from secondreader.modalities.cxr.config import CxrConfig, PreprocessConfig
    from secondreader.modalities.cxr.data import load_manifest, manifest_sha256

    if config is not None:
        pre = CxrConfig.load(config).preprocess
    else:
        pre = PreprocessConfig(
            image_size=size, window_low_pct=window_low, window_high_pct=window_high
        )
    typer.echo(f"preprocess: {pre.model_dump()}")
    if workers <= 0:
        workers = max(1, min(8, (os.cpu_count() or 2) - 2))
    for tier in [t for t in tiers.split(",") if t]:
        csv = splits / f"{tier}.csv"
        if not csv.exists():
            typer.echo(f"{tier}: no {csv}, skipped")
            continue
        df = load_manifest(csv)
        if not len(df):
            typer.echo(f"{tier}: empty, skipped")
            continue
        sha = manifest_sha256(csv)
        tier_dir = out / tier
        if not force and (tier_dir / "cache.json").exists():
            try:
                fc = FrameCache.open(tier_dir, pre=pre, manifest_sha256=sha)
                if fc.meta.get("limit") == limit:
                    typer.echo(f"{tier}: up to date ({len(fc)} images), skipped (use --force)")
                    continue
            except StaleCacheError as e:
                typer.echo(f"{tier}: stale cache, rebuilding ({e})")
        t0 = time.perf_counter()
        meta = build_cache(
            df, manifest_sha256=sha, pre=pre, out_dir=tier_dir, workers=workers, limit=limit
        )
        dt = time.perf_counter() - t0
        typer.echo(
            f"{tier}: {meta['n']} images → {tier_dir} in {dt:.0f}s "
            f"({meta['n'] / max(dt, 1e-9):.1f} img/s, {meta['n_failed']} failed)"
        )


bench_app = typer.Typer(help="Throughput micro-benchmarks", no_args_is_help=True)
app.add_typer(bench_app, name="bench")


@bench_app.command("loader")
def bench_loader_cmd(
    splits: Path = Path("datasets/splits/v1"),
    tier: str = "train",
    cache: Path | None = None,
    size: int = 320,
    batch_size: int = 16,
    num_workers: int = 2,
    steps: int = 30,
    augment: bool = True,
    limit: int | None = None,
) -> None:
    """DataLoader images/s for a split tier, with or without a preprocessing cache."""
    import json

    from secondreader.modalities.cxr.bench import bench_loader
    from secondreader.modalities.cxr.config import PreprocessConfig
    from secondreader.modalities.cxr.data import ManifestDataset
    from secondreader.modalities.cxr.splits import load_tier

    pre = PreprocessConfig(image_size=size)
    df = load_tier(splits, tier)
    if limit:
        df = df.iloc[:limit]
    fc = _open_caches(cache, splits, pre, [tier]).get(tier)
    ds = ManifestDataset(df, pre, augment=augment, cache=fc)
    typer.echo(
        json.dumps(bench_loader(ds, batch_size=batch_size, num_workers=num_workers, steps=steps))
    )


@bench_app.command("model")
def bench_model_cmd(
    size: int = 320,
    batch_size: int = 16,
    device: str = "auto",
    channels_last: bool = False,
    autocast: bool = False,
    steps: int = 20,
    backbone: str = "densenet121",
) -> None:
    """DenseNet forward/backward images/s on the selected device (synthetic tensors)."""
    import json

    from secondreader.modalities.cxr.bench import bench_model

    typer.echo(
        json.dumps(
            bench_model(
                size=size,
                batch_size=batch_size,
                device=device,
                channels_last=channels_last,
                autocast=autocast,
                steps=steps,
                backbone=backbone,
            )
        )
    )
