"""`secondreader cxr ...` commands."""

from __future__ import annotations

from pathlib import Path

import typer

from secondreader.core.constants import RESEARCH_USE_LABEL

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


@app.command("train")
def train_cmd(
    config: Path = Path("configs/experiments/cxr_baseline.yaml"),
    splits: Path = Path("datasets/splits/v1"),
    registry: Path = Path("models"),
    notes: str = "",
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
    mid, metrics = train(
        cfg,
        tr,
        va,
        registry_root=registry,
        split_manifest_sha256=sha,
        datasets_info=info,
        notes=notes,
    )
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


@app.command("manifest")
def manifest_cmd(
    dataset: str, root: Path, out: Path, csv_name: str = "", split: str = "train"
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
    sha = save_manifest(df, out)
    typer.echo(f"{len(df)} rows → {out} (sha256 {sha[:12]})")
