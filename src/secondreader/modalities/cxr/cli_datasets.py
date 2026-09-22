"""Manifests for the gate-evaluation sets (datasets/OOD_EVAL_PLAN.md).

Kept out of `cli.py` (owned by the training workstream); wiring this Typer app into
`secondreader cxr` (or extending its `manifest` command) is a follow-up. Until then:

    uv run python -m secondreader.modalities.cxr.cli_datasets \
        manifest kermany data/raw/kermany_pediatric
    uv run python -m secondreader.modalities.cxr.cli_datasets manifest nlm-tb data/raw/nlm_tb

For research use only. Not for diagnostic use.
"""

from __future__ import annotations

from pathlib import Path

import typer

from secondreader.core.constants import RESEARCH_USE_LABEL

app = typer.Typer(help=f"Gate-evaluation dataset manifests — {RESEARCH_USE_LABEL}")

_OUT = typer.Option(None, "--out", help="default: datasets/manifests/<dataset id>.csv")


@app.command("manifest")
def manifest_cmd(dataset: str, root: Path, out: Path | None = _OUT) -> None:
    """Build a unified manifest for kermany | nlm-tb (finding labels blank by design)."""
    from secondreader.modalities.cxr import adapters
    from secondreader.modalities.cxr.data import save_manifest

    if dataset == "kermany":
        df = adapters.kermany_manifest(root)
    elif dataset in ("nlm-tb", "nlm_tb"):
        df = adapters.nlm_tb_manifest(root)
    else:
        raise typer.BadParameter("dataset must be kermany|nlm-tb")
    if out is None:
        name = "nlm_tb" if dataset.startswith("nlm") else str(df["dataset"].iloc[0])
        out = Path("datasets/manifests") / f"{name}.csv"
    sha = save_manifest(df, out)
    typer.echo(f"{len(df)} rows → {out} (sha256 {sha[:12]})")
    for ds, n in df["dataset"].value_counts().sort_index().items():
        frontal = int((df.loc[df["dataset"] == ds, "view_label"] == 0).sum())
        typer.echo(f"  {ds:20s} n={int(n):6d}  frontal={frontal}")
    typer.echo("  finding labels: unknown (blank) for every row — gate evaluation only")


if __name__ == "__main__":
    app()
