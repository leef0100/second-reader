# Training throughput — Apple M1 Pro (16 GB), PyTorch MPS

> For research use only. Not for diagnostic use.

Measured 2026-09-21 on the synthetic 1024×1024 benchmark set (`data/bench-splits/v1`, 488 images),
DenseNet-121, 320 px, batch 16. Commands: `uv run secondreader cxr bench loader ...` and
`uv run secondreader cxr bench model ...` (see `src/secondreader/modalities/cxr/bench.py`).

## Data loader (images/s, augmentation on)

| Path | workers | img/s |
|---|---|---|
| decode 1024² PNG → window → letterbox (no cache) | 0 | 22.8 |
| same | 2 | 27.8 |
| same | 4 | 50.6 |
| same | 8 | 82.0 |
| **uint8 memmap cache** (`secondreader cxr cache`) | 0 | 279.7 |
| cache | 2 | 370.1 |
| cache | 4 | 629.9 |
| cache | 8 | 785.6 |

Decoding full-resolution PNGs on the fly caps an epoch over 112,120 NIH images at ~25–85 min
of pure I/O. The cache (one-time build, ~12 GB at 320 px for NIH) removes the loader from the
critical path entirely.

## Model, fwd+bwd on MPS (batch 16, 320 px)

| Config | train img/s | inference img/s | note |
|---|---|---|---|
| **plain fp32** | 25.6–29.0 | 114–116 | kept (default) |
| fp16 autocast | 15.2–15.6 | 87–89 | slower on MPS — not used |
| channels_last | 13.5 | 78–79 | slower on MPS — not used |
| channels_last + autocast | 4.4 | 51 | loss → NaN — rejected |
| CPU (reference) | 2.2 | 4.7 | |

Peak MPS memory after a training step: ~6.5 GB. Batch 32 was not completed (session limit);
batch 16 is the default.

## Epoch-time estimate

The model is the bottleneck once the cache exists: **112,120 images / ~27 img/s ≈ 69 min per
epoch** on this machine (≈ 11–12 h for 10 epochs on NIH; ≈ 1.5 days for NIH + CheXpert).
Inference-only passes (OOD fitting, evaluation) run at ~114 img/s (≈ 16 min for NIH).

Consequence (D-025): `channels_last` and `autocast` were removed from the config rather than
left as off-by-default knobs — on this hardware they are strictly worse, and a CUDA box should
be benchmarked on its own before re-adding them.
