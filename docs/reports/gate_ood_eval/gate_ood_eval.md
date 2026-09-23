# Gate evaluation on out-of-scope / in-scope unlabelled sets

> For research use only. Not for diagnostic use.

Model `d3dd60f2f596b2ed` (OOD threshold = 99th percentile of NIH validation distances). Expected: paediatric (Kermany) → rejected; NLM Shenzhen/Montgomery adult frontal → passed.

| dataset | n | passed | pass rate | median OOD distance (thr) | rejected: OOD | aspect ratio | too small | saturated |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| kermany_pediatric | 5856 | 4418 | 75.4% | 39.9 (44.3) | 1153 | 424 | 58 | 1 |
| nlm_tb_montgomery | 138 | 133 | 96.4% | 30.4 (44.3) | 1 | 0 | 0 | 4 |
| nlm_tb_shenzhen | 662 | 653 | 98.6% | 29.9 (44.3) | 9 | 0 | 0 | 0 |

## Verdict

- **Adult TB sets (in-scope population, different sites/scanners): 96–99% pass** — the gate is not over-rejecting plausible adult frontal films from new sites.
- **Paediatric (out-of-scope population): only 24.6% rejected.** The feature-space OOD score separates paediatric from adult films only partially (median distance 39.9 vs threshold 44.3); most rejections come from the geometric checks (extreme aspect ratios, tiny images), not from the learned score. The current gate therefore does **not** reliably enforce the adult-only intended use.
- Consequences: (1) `docs/INTENDED_USE.md` states paediatric exclusion is not yet validated — now measured; (2) next step is an explicit population/age gate (train the plugin's `other` view class on paediatric data, or adopt an open adult-vs-paediatric classifier per docs/research/open_source_calibration_ood_xai.md) and re-run this script; (3) until then, deployments must rely on DICOM PatientAge (≥18) as the primary guard, which the ingest pipeline preserves.

Reproduce: `uv run python scripts/eval_gate_ood.py <model_id> datasets/manifests/kermany_pediatric.csv datasets/manifests/nlm_tb.csv`
