# Gate (OOD / view) evaluation plan

> For research use only. Not for diagnostic use.

Written 2026-09-22. This plan covers the **quality / OOD / view gate** (principle 2,
D-019), not the finding heads. Nothing here trains anything: the sets below are
evaluation-only, their finding labels are blank in the manifests, and none of them may
ever enter `train`/`val` tiers (`role` in `registry.yaml` is the guard).

## Sets and what the gate should do with them

| Set (registry id) | Population | Views | n | Manifest | Gate expectation | Why |
|---|---|---|---|---|---|---|
| `kermany_pediatric` | children 1–5 y (Guangzhou) | AP frontal | 5,856 | `python -m secondreader.modalities.cxr.cli_datasets manifest kermany data/raw/kermany_pediatric` | **reject** (`unable_to_assess`, reason OOD) | intended use is adults (`docs/INTENDED_USE.md`); the view head must still call them *frontal* (`view_label = 0`) — rejection must come from the OOD branch, not a view error |
| `nlm_tb_shenzhen` | adults (Shenzhen No.3 Hospital) | PA | 662 | `… manifest nlm-tb data/raw/nlm_tb` | **accept** | in-population adult frontal films from a site/vendor never seen in training; TB findings are outside our targets but are not a reason to refuse an image |
| `nlm_tb_montgomery` | adults (Montgomery County, MD) | PA | 138 | same manifest | **accept** | as above; very high resolution (4020×4892), different detector |
| CheXpert / PadChest **laterals** (later, after the human click-throughs in `ACQUISITION.md`) | adults | lateral | thousands | `secondreader cxr manifest chexpert|padchest` (existing adapters; `view_label = 1`) | **reject** (reason: view) | laterals are out of scope; this is the view head's job, not Mahalanobis |
| Internal validation tier (NIH, D-008) | adults | PA/AP | ~11k | existing splits | accept ≈ 99 % | the threshold is *defined* on this tier (D-019), so its rejection rate is the reference point |

`dataset` is the grouping key for every table below; `kermany_pediatric` is additionally
labelled `population: paediatric` in `registry.yaml` so reports can call the group
"out-of-population" rather than "out-of-distribution by accident".

## Metrics to report (per set, with 95 % CIs)

Rejection is a binary decision at the deployed threshold (`gate.ood_percentile`, default
99th percentile of internal-validation distances, D-019), so the primary numbers are
rates at that fixed threshold; curves are secondary and must never be used to move the
threshold *after* looking at these sets.

1. **Rejection rate at the deployed threshold** per set, with a Wilson 95 % CI, split by
   reason (`ood`, `view`, `quality`, combination). Targets: `kermany_pediatric` high
   (report the number; a plausible bar is ≥ 90 %), NLM sets low (within a few points of
   the internal-validation rate; a bar of ≤ 5 % absolute above it), laterals ≈ 100 % by
   the view reason.
2. **OOD separability**: AUROC of the Mahalanobis distance, internal-validation
   (negative) vs `kermany_pediatric` (positive), and the same vs each NLM set (where a
   *low* AUROC is the good outcome — they should look in-distribution). Bootstrap CIs
   over images. Also report the distance distribution summary (median, 90th, 99th
   percentile) per set so drift monitoring has a reference.
3. **View head**: accuracy / confusion matrix frontal-vs-lateral per set with the
   truth from the manifest (`view_label`); on the paediatric set the frontal call must
   hold (report the fraction called frontal — it is a failure mode if rejection comes
   from the view branch).
4. **Quality sub-gate** (burned-in text, exposure, size): rejection rate per set and
   reason, so a high paediatric rejection is not silently caused by, e.g., image size.
5. **Subgroups** (principle 1): for the NLM sets, by `sex` and age band (both parsed
   from the clinical readings; ages missing → an explicit "unavailable" row, D-010); for
   all sets by source site. The paediatric set has no per-image age/sex — say so.
6. **Downstream sanity on accepted images** (no labels, so no accuracy): distribution
   of calibrated probabilities per finding on accepted NLM images vs internal validation
   (the drift monitor's output check), to catch a gate that accepts films the model then
   scores nonsensically.
7. **Throughput**: images/s for the gate alone on the Montgomery originals (largest
   files) — it must not be the bottleneck.

Every table carries `n`, and a `low_n` flag when the group has < 100 images
(Montgomery will).

## Procedure

1. `scripts/download_ood_sets.sh` (resumable; log `data/raw/ood_download.log`).
2. Build the two manifests (commands above) → `datasets/manifests/{kermany_pediatric,nlm_tb}.csv`
   (git-ignored); record their SHA-256 in the evaluation report.
3. Run the gate only (no finding heads needed) on each manifest with the **deployed**
   model version's gate statistics (`models/<id>/`), using the gate-evaluation entry
   point in the CXR CLI (follow-up: `secondreader cxr evaluate --gate-only <manifest>`
   or equivalent in `modalities/cxr/evaluate.py`, owned by the model workstream).
4. Write `docs/reports/gate_eval_<model id>/report.md` with the tables above; add the
   headline numbers as a D-entry in `DECISIONS.md` and update the model card.
5. If the paediatric rejection rate is low, the fix is a *policy* discussion (age
   metadata gate, different feature layer for Mahalanobis, or a paediatric-vs-adult
   head), recorded as a decision — not a threshold tweak on this set.

## What this plan does not do

* No training on any of these sets; no threshold tuning on them (D-019).
* No claim about TB detection — the TB labels are ignored by design.
* No external-test claim for the finding heads; that remains VinDr-CXR / PadChest (D-008).
