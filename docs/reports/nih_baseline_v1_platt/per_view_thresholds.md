# Per-view operating thresholds — `e4262a38b40fbe2c` vs global `cc2f56aa3325788c`

> **For research use only. Not for diagnostic use.**

`e4262a38b40fbe2c` = `secondreader cxr recalibrate cc2f56aa3325788c --per-view` (same weights,
same Platt scale/bias, Youden threshold chosen separately for PA and AP on `nih-v1/val`;
D-029). Evaluated on `nih-v1/test` on M1 Pro MPS from the cache (2.5 min). Full report:
`per_view_test_report.md`; calibration: `per_view_calibration.json`.

## Thresholds (validation, Youden)

| finding | global | PA | AP |
|---|---|---|---|
| pneumothorax | 0.0398 | **0.0710** | 0.0359 |
| pleural effusion | 0.0989 | **0.1223** | 0.0976 |
| nodule/mass | 0.0974 | **0.1142** | 0.1013 |

Youden on the pooled validation set lands close to the AP optimum because AP films have
lower scores overall (the model is less confident on portable films); the PA optimum sits
markedly higher for every finding. So "per view" in practice means "raise the PA threshold";
AP barely moves.

## Test tier (`nih-v1/test`, 11.3k images), sens / spec at the operating point

AUROC is unchanged by construction (same weights and calibration) — 0.883 / 0.886 / 0.795 —
and per-view AUROC is identical too (AP 0.862 / 0.848 / 0.738, PA 0.895 / 0.905 / 0.825).

| finding | view | global thr: sens | spec | per-view thr: sens | spec | Δ sens | Δ spec |
|---|---|---|---|---|---|---|---|
| pneumothorax | AP | 0.760 [0.705–0.822] | 0.780 | 0.788 [0.740–0.841] | 0.751 | +0.028 | −0.029 |
| pneumothorax | PA | 0.850 [0.812–0.883] | 0.783 | 0.771 [0.723–0.813] | 0.862 | −0.079 | +0.079 |
| pleural effusion | AP | 0.866 | 0.674 | 0.872 | 0.666 | +0.006 | −0.008 |
| pleural effusion | PA | 0.851 | 0.818 | 0.823 | 0.843 | −0.028 | +0.025 |
| nodule/mass | AP | 0.562 | 0.761 | 0.547 | 0.773 | −0.015 | +0.012 |
| nodule/mass | PA | 0.766 | 0.745 | 0.723 | 0.798 | −0.043 | +0.053 |
| **all** | pneumothorax | 0.815 | 0.782 | 0.778 | 0.818 | −0.037 | +0.036 |
| **all** | pleural effusion | 0.858 | 0.764 | 0.848 | 0.777 | −0.010 | +0.013 |
| **all** | nodule/mass | 0.697 | 0.751 | 0.663 | 0.788 | −0.034 | +0.037 |

(Global-threshold rows from `report.md` in this folder; per-view rows from
`per_view_test_report.md`. The two runs gate 148 vs 145 images because the per-view run reads
pixel statistics from the 8-bit cache instead of re-decoding, so three borderline saturation
cases flip; n differs by 3.)

## Reading

1. **Per-view thresholds do not close the AP-vs-PA gap.** The gap is discrimination (AUROC
   0.74 vs 0.83 on nodule/mass), and a threshold only slides each view along its own ROC
   curve. What per-view Youden does is make the *balance* of errors view-consistent: with a
   global threshold PA runs sensitivity-heavy (0.85 / 0.78 on pneumothorax) and AP
   specificity-heavy; per view both sit near their own Youden point. Whether that is wanted is
   a policy question, not a metric one.
2. **Youden is the wrong criterion to ship for a second reader**, per view or not: it moved
   PA pneumothorax sensitivity from 0.85 to 0.77 to buy specificity. For a missed-findings aid
   the operating point should be `sens90` (`train.operating_point: sens90`), where a per-view
   threshold has a clear benefit — it keeps the 90 % sensitivity promise on AP films, which a
   pooled threshold silently breaks (AP pneumothorax at the global threshold is 0.76
   sensitive). This is the recommended configuration for the next recalibration; not run
   here because the intent was to A/B the mechanism at the existing operating point.
3. Calibration is untouched (ECE eq-width / eq-mass ≤ 0.011 overall, Brier
   0.036 / 0.078 / 0.073); only where the line is drawn changed, and the SR/JSON now carry the
   view-specific threshold next to the probability so a clinician sees which line applied.
4. Unknown / missing `ViewPosition` falls back to the global threshold — recorded per finding
   in `GateResult`-adjacent output (`Finding.threshold`).

## Decision

Per-view thresholds are implemented and validated as a mechanism (D-029). The active model
stays `cc2f56aa3325788c` (global Youden) until a `sens90` per-view recalibration is run and
reviewed, because at Youden the change mainly re-balances PA rather than helping AP.
