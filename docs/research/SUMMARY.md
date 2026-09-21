# Research summary for the pitch

> **For research use only. Not for diagnostic use.**
> One-page digest of `market.md`, `competitors.md`, `data_landscape.md` and `evidence_bar.md` (all in this folder). Every fact below is sourced in the section referenced; access date 2026-09-21. Do not quote a number from here without the citation in the underlying file.

## Top 10 facts for the pitch

1. Independent analyst Signify Research sizes AI medical-image analysis at $576M (2022) growing ~25%/yr to $1.73B by 2027; pulmonology was $88M in 2022. Other houses range from $0.5B to $14.6B for "the same" market because they count different things — quote Signify. (market.md §1)
2. FDA authorised 331 AI devices in 2025, the most ever; 1,104 of 1,451 all-time AI devices (76%) are radiology. (market.md §1)
3. There is **no CPT code and no NTAP for chest X-ray AI**; as of January 2026 only two Category I codes exist for newer imaging AI (coronary plaque 75577, FFR-CT 75580). CMS's July 2026 "Software as a Medical Service" proposal is the first generic payment vehicle. (market.md §2)
4. Radiologist supply and imaging demand are projected to grow at similar rates to 2055 (supply +25.7%, demand +16.9–26.9%), so the shortage persists; attrition doubled 2014→2022 (1.1%→2.5%); UK consultant shortfall is 29% rising to 39% by 2029. (market.md §3)
5. Missed lung cancer on CXR: 19% of nodular NSCLC missed (median 16 mm), median delay 472 days, 43% up-staged T1→T2 (Quekel 1999); in a 2,407-CXR multi-site audit, nodules were 43% of clinically important missed findings. (market.md §4)
6. Every FDA-cleared CXR product is either a worklist-triage tool (QFM; PTX/effusion; no reader study required) or a CADe with an MRMC study (Gleamer ChestView, Qure qXR-LN/qXR-Detect, AZmed). Second Reader is a CADe; the direct comparators are Gleamer and Qure. (competitors.md §1–2)
7. Only Qure.ai and Annalise describe subgroup analysis in their public 510(k) summaries; **nobody** publishes calibration, per-site performance, or AP-vs-PA performance — the exact things Second Reader's principles 1, 2 and 8 mandate. (competitors.md §3; evidence_bar.md §3)
8. Independent external validations are sobering: a commercial tool scored AUROC 0.648 / sensitivity 35% at 2.2% prevalence; four tools had sensitivity 9–94% for small PTX/effusion; seven lung-cancer devices had sensitivity 20.8–77.8% with κ=0.24 agreement; radiograph view (AP vs PA) explained 69–87% of performance variance. (competitors.md §4)
9. The largest RCT of CXR AI (LungIMPACT, 93,326 CXRs, NHS) found **no reduction** in time to CT or lung-cancer diagnosis from AI prioritisation. Triage is not a proven outcome lever; catching misses is a different claim. (market.md §4.4)
10. Public data can take us past 1M training images only under research-only licences (NIH + CheXpert + MIMIC + PadChest + ReXGradient-160K); the commercial-clean public corpus is ~118k adult images (NIH + TBX11K). ReXGradient-160K (79 US sites, held-out private test) and MIDRC (commercial DUA, sequestered test commons) are the two new assets for multi-site external testing. (data_landscape.md §2–3)

## Top 5 risks

1. **No reimbursement path** — CXR CADe will be bought from hospital operating budgets, and Radiology Business reports that lung-nodule/incidental-finding AI is unlikely to get its own paid code. Behold.ai (FDA-cleared) went into administration in 2025 after losing NHS procurement. (market.md §2; competitors.md §5)
2. **Crowded, consolidating category** — Qure.ai now holds a six-category CXR CADe clearance with a PCCP; Gleamer/DeepHealth, AZmed and Annalise are cleared; Sectra bought Oxipit; PACS vendors will bundle. A three-finding product must win on trust/validation, not breadth. (competitors.md §2)
3. **Generalisation failure is the norm** — every independent study shows large drops on new sites, small findings, PA views and multi-pathology films; our own model will not be exempt, and the external test may show numbers well below the 510(k) headlines. (competitors.md §4)
4. **Licence contamination** — CheXpert, MIMIC, PadChest, VinDr, BRAX, ReXGradient (per its HF agreement) are research-only; a commercial retrain must exclude them or re-license, and NIH ChestX-ray14 was used in Lunit's FDA test set. Data brokers publish no terms. (data_landscape.md §3–4)
5. **Regulatory bar is rising, not falling** — FDA denied Harrison.ai's petition to exempt CADe/CADt from 510(k) (April 2026) and its 2025 draft guidance asks for stratified subgroup performance; expect a full standalone + MRMC package. (competitors.md §2 Annalise row; evidence_bar.md §1)

## 5 concrete recommendations

1. **Position as "missed-finding safety net", not triage.** Lead with Quekel/Kaviani miss rates and the read-first-then-reveal design; do not claim faster diagnosis (LungIMPACT). Make nodule/mass the headline finding. (market.md §4; INTENDED_USE.md §1)
2. **Make the validation report the product.** Publish a locked external test (≥3,000 consecutive CXRs, ≥4 sites, ≥2 countries, ≥300 positives per finding) with per-site, per-vendor, AP/PA, size-stratified and calibration (ECE) tables with CIs — the table no competitor publishes. Size and design per evidence_bar.md §4.
3. **Secure two multi-site external test sources now:** apply for ReXGradient-160K (and ask Gradient Health in writing whether commercial use is permitted, since their HF agreement and arXiv paper disagree) and register for MIDRC's commercial DUA / sequestered commons; add PLCO's 25k subset for outcome-labelled nodules. (data_landscape.md §2–3)
4. **Keep two model lineages from day one:** a research lineage trained on everything, and a commercial-clean lineage trained only on `commercial_ok: true` data plus licensed hospital data, so the registry's `commercial_ok` gate is exercised before it matters. Get quotes from Gradient Health, Segmed and Protege with explicit questions on de-identification method, model-commercialisation rights and site identifiers. (data_landscape.md §3.3, §4)
5. **Plan the regulatory package as CADe (MYN), not triage:** pre-register a fully crossed MRMC (≥12 readers incl. ED physicians, ≥250 cases, ≥4-week washout, wAFROC primary) whose reading protocol mirrors read-first-then-reveal, and add a PCCP for re-calibration per site, following Qure's precedent. Budget for the January 2025 draft guidance's subgroup expectations. (evidence_bar.md §1, §4)
