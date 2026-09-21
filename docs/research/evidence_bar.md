# Evidence bar: what FDA-cleared CXR AI actually submitted

> **For research use only. Not for diagnostic use.**
> Due-diligence research for the Second Reader project. Everything in §2 is transcribed from the FDA 510(k) summaries (primary sources, URLs at the end); §1 from FDA guidance text. All URLs accessed 2026-09-21.

## 1. What FDA asks for

### 1.1 CADe (21 CFR 892.2070, product code MYN) — our category
FDA's *Clinical Performance Assessment: Considerations for CADe Devices ... in 510(k) Submissions* (issued 2022-09-28, superseding 2012/2020 versions) recommends: [1]
- **Standalone performance** and a **multiple-reader multiple-case (MRMC) reader study**, ideally fully crossed ("offers the greatest statistical power for a given number of cases"); endpoints based on ROC/FROC/LROC plus sensitivity and specificity at a clinical action point.
- Study designs: prospective field test, retrospective enriched reader study, or "stress test" enriched with challenging cases within the intended-use population; enrichment "based on the performance of the CADe device" is explicitly discouraged.
- **Study population**: consecutive collection under pre-specified inclusion/exclusion criteria from each site; "cases collected for your clinical trial should be independent of the cases used during your device development"; sample size powered for the claims; "a sufficient number of cases from important cohorts (e.g., subsets defined by clinically relevant confounders, effect modifiers, and concomitant diseases) such that clinical performance estimates and confidence intervals can be obtained for these individual subsets"; subsets need not be individually powered unless subset claims are made, in which case multiplicity adjustment is required.
- **Reporting** of case distributions stratified by lesion type, lesion size, lesion location, disease stage, organ characteristics, concomitant diseases, **imaging hardware (makes and models)**, imaging protocols, collection sites, and patient demographics (age, ethnicity, race, sex).
- **Data poolability**: justify statistically and clinically that multi-site (and foreign) data can be pooled; foreign data must be shown applicable to US population and practice.
- **Test-data reuse** is discouraged once any algorithm training has occurred; if reused, a firewall between developers and test results is expected.
- Reference standard, reader training and scoring rules must be pre-specified; a full protocol and statistical analysis plan go in the submission.

The January 2025 draft guidance on AI-enabled device software functions additionally recommends stratified subgroup analyses (race, sex, age) in performance validation. [2]

### 1.2 CADt (21 CFR 892.2080, QFM/QAS) — the triage products
Special-control practice visible in every summary below: standalone AUC > 0.95 with sensitivity and specificity > 80% (lower 95% CI bounds), a ground truth from ≥2 ABR-certified radiologists with a third adjudicator, and a bench "time-to-notification"/turnaround study; **no reader study is required**. [3][4][5]

## 2. What was actually submitted (chest X-ray products)

| Device (K-number, year) | Category | Standalone test set | Sites / geography | Reference standard | Headline result | Subgroups reported in public summary | Reader study |
|---|---|---|---|---|---|---|---|
| Zebra HealthPNX (K190362, 2019) [6] | CADt PTX | 588 (146+/442−) | USA + Israel (count of sites not stated) | 3 US board-certified radiologists | AUC 0.983 (0.974–0.990); Se 93.15%; Sp 92.99% | None | Yes, but for triage time only: 3 readers, prioritised vs FIFO queue, 8.05 vs 68.98 min |
| Behold.ai red dot (K191556, 2020) [3] | CADt PTX | 888 (299+/589−) | **4 sites: 2 US (n=738), 2 UK (n=150)**; per-site counts published | ≥2 ABR radiologists + third | AUROC 0.975 (0.966–0.984); Se 94.65%; Sp 87.95% | Per-site counts only | None (time study: 29.3 s total) |
| Lunit INSIGHT CXR Triage (K211733, 2021) [7] | CADt PTX + effusion | Internal 1,385; pivotal 1,708 (754 effusion, 954 PTX) | **NIH ChestX-ray14 (public) + Indian institutions (6 sites effusion, 3 PTX)** | Board-certified radiologists (number not stated) | Effusion AUC 0.9686, Se 89.86%, Sp 93.48%; PTX AUC 0.9630, Se 88.92%, Sp 90.51% | None | None (time 20.5–20.8 s) |
| Annalise Enterprise CXR Triage PTX (K213941, 2022) [4] | CADt PTX + tension PTX | 949 (413 PTX+, 123 tension; 536−) | **4 US hospital-network sites**, consecutive; 10 manufacturers, 22 models; erect + supine; AP/PA/lateral | ≥2 ABR radiologists + third | AUC 0.979 (0.970–0.986) PTX, 0.988 tension; 3 operating points (Se 89.1–96.6%) | Sex (45% F), age (mean 62.3, range 22–99), ethnicity (7.7% Hispanic), race (83% White, 5.5% African American, 3.3% Asian) — composition only, not per-subgroup AUC | None (bench TAT 20.57 s on 621 positives) |
| Aidoc BriefCase PTX (K214043, 2022) [5] | CADt PTX | 619 (139+/480−) | **5 sites (4 US, 1 non-US)**, 89% US | 2 radiologists + third | AUC 0.969 (0.954–0.985); Se 94.2%; Sp 90.8%; NPV 99.3%, PPV 53.3% | Manufacturer table (6 vendors); comorbid-pathology table for negatives; sex 51% M; "ethnicity ... unavailable" | None (TTN 13.1 s) |
| Annalise CXR Triage Trauma / VCF (K222268, 2022) [8] | CADt VCF | 589 (272+/317−) | 4 US sites, consecutive | ≥2 ABR + third | AUC 0.954; Se 89.3% / Sp 89.0% | "representation across subgroups for patient demographics (gender, age, ethnicity, race), technical parameters (make, model), imaging parameters (positioning, projections)"; erect-only caveat in IFU | None |
| Qure.ai qXR-PTX-PE (K230899, 2023) [9] | CADt PTX + effusion | PTX 613 (201+/412−); effusion 1,070 (344+/726−) | US, multi-region (Midwest/West/Northeast/South), multiple manufacturers; **test sites disjoint from training sites**; protocol pre-specified subgroup quotas | 3 ABR thoracic radiologists (≥10 y) | PTX AUC 0.9894, Se 94.53%, Sp 96.36%; effusion AUC 0.989, Se 96.22%, Sp 94.90% | **AUC by sex and by 4 age bands, both findings** | None (10 s) |
| Qure.ai qXR-LN (K231805, 2023) [10] | **CADe nodule 6–30 mm** | Not stated as a count in the summary; "40 sites" in 8 US states; 55% female; 31% portable; >10 manufacturers; confounders included | 40 US sites | **5 ABR radiologists reading CXR with paired CT and reports** | Nodule-level Se 84.1%, FPPI 0.18; scan-level AUC 0.9451, Se 93.83%, Sp 81.09% | age, sex, race, nodule characteristics, manufacturer, nodule location, view, confounders (standalone and MRMC) | **Yes**: fully crossed sequential second-read MRMC; radiologists, pulmonologists, ER physicians (<3, 3–7, >7 y); AFROC 0.7333 → 0.8095 (Δ 0.0762, p<1e-5); AUC Δ 0.0697; nodule Se +11.96% |
| VUNO Med-Chest X-ray Triage (K241439, 2024) [11] | CADt PTX + effusion | Not captured in our extract | Training: 4 sites Korea + India | — | PTX AUC 0.9883, Se 95.45%, Sp 96.41%; effusion AUC 0.9900, Se 96.53%, Sp 95.11% | — | None |
| Gleamer ChestView US (K241620, 2025) [12] | **CADe 4 ROI categories** | 3,884 cases | Not stated in summary | Panel of US board-certified radiologists | AUC nodule 0.93, consolidation 0.952, pleural 0.973, mediastinum 0.922; two operating points (nodule Se 0.829/Sp 0.956 vs Se 0.482/Sp 0.994) | Not in summary | **Yes**: fully crossed MRMC, **240 cases**, radiologists + EM physicians, ≥1-month washout; AUC gains 0.034–0.158 (all p<0.001) |
| Annalise Enterprise (K250831, 2025) [13] | CADt 5 findings | 3,252 cases | 4 US sites; multiple manufacturers | ≥2 ABR + third | AUC PTX 0.984, tension 0.989, pneumoperitoneum 0.987, effusion 0.977, VCF 0.972 | "demographic stratification (gender, age, race/ethnicity)" | None (TAT 42.3 s, n=303) |
| Qure.ai qXR-Detect (K251934, 2026) [14] | **CADe 6 ROI categories, with PCCP** | Count not stated; US, 40 states, 5 regions, ~50/50 sex, 22–85+ y | Multi-state US | 3 ground-truthers | AUC 0.879–0.958 by category; wAFROC 0.771–0.89 | Sex/age balance stated | **Yes**: fully crossed concurrent MRMC, **301 cases, 18 readers** (radiologists, ER, family medicine), 28-day washout; wAFROC 0.6894 → 0.7505 (Δ 0.0611, p<0.001); FP/image 0.418 → 0.330 |

Peer-reviewed reader studies that vendors cite alongside submissions: Seah 2021 (Annalise) — 20 radiologists, 2,568 enriched cases, 127 findings, 3-month washout, AUC 0.713 → 0.808, model trained on 821,681 images [15]; Hwang 2019 (Lunit) — 1,015 external CXRs from 5 institutions, 15 physicians [16]; Ahn 2022 (Lunit) — 497 CXRs, 6 readers, 4-week washout [17].

## 3. Patterns worth copying, and the gaps

**Patterns:**
1. Triage submissions cluster at **600–1,000 cases with ~150–400 positives, from 4–5 sites**, with a 2+1 ABR-radiologist reference; the later ones (Annalise 2025) go to ~3,000 cases.
2. CADe submissions add a fully crossed MRMC of **240–300 cases with 6–18 readers** spanning the intended user types (radiologists plus ED/family-medicine physicians), a ≥28-day washout, and an AFROC/wAFROC or ROC-AUC figure of merit with bootstrap CIs.
3. Nodule ground truth is done with **paired CT** (qXR-LN: 5 truthers with CT; NODE21 test sets: CT within 60 days).
4. Every summary reports manufacturer diversity; the better ones (Qure, Annalise) report sex/age/race composition and, in Qure's case, per-subgroup AUCs with CIs.
5. Training-set scale is now 0.8–2.5M images from 3–5 countries; test sets are deliberately from **new sites**.

**Gaps nobody fills (our opening):**
- No summary reports **calibration** (ECE/reliability), **per-site** performance, **AP-vs-PA** or portable-vs-fixed performance with CIs — even though independent studies show view type dominates variance [18] and small findings collapse sensitivity [19].
- No public locked external test that customers can re-run; no drift monitoring commitments beyond Qure's PCCP.

## 4. Sizing our own validation study (derived from the above; not a regulatory opinion)

| Element | Target | Rationale |
|---|---|---|
| External standalone test | **≥ 3,000 frontal CXRs, consecutive, from ≥ 4 institutions in ≥ 2 countries**, with **≥ 300 positives per finding** (nodule/mass, effusion, pneumothorax) | Matches the 2025 Annalise scale (3,252) and exceeds all triage precedents; 300 positives gives roughly ±3–4 percentage points half-width on a sensitivity near 90% and lets us report ≥100 positives in at least two subgroups per axis (our `low-n` rule). |
| Subgroup axes with CIs | source/site, vendor (make/model), age band, sex, view (AP/PA), portable vs fixed, finding size (small vs large), number of co-existing findings, race/ethnicity where available | FDA 2022 guidance list + the two failure modes independent studies found (size, multiplicity, view). [1][18][19] |
| Reference standard | 2 board-certified radiologists + third adjudicator for effusion/PTX; **paired CT within 60 days** for nodule positives; consecutive normals with ≥1-year follow-up where possible | Qure qXR-LN and NODE21 precedent; Plesner's thoracic-radiologist panel with access to prior/future imaging. [10][20][19] |
| Calibration | ECE with bootstrap CI per site and per subgroup; recalibrate per site and re-test | Our principle 2; nobody reports it. |
| MRMC (if we go for CADe) | Fully crossed, **≥ 12 readers** (radiologists + ED physicians), **≥ 250 enriched cases**, ≥ 4-week washout, wAFROC primary, sensitivity/specificity at the deployed operating point secondary, read-first-then-reveal design mirrors the product | Gleamer (240/mixed readers), qXR-Detect (301/18), Ahn (497/6). Pre-register the SAP as FDA asks. |
| Data hygiene | Test sites never used in training or tuning; hash-locked split (D-008); firewall between developers and test results; no test-set reuse after retraining | FDA test-data-reuse and independence language. [1] |
| Candidate sources | ReXGradient private test (79 US sites), MIDRC sequestered commons, VinDr-CXR, BRAX, PadChest manual subset, PLCO 25k subset for nodules with outcomes | See `data_landscape.md` §3. |

## Sources
1. FDA, Clinical Performance Assessment: Considerations for CADe Devices Applied to Radiology Images ... 510(k) Submissions (2022-09-28). https://www.fda.gov/media/77642/download
2. FDA press announcement, draft guidance on AI-enabled device software functions (2025-01-07). https://www.fda.gov/news-events/press-announcements/fda-issues-comprehensive-draft-guidance-developers-artificial-intelligence-enabled-medical-devices
3. FDA 510(k) summary K191556 (Behold.ai red dot). https://www.accessdata.fda.gov/cdrh_docs/pdf19/K191556.pdf
4. FDA 510(k) summary K213941 (Annalise CXR Triage Pneumothorax). https://www.accessdata.fda.gov/cdrh_docs/pdf21/K213941.pdf
5. FDA 510(k) summary K214043 (Aidoc BriefCase PTX). https://www.accessdata.fda.gov/cdrh_docs/pdf21/K214043.pdf
6. FDA 510(k) summary K190362 (Zebra HealthPNX). https://www.accessdata.fda.gov/cdrh_docs/pdf19/K190362.pdf
7. FDA 510(k) summary K211733 (Lunit INSIGHT CXR Triage). https://www.accessdata.fda.gov/cdrh_docs/pdf21/K211733.pdf
8. FDA 510(k) summary K222268 (Annalise CXR Triage Trauma). https://www.accessdata.fda.gov/cdrh_docs/pdf22/K222268.pdf
9. FDA 510(k) summary K230899 (qXR-PTX-PE). https://www.accessdata.fda.gov/cdrh_docs/pdf23/K230899.pdf
10. FDA 510(k) summary K231805 (qXR-LN). https://www.accessdata.fda.gov/cdrh_docs/pdf23/K231805.pdf
11. FDA 510(k) summary K241439 (VUNO Med-Chest X-ray Triage). https://www.accessdata.fda.gov/cdrh_docs/pdf24/K241439.pdf
12. FDA 510(k) summary K241620 (Gleamer ChestView US). https://www.accessdata.fda.gov/cdrh_docs/pdf24/K241620.pdf
13. Innolitics FDA browser, K250831 (Annalise Enterprise). https://fda.innolitics.com/submissions/RA/subpart-b%E2%80%94diagnostic-devices/QFM/K250831
14. FDA 510(k) summary K251934 (qXR-Detect). https://www.accessdata.fda.gov/cdrh_docs/pdf25/K251934.pdf
15. Seah JCY et al. Lancet Digit Health 2021;3:e496. PMID 34219054. https://pubmed.ncbi.nlm.nih.gov/34219054/
16. Hwang EJ et al. JAMA Netw Open 2019;2:e191095. PMID 30901052. https://pubmed.ncbi.nlm.nih.gov/30901052/
17. Ahn JS et al. JAMA Netw Open 2022;5:e2229289. PMID 36044215. https://pubmed.ncbi.nlm.nih.gov/36044215/
18. Farquhar H. medRxiv 2026.01.20.26344495. https://www.medrxiv.org/content/10.64898/2026.01.20.26344495v1
19. Plesner LL et al. Radiology 2023;308:e231236. PMID 37750768. https://pubs.rsna.org/doi/10.1148/radiol.231236
20. NODE21 data page (CT-confirmed test reference). https://node21.grand-challenge.org/Data/
