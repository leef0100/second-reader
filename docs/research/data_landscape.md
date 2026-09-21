# Data landscape: public chest X-ray datasets and paths to ≥100k scans

> **For research use only. Not for diagnostic use.**
> Due-diligence research for the Second Reader project. Licence terms change; every entry must be re-verified and dated at download time (`secondreader datasets verify`) before it is added to `datasets/registry.yaml`. All URLs accessed 2026-09-21. "Verify" means we could not read the licence text itself.

## 1. Datasets already in `datasets/registry.yaml` (for reference only)
NIH ChestX-ray14, CheXpert, MIMIC-CXR, VinDr-CXR, PadChest, BRAX, CANDID-PTX, SIIM-ACR Pneumothorax, ChestX-Det, Kermany pediatric. Two facts from this research that bear on them:
- **PadChest** licence page: "Use of PadChest is free for all researchers", requires a research-use agreement, no sharing, no re-identification; commercial use not permitted. Confirms `commercial_ok: false`. [1]
- **MIMIC-CXR**: PhysioNet Credentialed Health Data License 1.5.0 — "for the sole purpose of lawful use in scientific research and no other"; no redistribution; code from publications must be shared. Confirms `commercial_ok: false`. [2]
- **NIH ChestX-ray14** was used as part of Lunit's FDA pivotal test set (K211733). Anyone comparing against Lunit on NIH is comparing on their test data, and our D-007 rule (no CXR-pretrained public weights) is the right instinct. [3]

## 2. Additional public datasets (verified this round)

| Dataset | Size | Labels and label method | Site / country | Licence — commercial use? | Access | Notes for us |
|---|---|---|---|---|---|---|
| **CheXpert Plus** (Stanford AIMI, 2024) | 223,462 report–image pairs; 187,711 studies; 64,725 patients; **DICOM** with 47 metadata elements; 8 patient-metadata elements | 14 CheXpert labels + full radiology reports (11 subsections) | Stanford, USA | Stanford University Dataset Research Use Agreement — **no commercial use**, no redistribution (per dataset card summary; licence text not directly readable) | Redivis, click-through | Adds DICOM headers (vendor, view) and demographics that plain CheXpert lacks — upgrades our vendor/age/sex subgroup axes for the same Stanford population. [4][5] |
| **MIMIC-CXR-JPG** (PhysioNet) | Same studies as MIMIC-CXR, JPG + structured labels | CheXpert + NegBio labels | BIDMC, USA | PhysioNet Credentialed 1.5.0 — **research only** | Credentialed (CITI training + DUA) | Convenience format of a dataset we already have. [6] |
| **ReXGradient-160K** (Rajpurkar Lab + Gradient Health, 2025) | 160,000 studies; 273,004 images; 109,487 patients; **79 medical sites across 3 US health systems**; splits 140k/10k/10k public test/10k private test | Free-text reports (indication/comparison/findings/impression); no structured labels | USA, multi-site | **Conflict:** the Hugging Face card states a "ReXGradient Non-Commercial Data Access and Use Agreement" — "No commercial use of the Data is permitted"; the arXiv paper says CC BY 4.0. Treat as **non-commercial until Gradient Health confirms in writing.** | Hugging Face, gated (contact info + agreement) | The single most valuable new set: 79 sites and a held-out private test on the ReXrank benchmark — ideal for site-level external testing. Labels must be NLP-mined by us. [7][8] |
| **PLCO chest radiographs** (NCI CDAS) | ~198,000 digitised screening CXRs (TIF) from ~58,000 participants; requests currently limited to a **standard 25,000-participant subset** (585 lung cancers, 10,152 abnormal screens, random normals; 12 batches, ~850 GB) | Screening result (abnormal-suspicious / abnormal-not-suspicious / negative), **confirmed lung-cancer diagnoses with stage and long follow-up** | Multi-centre US randomised screening trial (1993–2001 era films) | NCI Data Transfer and Use Agreement after project approval; commercial eligibility not stated on the page (ask) | Proposal via CDAS; download or shipped drive at requester's cost | The only public set with **cancer-outcome ground truth** for nodules at scale; older film-digitised images are a domain shift, which is itself a useful stress test. [9][10] |
| **MIDRC** (NIBIB/ARPA-H, RSNA+ACR+AAPM) | ">150,000 imaging studies for 67,728 patients" (search-engine summary; portal count not retrieved); includes a **sequestered, non-public commons** for validation | COVID-era CXR/CT with clinical data; annotations vary by sub-collection | Multi-institution USA | **Two DUAs: non-commercial and commercial**; no redistribution under either | Registration on data.midrc.org | Only public US commons with an explicit commercial DUA and a sequestered test set — a candidate for a neutral external test. [11][12][13] |
| **RSNA Pneumonia Detection Challenge** (2018) | ~26,684–30,000 frontal CXRs with pneumonia bounding boxes | Radiologist boxes for "lung opacity" | Derived from NIH ChestX-ray14 | RSNA terms: academic research and education and other non-commercial purposes (search summary; PDF summary inconsistent) — **verify** | Kaggle / RSNA | Not independent of NIH; use for localisation eval only. [14][15] |
| **Open-i / Indiana University CXR** (NLM) | 8,121 images, 3,996 reports | Free-text reports + MeSH | Indiana, USA | Reported as CC BY-NC-ND 4.0 on third-party mirrors; NLM site under maintenance when checked — **verify** | Open download (NLM) | Small; useful as an extra site for report-mined labels. [16][17] |
| **JSRT** (Japanese Society of Radiological Technology) | 247 CXRs (154 with a CT-confirmed nodule, 93 without, per the original paper) | Nodule location, size, malignancy | Japan | Site unreachable (certificate error) — **verify terms**; historically free for research | Registration on db.jsrt.or.jp | Classic nodule benchmark; too small for training, fine for a sanity check. [18] |
| **Montgomery County (138) and Shenzhen (662) TB sets** (NLM) | 138 (80 normal / 58 TB) and 662 (326 / 336) frontal CXRs | TB yes/no; lung masks available separately | Maryland USA; Shenzhen China | Public; attribution to NLM requested; no explicit licence text found — **verify** | Open download | Extra sites/vendors (CR); TB is out of our scope but they are useful OOD/quality negatives. [19] |
| **NODE21** (Grand Challenge, 2021) | 4,882 training CXRs: 1,134 with 1,476 annotated nodules + 3,748 nodule-free; test sets CT-confirmed (≤60 days) | Radiologist bounding boxes; test reference from CT | Composite of JSRT, PadChest, ChestX-ray14, Open-I | Built from sources that "allow us to remix and redistribute" — i.e., inherits PadChest's research-only terms; **not commercial-clean** | Zenodo | Nodule detection benchmark; not independent of NIH/PadChest. [20] |
| **BIMCV-COVID19+ / −** (Valencia) | 21,342 CR + 34,829 DX + 7,918 CT studies (COVID+ set) | UMLS-mapped findings from reports; 23 expert-segmented images | Valencian Community, Spain | Repository says MIT for code; data distribution rights in LICENSE.md — **verify** | B2DROP archives | Same region/vendors as PadChest, so a weak "second site". [21] |
| **VinDr-PCXR** | 9,125 paediatric studies (7,728 train / 1,397 test), <10 years | 36 findings + 15 diagnoses, boxes, 3 radiologists | Vietnam | PhysioNet **Restricted** Health Data License 1.5.0 — research only | PhysioNet DUA | Exactly the population our gate must reject; use as `ood_negative`. [22] |
| **REFLACX** | 3,032 readings of 2,616 MIMIC CXRs by 5 radiologists | Eye-tracking at 1,000 Hz, dictations, ellipses, certainty 0–5 | USA (MIMIC) | PhysioNet Restricted 1.5.0 | PhysioNet DUA | Ground truth for "where the reader looked" — useful for evaluating heat-map plausibility. [23] |
| **Chest ImaGenome** | 242,072 MIMIC frontal CXRs with automatic scene graphs; 500-patient (1,000 image) manually validated gold set | 29 anatomical locations × attributes; bounding boxes | USA (MIMIC) | PhysioNet Credentialed 1.5.0 | Credentialed | Anatomy boxes for weak localisation supervision. [24] |
| **CXR-LT** (MICCAI 2023/2024) | Expanded long-tailed labels on MIMIC-CXR-JPG | 26+ labels incl. rare ones | USA | PhysioNet Credentialed | Credentialed | Harmonised label set that already separates nodule from mass. [25] |
| **TBX11K** | 11,200 CXRs with TB bounding boxes | TB boxes | China | **CC BY 4.0** (commercial OK) | Open | One of the few commercial-clean sets; TB, not our findings. [26] |
| **Object-CXR** | 10,000 CXRs (5,000 with foreign objects) | Foreign-object boxes | China | "scientific research and non-commercial usage" | Academic Torrents / GitHub | Hardware/foreign-object confounder for false-positive analysis. [27] |
| **SIIM-FISABIO-RSNA COVID-19** | ~6,334 training CXRs (DICOM) | Opacity boxes, study-level classes | BIMCV + MIDRC | "academic and noncommercial use" | Kaggle / TCIA | Not independent of BIMCV/MIDRC. [28] |
| **UNIFESP body-part classifier** | 2,481 X-rays of 20+ body parts | Body part, multilabel | Brazil | Kaggle competition terms — **verify** | Kaggle | Training data for the "wrong body part" gate. [29] |
| Emory-CXR | 226,640 CXRs / 57,909 patients, 5 hospitals | — | USA | **Private** (used in Lancet Digit Health race-recognition paper); not available | — | Listed only so nobody counts it. [30] |

## 3. Realistic routes to ≥100,000 training scans and multi-site external tests

### 3.1 Research-only path (what the current registry already enables)
NIH ChestX-ray14 (112k) + CheXpert / CheXpert Plus (224k) + MIMIC-CXR (377k) + PadChest (161k) ≈ **870k images / ~230k patients** — but only NIH is commercial-clean; the rest are research-use only. Adding ReXGradient-160K (273k images, 79 sites) pushes training past 1.1M images and, more importantly, gives site-level held-out tests. External test candidates that are genuinely disjoint by institution and country: VinDr-CXR (Vietnam), BRAX (Brazil), PadChest (Spain), PLCO (US, 1990s film), MIDRC sequestered set (US multi-institution), ReXGradient private test (US, 79 sites).

### 3.2 Commercial-clean path (needed before any commercial retrain)
Public options with permissive licences are thin: NIH ChestX-ray14 (112k, "no use restrictions, attribution requested"), TBX11K (11k, CC BY 4.0), Kermany (5.9k, CC BY 4.0, paediatric — wrong population), plus MIDRC under its commercial DUA. Everything else requires either (a) a commercial licence from the data holder (Stanford AIMI explicitly offers to "License AIMI's AI-ready shared datasets for commercial use" [31]; ask BIMCV and Gradient Health the same question), or (b) purchased/licensed hospital data.

### 3.3 Data brokers (terms are not public; what we learned)
| Broker | Scale claimed | Business model | What to ask before signing |
|---|---|---|---|
| **Gradient Health** (Atlas 2 self-service platform; released ReXGradient-160K) | ">20 million imaging studies, with a further 30 million in progress or contracting"; DataFirst partnership adds imaging-centre data | Per-project quotes via a pricing calculator; no public price list; Protege partnership "unlocking seven-figures in net new revenue in under a year" for Gradient | Whether commercial model training and deployment rights are included; de-identification method (Safe Harbor vs Expert Determination); whether site identifiers are retained so we can hold out sites; redistribution/publication limits. [32][33][34] |
| **Segmed** (Insight platform) | Data "in hand in 2 weeks or less"; partners with Protege; offers hospitals a revenue-share to contribute de-identified data | Order-based; no public pricing | Same as above; plus whether the same studies are also sold to competitors (non-exclusivity affects test-set independence). [35][36] |
| **Protege** (marketplace; raised $30M Jan 2026) | Claims access to "over 3B clinical notes and 100M medical images" across partners | Platform fee to buyers + revenue share with data vendors | Provenance chain (which hospital, which IRB/waiver); whether the licence survives an acquisition. [37][38] |
| Others named in press: Avandra, Truveta (EHR, not imaging) | — | — | — [39] |

Pricing signal we could verify: none of the three publishes prices; every page routes to a sales call. The STAT investigation (Jan 2025) names these firms as paying hospitals for de-identified data but does not disclose rates (subscriber content). [39]

### 3.4 Hospital and academic collaboration routes
- **Stanford AIMI**: shared datasets free for non-commercial research; separate commercial licensing and an industry-affiliate programme (contact given on page). [31][40]
- **Mayo Clinic Platform_Validate / _Accelerate**: independent validation of third-party AI on Mayo's de-identified data "across different patient demographics ... to identify potential biases" (secondary description; platform homepage returned 403). Plausible route for a neutral external test with subgroup reporting. [41][42]
- **MIDRC**: commercial DUA plus a sequestered validation commons intended for "obtaining regulatory approval". [12][13]
- **NCI PLCO**: proposal-based; long-follow-up cancer outcomes. [9]
- UK NHS: the LungIMPACT RCT (93k CXRs) shows NHS trusts can run pragmatic AI trials; access is via research partnerships, not data sales. [43]

## 4. Regulatory and ethical constraints per avenue

| Avenue | Constraint | Practical consequence for us |
|---|---|---|
| PhysioNet sets (MIMIC-CXR, VinDr-CXR, BRAX, PCXR, REFLACX, ImaGenome) | Credentialing (CITI human-subjects training) + DUA per user; no redistribution; no re-identification; code release with publications; research only [2] | Each engineer credentials individually; never bake these images into a distributed artefact; a commercial retrain must exclude them (`commercial_ok: false`). |
| CheXpert / CheXpert Plus, PadChest | Research-use agreements, no commercial use, no sharing [1][4] | Same as above; commercial use needs a separate licence from Stanford / BIMCV. |
| NIH ChestX-ray14, TBX11K, Kermany | Permissive | Only commercial-clean training corpus today (~118k adult images). |
| PLCO | NCI DTUA after scientific review; images shipped at requester's cost | Timeline of months; outcome labels are the prize. |
| MIDRC | Non-commercial vs commercial DUA; no redistribution; sequestered set is not downloadable | Use the sequestered commons for a blinded external test rather than downloading. |
| Brokered hospital data | HIPAA de-identification by Safe Harbor (18 identifiers incl. full-face images, device serial numbers; DICOM metadata stripping; burned-in text) or Expert Determination; hospitals should ensure "properly de-identified" data for commercial uses [44][45] | Demand written de-identification method; our own two-Orthanc boundary and burned-in-text quarantine remain necessary because broker de-identification is not verifiable by us. |
| EU-origin data (PadChest, BIMCV) | GDPR; data subject to Spanish research-use terms | Keep EU data in research-only splits; expect re-consent/notified-body questions for CE. |
| FDA evidence rules (see `evidence_bar.md`) | Test data must be independent of training and from different sites; consecutive collection; test-data reuse discouraged; foreign data needs a poolability justification for US claims [46] | A dataset used for training anywhere in the pipeline can never be the external test; lock the test set (D-008) and log every access. |
| FDA Jan-2025 draft AI guidance | Recommends stratified subgroup analysis (age, sex, race/ethnicity) in performance validation [47] | Our subgroup-first evaluation is aligned with where FDA is heading; we need datasets that carry race/ethnicity (CheXpert Plus, MIMIC-CXR via MIMIC-IV, MIDRC) to report it. |

## Sources
1. BIMCV PadChest page. https://bimcv.cipf.es/bimcv-projects/padchest/
2. PhysioNet, MIMIC-CXR licence v2.1.0. https://physionet.org/content/mimic-cxr/view-license/2.1.0/
3. FDA 510(k) summary K211733 (Lunit). https://www.accessdata.fda.gov/cdrh_docs/pdf21/K211733.pdf
4. Stanford AIMI, CheXpert Plus. https://aimi.stanford.edu/datasets/chexpert-plus
5. Chambon P et al. CheXpert Plus (arXiv 2405.19538). https://arxiv.org/abs/2405.19538
6. PhysioNet, MIMIC-CXR-JPG v2.1.0. https://physionet.org/content/mimic-cxr-jpg/2.1.0/
7. Hugging Face, rajpurkarlab/ReXGradient-160K (dataset card and access agreement). https://huggingface.co/datasets/rajpurkarlab/ReXGradient-160K
8. ReXGradient-160K paper (arXiv 2505.00228). https://arxiv.org/abs/2505.00228
9. NCI CDAS, PLCO images. https://cdas.cancer.gov/learn/plco/images
10. NCI CDAS, access to PLCO data/images. https://cdas.cancer.gov/learn/plco/instructions/?type=images
11. IMAG wiki, MIDRC overview. https://www.imagwiki.nibib.nih.gov/resources/models-tools-databases/midrc-medical-imaging-and-data-resource-center
12. MIDRC Data Use Agreement. https://www.midrc.org/midrc-data-use-agreement
13. MIDRC homepage (sequestered commons statement). https://www.midrc.org/
14. RSNA Pneumonia Detection Challenge terms of use. https://www.rsna.org/-/media/Files/RSNA/Education/AI-resources-and-training/AI-image-challenge/pneumonia-detection-challenge-terms-of-use-and-attribution.ashx
15. Kaggle, RSNA Pneumonia Detection Challenge. https://www.kaggle.com/c/rsna-pneumonia-detection-challenge
16. Open-i FAQ (NLM; site under maintenance at access). https://openi.nlm.nih.gov/faq
17. Kaggle mirror, Indiana University chest X-rays. https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university
18. JSRT database (certificate error at access). http://db.jsrt.or.jp/eng.php
19. Jaeger S et al. Two public chest X-ray datasets. Quant Imaging Med Surg 2014. https://qims.amegroups.org/article/view/5132/6030
20. NODE21 data page. https://node21.grand-challenge.org/Data/
21. BIMCV-COVID-19 GitHub. https://github.com/BIMCV-CSUSP/BIMCV-COVID-19
22. PhysioNet, VinDr-PCXR. https://physionet.org/content/vindr-pcxr/1.0.0/
23. PhysioNet, REFLACX. https://physionet.org/content/reflacx-xray-localization/1.0.0/
24. PhysioNet, Chest ImaGenome. https://physionet.org/content/chest-imagenome/1.0.0/
25. PhysioNet, CXR-LT. https://physionet.org/content/cxr-lt-iccv-workshop-cvamd/2.0.0/
26. TBX11K (arXiv 2307.02848; CC BY 4.0). https://arxiv.org/abs/2307.02848
27. Object-CXR. https://github.com/hlk-1135/object-CXR
28. SIIM-FISABIO-RSNA COVID-19 (TCIA). https://wiki.cancerimagingarchive.net/display/Public/SIIM-FISABIO-RSNA+COVID-19+Detection
29. Kaggle, UNIFESP X-ray body part classifier. https://www.kaggle.com/competitions/unifesp-x-ray-body-part-classifier
30. Gichoya JW et al. AI recognition of patient race. Lancet Digit Health 2022 (Emory-CXR description). https://www.thelancet.com/journals/landig/article/PIIS2589-7500(22)00063-2/fulltext
31. Stanford AIMI industry page. https://aimi.stanford.edu/industry
32. Gradient Health homepage / calculator. https://gradienthealth.io/calculator/
33. Gradient Health, Atlas 2 launch. https://gradienthealth.io/blog/gradient-health-launches-atlas-2-setting-a-new-standard-for-medical-imaging-data-access
34. Protege, Gradient Health partnership. https://www.withprotege.ai/blog/gradient-health-and-protege-partner-to-bring-multi-modal-scale-and-data-diversity-to-healthcare-ai
35. Segmed Insight. https://www.segmed.ai/insight
36. Segmed, for healthcare providers. https://segmed.ai/solutions/for-healthcare-providers
37. SiliconANGLE, Protege raises $30M (2026-01-08). https://siliconangle.com/2026/01/08/protege-raises-30m-grow-governed-marketplace-ai-training-data/
38. Out-Of-Pocket, "Getting healthcare data to train an AI model – with Protege". https://www.outofpocket.health/p/getting-healthcare-data-to-train-an-ai-model---with-protege
39. STAT News, companies paying hospitals for de-identified data (2025-01-16). https://www.statnews.com/2025/01/16/health-ai-electronic-health-records-hipaa-deidentified-data-market/
40. Stanford AIMI shared datasets. https://aimi.stanford.edu/shared-datasets
41. Mayo Clinic Platform, "Finding the best AI algorithms: is FDA approval enough?" (2025-04-17). https://www.mayoclinicplatform.org/2025/04/17/finding-the-best-ai-algorithms-is-fda-approval-enough/
42. npj Health Systems 2026, Mayo Clinic Platform research applications. https://www.nature.com/articles/s44401-026-00068-1
43. Woznitza N et al. LungIMPACT. Nat Med 2026. https://pubmed.ncbi.nlm.nih.gov/41876649/
44. Accountable HQ, HIPAA de-identification (Safe Harbor / Expert Determination). https://www.accountablehq.com/post/guide-to-hipaa-de-identification-requirements-for-safe-harbor-and-expert-determination
45. Fierce Healthcare, hospitals fuelling AI innovation (de-identification for commercial use). https://www.fiercehealthcare.com/ai-and-machine-learning/industry-voices-hospitals-are-fueling-ai-innovation-should-they-own-piece
46. FDA guidance, Clinical Performance Assessment: CADe devices (2022-09-28). https://www.fda.gov/media/77642/download
47. FDA draft guidance, AI-Enabled Device Software Functions (2025-01-07). https://www.fda.gov/news-events/press-announcements/fda-issues-comprehensive-draft-guidance-developers-artificial-intelligence-enabled-medical-devices
