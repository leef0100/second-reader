# Market: radiology AI and chest X-ray CADe

> **For research use only. Not for diagnostic use.**
> Due-diligence research for the Second Reader project. All figures are quoted from the cited source; every URL was accessed on 2026-09-21 unless stated. Where we could only reach a secondary summary (search-engine snippet, trade press) rather than the primary page, this is flagged as *(secondary)*. Nothing here is a forecast of our own.

## 1. Market size and growth

The published estimates disagree by more than an order of magnitude because they count different things (radiology-only vs "AI in medical imaging", software-only vs software + services + hardware). Treat them as ranges, and always say which report you are quoting.

| Source (date) | Base → forecast | CAGR | What is counted / methodology |
|---|---|---|---|
| **Signify Research** (2023-07-21) | $576M (2022) → $1.73B (2027) | 25% | "AI-based medical image analysis" software; cloud ≈ $1.1B of 2027; four segments = 88% of 2022 revenue: cardiology $192M, neurology $159M, **pulmonology $88M**, breast $69M. No separate chest X-ray line. Signify is the only independent analyst house specialising in imaging; earlier edition (2022-08-16) had >$400M (2021) → ~$1.4B (2026), 26.7% CAGR, with 73% of 2021 revenue from stand-alone AI vendors vs 27% bundled into advanced-visualisation platforms. [1][2] |
| **MarketsandMarkets** (Nov 2025) | $0.76B (2025) → $2.27B (2030) | 24.5% | Base year 2024. Scope: "AI-driven solutions ... to support medical imaging acquisition, enhancement, analysis, interpretation, and reporting". Top-down + bottom-up; ~50 primary interviews. CT is the largest modality segment; X-ray is a named segment but no figure is public. [3] |
| **Fortune Business Insights** (2026) | $518.1M (2025) → $600.8M (2026) → $3,227.1M (2034) | 23.38% | Base year 2025; software + services; CT 32.0% share of 2026; X-ray one of six modalities, no isolated figure. [4] |
| Precedence Research *(secondary)* | $794.12M (2025) → $7,168.28M (2035) | 24.61% | Methodology page not retrieved. [5] |
| Grand View Research *(secondary; page returned HTTP 403)* | $14.6B (2025) → $193.0B (2033) | 38.2% | Almost certainly a broader "AI in medical imaging incl. hardware/services" scope; do **not** quote next to the figures above. [6] |
| Dataintelo "AI in chest X-ray analysis" *(secondary, low credibility)* | $1.8B (2025) → $8.4B (2034) | 18.6% | No methodology visible; inconsistent with Signify's $88M pulmonology segment. Not recommended for a pitch. [7] |

**Recommended framing for a pitch:** "Independent analyst Signify Research sizes AI-based medical image analysis at $576M (2022) growing ~25%/yr to $1.73B by 2027; pulmonology (chest X-ray and chest CT) was $88M in 2022." [1] Signify's February 2026 review says the sector is "entering a more decisive phase; one defined less by experimentation and more by evidence, execution, and accountability" and that "post-market requirements and regulatory expectations are reshaping which vendors can operate, scale, and sustain commercial success". [8]

**Regulatory volume as a proxy for market activity.** FDA had authorised 1,451 AI-enabled devices through end-2025; 331 were authorised in 2025 alone (the most in any year); 1,104 (76%) are radiology devices. [9]

## 2. Reimbursement status (US)

Short version: **there is no CPT code or add-on payment for chest X-ray AI.** Reimbursement exists for a handful of AI services (stroke triage, coronary plaque, FFR-CT), and CMS is only now (July 2026) building a generic payment vehicle for software.

### 2.1 NTAP (inpatient add-on payment)
- Viz.ai's Viz LVO (formerly ContaCT) was the first AI software to receive a CMS New Technology Add-on Payment (granted for FY2021, renewed for FY2022), "up to $1,040 per eligible patient". [10] NTAP is explicitly temporary (CMS grants it for a limited number of years while claims data accumulate). [11]
- Newer imaging-AI NTAPs: Aidoc's CARE Multi-Triage CT Body becomes NTAP-eligible for inpatient Medicare cases from 2026-10-01 (FY2027), "with reimbursement eligibility lasting for three years"; InVision's cardiac-amyloid echo AI also received an FY2027 NTAP. [12][13]
- None of these are chest radiograph products. NTAP applies only to inpatient (IPPS) cases, which is a poor fit for the ED/outpatient chest X-ray volume.

### 2.2 CPT codes
- **Category I codes for AI as of 2026-01-01:** 75577 (AI quantification/characterisation of coronary plaque on CCTA, converted from Category III 0623T–0626T) and 75580 (FFR-CT, converted 2024). [14][15][16] Radiology Business summarised in early 2026 that "there will only be two CPT category 1 payment codes for newer AI, despite there being hundreds of FDA-cleared medical imaging algorithms" and that "tools for fracture detection, incidental findings or lung nodule identification are unlikely to receive their own paid CPT codes because radiologists are already reimbursed for interpreting imaging studies and are already supposed to be identifying those findings" *(secondary; article page returned 403)*. [17]
- Category III (tracking, no guaranteed payment) codes exist for e.g. quantitative ultrasound tissue characterisation (0689T/0690T) and several cardiology AI services; new 2026 Category III AI codes include 0992T (perivascular fat analysis) and 0993T–0995T (algorithmic ECG). [18][14]
- **Caution:** at least one billing-company blog claims that "AI-assisted imaging tasks, such as detecting lung nodules ... have transitioned from Category III to Category I codes" in 2026. [19] We could not find an AMA or CMS source for this and it contradicts the Radiology Business summary above; do not repeat it in a pitch without verifying against the 2026 CPT code set.
- AMA CPT **Appendix S** classifies AI services as *assistive / augmentative / autonomous*; the CPT Editorial Panel revised it at its May 2026 meeting. A second-reader CADe would be "assistive" (the lowest tier, machine detects, human interprets). [20]

### 2.3 CMS "Software as a Medical Service" (SaMS) — the emerging path
On 2026-07-02 CMS released the CY2027 OPPS proposed rule, renaming "Software as a Service" codes to **Software as a Medical Service (SaMS)**, designating 36 HCPCS codes as SaMS, creating a new status indicator **O1** for separately payable SaMS, and moving 21 SaMS codes into New Technology APCs. CMS said it had been evaluating "a comprehensive and consistent approach to SaMS payment for several years". Final rule expected late 2026. [21][22][23] This is the mechanism a future CXR CADe code would most plausibly use, but it presupposes a code exists.

### 2.4 Outside the US
- UK: Annalise's NHS list price example (from the Scottish Health Technologies Group review) was an annual subscription of £51,250 for up to 25,000 CXRs/yr plus a one-off implementation fee; NICE's early value assessment of CXR AI for suspected lung cancer found the cost-effectiveness evidence immature. [24][25]
- Behold.ai (FDA-cleared pneumothorax triage) entered administration in January 2025 "after failing to secure contracts from the NHS's AI Diagnostic Fund", a concrete example that clearance without procurement is fatal. [26]

## 3. Radiologist workload and shortage

| Statistic | Source |
|---|---|
| HRSA (Dec 2025) projects a US shortage of 141,160 FTE physicians by 2028 (89% adequacy) and 158,990 by 2038; radiology is estimated at "90% adequacy in 2038". | ACR HPI workforce update, 2026 [27] |
| Radiologist attrition more than doubled 2014→2022, from 1.1% to 2.5%; subspecialists 37% more likely to leave; female radiologists 26% higher attrition. | ACR HPI / Neiman HPI (AJR 2025) [27] |
| Neiman HPI (2025-02-12, two JACR papers): radiologist supply +25.7% 2023→2055 with no residency growth; imaging demand +16.9% to +26.9% by 2055 depending on modality; population 75–84 +51.5%, 85–94 +149.6%. "The present radiologist shortage is projected to persist unless steps are taken." | [28] |
| Residency positions +33% (2010–2025) but practising radiologists only +12%; radiologists per 100,000 population flat at 11.1–11.5. | JACR 2026 via AuntMinnie *(secondary)* [29] |
| ACR job board >1,800 open radiologist positions; imaging volume growth 3–4%/yr vs ~2%/yr residency growth. | Becker's / Medicus *(secondary)* [30][31] |
| UK (RCR 2024 census): consultant radiologist shortfall 29% (1,953 consultants), rising to 39% by 2029; workforce +4.2% in 2024 vs CT/MRI volume +8%; 100% of clinical directors concerned that backlogs are worsening; >434,000 CT/MRI scans unreported at 4 weeks. | RCR [32] |

## 4. Error rates in chest radiograph reading

### 4.1 Overall and by error type
- Day-to-day radiology error/discrepancy rate is estimated at 3–5% of studies reported, with much higher rates in targeted studies (Brady, *Insights Imaging* 2017;8:171–182). [33]
- Bruno et al. (*RadioGraphics* 2015) distinguish **perceptual errors** ("an important abnormality is simply not seen") from **cognitive errors** (seen but misinterpreted) and state that overall prevalence "does not appear to have changed since it was first estimated in the 1960s". [34] The commonly quoted split (perceptual errors ≈ 60–80% of interpretive errors) is attributed to this paper in secondary sources; we could not verify the figure in the abstract, so cite it as "commonly cited" rather than as a hard number. [34][35]
- Waite et al. (*AJR* 2017;208:739–749): "interpretive error rates remain unchanged" since Garland 1949; errors occur in an environment of "numerous extrinsic distractors, increasing workloads and fatigue". [36]
- Kim & Mansfield (*AJR* 2014;202:465–470) reviewed 656 delayed-diagnosis examinations with 1,269 errors and emphasised *perpetuated* errors, i.e. a miss on a prior is often repeated on the follow-up film. This is the strongest literature argument for our prior-comparison feature. [37]

### 4.2 Missed lung cancer / nodules on CXR (peer-reviewed)
- Quekel et al. (*Chest* 1999;115:720–724), 396 NSCLC patients: **19% (49/259)** of cancers presenting as a nodule were missed; median diameter of missed lesions 16 mm vs 40 mm for detected; median diagnostic delay 472 days vs 29; tumour stage moved T1→T2 in 43% of missed cases; superimposing structures present in 71% of missed vs 2% of detected. [38]
- Kaviani et al. (*Diagnostics* 2022;12:2382), 2,407 CXRs from 8 sites (3 India, 5 US): missed findings in **14.2%** of CXRs; of clinically important missed findings, **pulmonary nodules were 43.1%** (n=177), consolidation 15.1%, mediastinal widening 5.8%, pleural effusion 2.7% (n=11), pneumothorax n=4. [39]
- Ahn et al. (*JAMA Netw Open* 2022;5:e2229289), 6 readers, 497 enriched CXRs (MIMIC-CXR + MGH): unaided reader sensitivity for nodule 0.567, pneumothorax 0.792, pleural effusion 0.889; AI sensitivity 0.816 / 0.988 / 0.872. [40]
- Retrospective reviews report that most lung cancers are visible in hindsight on earlier films reported as normal (70–90% quoted in a review summary; the primary review, del Ciello et al. *Diagn Interv Radiol* 2017, could not be fetched) *(secondary)*. [41][42] The UK Health Services Safety Investigations Body has a formal investigation into missed lung cancer on primary-care CXRs. [43]

### 4.3 Pneumothorax and effusion
- Plesner et al. (*Radiology* 2023;307:e222268), 1,529 consecutive PA CXRs, four Danish hospitals: sensitivity of the **clinical radiology report** was 72.3% for any abnormality and 93.5% for critical abnormalities against a three-thoracic-radiologist reference; the AI (a normal-vs-abnormal tool) had 99.1% / 99.8%. [44]
- Kaviani 2022 (above) found pneumothorax and effusion misses to be rare in absolute terms (4 and 11 of 2,407) but they are the findings with the most immediate harm. [39]
- A single-centre study reported 92.5% resident/attending concordance for pneumothorax on ICU/ED films *(secondary)*. [45]

### 4.4 What this means for a "second reader" positioning
The literature supports a **nodule-first** value proposition (largest share of clinically important misses; large stage-shift consequence) and a **prior-comparison** feature (perpetuated errors). It does **not** support claiming that AI triage speeds up cancer diagnosis: the LungIMPACT RCT (*Nat Med* 2026, 93,326 CXRs, multicentre NHS) found no difference in time to CT (ratio 0.97, p=0.31) or time to lung-cancer diagnosis (ratio 0.98, p=0.84) with AI prioritisation on vs off. [46]

## Sources
1. Signify Research, "Medical Imaging AI Market Forecasted to Surpass $1.7 Billion by 2027" (2023-07-21). https://www.signifyresearch.net/insights/medical-imaging-ai-market-forecasted-to-surpass-1-7-billion-by-2027/
2. Signify Research, "Medical Imaging AI Market Set to Reach Almost $1.4Bn by 2026" (2022-08-16). https://www.signifyresearch.net/insights/medical-imaging-ai-market-set-to-reach-almost-1-4bn-by-2026/
3. MarketsandMarkets, Radiology AI Market report (Nov 2025). https://www.marketsandmarkets.com/Market-Reports/radiology-ai-market-231900847.html
4. Fortune Business Insights, AI in Radiology Market (2026–2034). https://www.fortunebusinessinsights.com/ai-in-radiology-market-115732
5. Precedence Research, Radiology AI Market. https://www.precedenceresearch.com/radiology-ai-market
6. Grand View Research, AI in Radiology Market (page returned 403; figures from search snippet). https://www.grandviewresearch.com/industry-analysis/artificial-intelligence-ai-radiology-market-report
7. Dataintelo, AI in Chest X-ray Analysis Market. https://dataintelo.com/report/ai-in-chest-x-ray-analysis-market
8. Signify Research, "What the data tells us: the 2025 medical imaging AI market review" (2026-02-09). https://www.signifyresearch.net/insights/what-the-data-tells-us-the-2025-medical-imaging-ai-market-review/
9. The Imaging Wire, "FDA updates AI list with new clearances" (2026-03-11). https://theimagingwire.com/2026/03/11/numbers-from-the-fda-show-radiology-is-maintaining-its-lead/
10. Viz.ai, NTAP renewal press release (2021-08-04). https://www.viz.ai/news/viz-ai-receives-new-technology-add-on-payment-renewal-for-stroke-ai-software-from-cms
11. CMS, New Medical Services and New Technologies (NTAP). https://www.cms.gov/medicare/payment/prospective-payment-systems/acute-inpatient-pps/new-medical-services-and-new-technologies
12. Aidoc, CARE Body CT Multi-Triage NTAP eligibility (PR Newswire, 2026). https://www.prnewswire.com/news-releases/aidocs-care-body-ct-multi-triage-receives-eligibility-for-medicare-new-technology-add-on-payment-302850312.html
13. InVision Medical Technology, NTAP for cardiac amyloid AI. https://invisionmedtech.com/press/ntap-cardiac-amyloid/
14. ACC, "Coding Corner: Overview of New CPT Codes for 2026" (2025-12-04). https://www.acc.org/latest-in-cardiology/articles/2025/12/04/18/57/coding-corner-overview-of-new-cpt-codes-for-2026
15. Cardiovascular Business, "New Category I CPT code issued for AI-enabled coronary plaque analysis software". https://cardiovascularbusiness.com/topics/cardiac-imaging/computed-tomography-ct/new-category-i-cpt-code-issued-ai-enabled-coronary-plaque-analysis-software
16. Ventra Health, "AI in Radiology – Reimbursement Landscape 2025". https://ventrahealth.com/blog/ai-in-radiology-reimbursement-landscape-2025/
17. Radiology Business, "Radiology dominates FDA-cleared AI, but reimbursement lags far behind" (page 403; quoted from search snippet). https://radiologybusiness.com/topics/artificial-intelligence/radiology-dominates-fda-cleared-ai-reimbursement-lags-far-behind
18. MedCentral, "CPT 2026 Updates Expand Lab, Category III, and AI Codes". https://www.medcentral.com/coding-reimbursement/cpt-2026-updates-expand-lab-category-iii-and-ai-codes
19. 24/7 Medical Billing Services blog (unverified claim). https://www.247medicalbillingservices.com/blog/radiology-billing-reimbursement-2026-complete-guide-to-ai-assisted-diagnostic-coding
20. AMA, "CPT Editorial Panel strengthens AI taxonomy" (Appendix S). https://www.ama-assn.org/practice-management/cpt/cpt-editorial-panel-strengthens-ai-taxonomy-keep-pace-tech
21. DistilINFO, "CMS builds new payment path for AI 2026" (2026-07-20). https://distilinfo.com/2026/07/20/software-medical-service-cms-ai-payment/
22. Wilson Sonsini, "CMS Proposes Payment Frameworks for 'Software as a Medical Service'". https://www.wsgr.com/en/insights/cms-proposes-payment-frameworks-for-software-as-a-medical-service.html
23. Holland & Knight, CY2027 OPPS proposed rule summary (July 2026). https://www.hklaw.com/en/insights/publications/2026/07/cms-issues-sweeping-cy-2027-hospital-opps
24. Scottish Health Technologies Group, AI-supported clinician review of chest X-ray (2024). https://shtg.scot/media/2442/20240312-chest-x-ray-ai-imto-v10.pdf
25. NICE early value assessment, AI software for CXR lung cancer (NCBI Bookshelf). https://www.ncbi.nlm.nih.gov/books/NBK606983/
26. Digital Health, "Digital health startup Behold.ai enters administration" (2025-01). https://www.digitalhealth.net/2025/01/digital-health-startup-behold-ai-enters-administration/
27. ACR Bulletin, "The Radiologist Shortage: A Workforce Update from HPI" (2026). https://www.acr.org/Clinical-Resources/Publications-and-Research/ACR-Bulletin/2026/radiologist-shortage-work-force-update
28. Neiman HPI press release (2025-02-12). https://www.neimanhpi.org/press-releases/new-studies-shed-light-on-the-future-radiologist-workforce-shortage-by-projecting-future-radiologist-supply-and-demand-for-imaging/
29. AuntMinnie, "Radiologist pipeline trails increase in training positions" (JACR 2026). https://www.auntminnie.com/practice-management/article/15834030/radiologist-pipeline-trails-increase-in-training-positions
30. Becker's, "The radiologist shortage, explained". https://www.beckershospitalreview.com/radiology/the-radiologist-shortage-explained/
31. Medicus Healthcare Solutions, radiologist shortage. https://medicushcs.com/resources/the-radiologist-shortage-addressing-the-gap-between-supply-and-demand
32. RCR, 2024 clinical radiology workforce census. https://www.rcr.ac.uk/news-policy/latest-updates/2024-workforce-census-reports-lay-bare-the-challenges-facing-radiology-and-clinical-oncology/
33. Brady AP. Insights Imaging 2017;8:171–182. https://link.springer.com/article/10.1007/s13244-016-0534-1
34. Bruno MA et al. RadioGraphics 2015;35:1668–1676. PMID 26466178. https://pubs.rsna.org/doi/10.1148/rg.2015150023
35. RSNA News, "Human factors drive radiology error rates" (2022). https://www.rsna.org/news/2022/march/human-error-in-radiology
36. Waite S et al. AJR 2017;208:739–749. PMID 28026210. https://pubmed.ncbi.nlm.nih.gov/28026210/
37. Kim YW, Mansfield LT. AJR 2014;202:465–470. PMID 24555582. https://pubmed.ncbi.nlm.nih.gov/24555582/
38. Quekel LG et al. Chest 1999;115:720–724. PMID 10084482. https://pubmed.ncbi.nlm.nih.gov/10084482/
39. Kaviani P et al. Diagnostics 2022;12:2382. https://pmc.ncbi.nlm.nih.gov/articles/PMC9600490/
40. Ahn JS et al. JAMA Netw Open 2022;5:e2229289. PMID 36044215. https://pubmed.ncbi.nlm.nih.gov/36044215/
41. AuntMinnieEurope, "Why errors in radiology are no easy matter". https://www.auntminnieeurope.com/clinical-news/article/15651837/why-errors-in-radiology-are-no-easy-matter
42. del Ciello A et al. "Missed lung cancer: when, where, and why?" Diagn Interv Radiol 2017. https://europepmc.org/article/PMC/PMC5338577
43. HSSIB investigation report, missed lung cancer on primary-care CXRs. https://www.hssib.org.uk/patient-safety-investigations/missed-detection-of-lung-cancer-on-chest-x-rays-of-patients-being-seen-in-primary-care/investigation-report/
44. Plesner LL et al. Radiology 2023;307:e222268. PMID 36880947. https://pubmed.ncbi.nlm.nih.gov/36880947/
45. Concordance of resident and attending pneumothorax reporting (PMC9616555). https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9616555/
46. Woznitza N et al. LungIMPACT RCT. Nat Med 2026;32:1737–1744. PMID 41876649. https://pubmed.ncbi.nlm.nih.gov/41876649/
