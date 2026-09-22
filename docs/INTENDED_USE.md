# Intended use

> **For research use only. Not for diagnostic use.**
> This document describes a research prototype. It is not a cleared or approved medical device and must not be used to make or support clinical decisions for real patients.

## 1. Intended use statement (as it would be written for a regulated product)

Second Reader is software intended to be used by qualified clinicians as a **concurrent or second read** aid for **adult frontal (PA/AP) chest radiographs**. For each eligible image it produces, for three findings — **pneumothorax, pleural effusion, and pulmonary nodule/mass** — a calibrated probability, a localisation (heat-map and bounding box), and a binary flag at a stated operating point. It also states when an image **cannot be assessed** (wrong view, wrong body part, poor quality, or out-of-distribution input).

The software **does not** make diagnoses, triage, prioritise, or alter the standard of care. The clinician's own interpretation is recorded before AI output is shown; the AI output is one input among others.

## 2. Intended users and environment

- Radiologists, radiology residents, and other clinicians credentialed to interpret chest radiographs.
- Reading-room or bench environment with a PACS (tested with Orthanc) and a DICOM viewer (tested with OHIF).
- Not for patients, not for self-referral, not for unattended/automated workflows.

## 3. Input specification

| Property | Requirement |
|---|---|
| Modality | DICOM CR or DX (CR/DX Image Storage SOP classes) |
| View | Frontal PA or AP (`ViewPosition` and learned view head) |
| Body part | Chest/thorax |
| Frames | Single frame |
| Patients | Adults (≥ 18 y). Paediatric images are out of scope and should be gated out by the OOD detector — this is **not** yet validated. |
| Pixel data | Any bit depth; MONOCHROME1/2; rescale slope/intercept honoured |
| Identity | Inputs are de-identified before storage; the pseudonymised patient key links priors |

Anything else is rejected with `unable_to_assess` and a stated reason. Secondary-capture objects and images flagged `BurnedInAnnotation=YES` are quarantined at ingestion and never reach the model.

## 4. Outputs

- **JSON** (`POST /v1/infer`, review API): per-finding calibrated probability, threshold, `present` flag, boxes, model version, input hash, gate result.
- **DICOM Structured Report** (TID 1500): finding type (SNOMED CT), probability measurement, Presence, image region, algorithm id = model version. `VerificationFlag` is always `UNVERIFIED`.
- **DICOM Secondary Capture** overlays: one per finding with heat-map, box, label and the research-use banner burned in.
- Every output carries `For research use only. Not for diagnostic use.`

## 5. Performance

**No claim of performance is made.** The only trained model in this repository (`models/ACTIVE_MODEL_ID`) was trained and evaluated on a **synthetic** dataset to prove the pipeline (see `docs/reports/smoke_external_report.md`). Its numbers say nothing about real radiographs.

When a model is trained on real data, `secondreader cxr eval` produces the report that would populate this section: AUROC, sensitivity, specificity, ECE with bootstrap CIs, **overall and per subgroup** (data source, vendor, age band, sex, view) on the locked external test set (`DECISIONS.md` D-008). Subgroups with < 100 positives are flagged `low-n`.

### Subgroup performance gaps (to be filled from the external report)

| Subgroup axis | Known/expected gap | Mitigation |
|---|---|---|
| Data source / site | Performance typically drops at new sites (scanner, protocol, population) | External test on ≥ 2 sites; OOD gate; drift monitor |
| Vendor | Only PadChest exposes vendor; other sets report `unavailable` | Report honestly as unavailable; add vendor from DICOM headers on DICOM datasets |
| Age band | Elderly patients have more comorbid findings; paediatric out of scope | Age-banded reporting; OOD gate for paediatric |
| Sex | Breast tissue / body habitus changes lung-base appearance; effusion and nodule detection may differ | Sex-stratified reporting |
| View (AP vs PA) | Portable AP films are lower quality and sicker patients; pneumothorax appearance differs supine | View-stratified reporting; AP/PA passed to the model as metadata in a future version |

## 6. Limitations

- Three findings only. Absence of a flag is **not** a normal read.
- Nodule/mass is a merged target (D-012); the model does not size or characterise lesions.
- **The heat-map/box is a coarse region of model attention, not a localisation** (D-030). Measured against 415 NIH radiologist boxes for our three findings (`docs/reports/nih_baseline_v1_platt/localization_report.md`, model `cc2f56aa3325788c`): the heat-map peak lies inside the radiologist's box in 47 % [42–52] of true findings (random 5 %, centre prior 3 %); mean box IoU 0.20 (random 0.04); only 30 % of boxes reach IoU ≥ 0.3 and 9 % reach IoU ≥ 0.5; 19 % of heat-map mass falls inside the radiologist's box. Per finding, peak-in-box / IoU: pneumothorax 32 % / 0.17, pleural effusion 56 % / 0.25, nodule/mass 48 % / 0.18. Most of these images were in the model's training tier; the held-out subset (77 rows) is consistent but low-n. Grad-CAM is 10×10 at 320 px, so a small nodule cannot be tightly boxed by construction.
- Calibration is fitted on internal validation; it is not guaranteed on a new site (monitor ECE via feedback and re-calibrate per site).
- Prior comparison is side-by-side display only; the model does not compare with priors.
- Training labels from NLP-mined reports are noisy; the label harmonisation across datasets is approximate.
- Latency, throughput and hardware requirements are not characterised.

## 7. Known failure modes

| Failure mode | Behaviour | Detection |
|---|---|---|
| Lateral or non-chest image | Gated: `unable_to_assess` | header + view head |
| Cine / multi-frame object | Gated | frame count |
| Very small, blank, saturated or extreme-aspect image | Gated | pixel checks |
| Scanner/protocol shift | Elevated OOD distance → gated; if below threshold, silently degraded accuracy | OOD gate; drift monitor (PSI on inputs and outputs, gate-out rate, disagreement rate) |
| Chest tubes, lines, pacemakers, surgical clips | May trigger false positives (esp. pneumothorax, nodule) | Clinician review; feedback capture |
| Apical pneumothorax on supine AP | Reduced sensitivity | View-stratified evaluation |
| Subpulmonic effusion | Reduced sensitivity | — |
| Nipple shadows, bone islands, vessels end-on | Nodule false positives | Clinician review |
| Heat-map/box off the finding on a true positive | In ≈ 53 % of true findings the heat-map peak is outside the radiologist's box; ≈ 91 % of boxes have IoU < 0.5 (D-030). The box can also sit on a second plausible focus (e.g. the other costophrenic angle) that the reference reader did not mark. The flag may be right while the box is wrong — never treat the box as a negative read of the rest of the film | `cxr eval-localization` per model version; clinician review; feedback capture |
| Burned-in text | Quarantined at ingestion (header + pixel heuristic); heuristic is not OCR and can miss | Manual quarantine review |
| Model/threshold change | New `model_version_id`; every prediction records it | Audit |

## 8. Safety and human factors

- **Read-first mode** is the default: AI output is withheld until the clinician records their own impression (server-enforced, D-021).
- Every finding is displayed with its probability, threshold and heat-map; a finding is never shown without its confidence.
- All views, reveals, downloads and feedback are audit-logged with user and time.
- Clinician agree/disagree is captured per finding and surfaced in the drift report.
