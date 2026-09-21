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
- Localisation is Grad-CAM: coarse, sometimes off-target, and not a segmentation.
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
| Burned-in text | Quarantined at ingestion (header + pixel heuristic); heuristic is not OCR and can miss | Manual quarantine review |
| Model/threshold change | New `model_version_id`; every prediction records it | Audit |

## 8. Safety and human factors

- **Read-first mode** is the default: AI output is withheld until the clinician records their own impression (server-enforced, D-021).
- Every finding is displayed with its probability, threshold and heat-map; a finding is never shown without its confidence.
- All views, reveals, downloads and feedback are audit-logged with user and time.
- Clinician agree/disagree is captured per finding and surfaced in the drift report.
