# Regulatory orientation notes

> **General orientation only — not legal or regulatory advice.** Regulatory strategy must be set with qualified regulatory-affairs counsel. Requirements and guidance change; verify against current FDA, EU, and notified-body publications.
>
> **This prototype is for research use only and is not for diagnostic use.**

## 1. What this software would be, regulatorily

A software-only device (SaMD) that analyses chest radiographs and flags findings to a clinician is a **computer-assisted detection (CADe)** device. Key precedents:

- **FDA:** CADe for chest radiographs is class II. Product codes used for comparable cleared devices include **QFM** ("radiological computer aided triage and notification software", for triage/notification claims) and **MYN / QIH-type** codes for CADe/CADx lesion detection. Most comparable AI chest-X-ray devices have gone through **510(k)** using an already-cleared predicate; **De Novo** applies when no predicate exists for the claimed intended use (e.g. a novel finding or workflow claim).
- **EU MDR (2017/745):** software intended to provide information used to take diagnostic decisions is at least **class IIa** (Rule 11); IIb if such decisions may cause serious deterioration or surgery. Requires a notified body, a Quality Management System (ISO 13485), technical documentation per Annex II/III, clinical evaluation (MDCG 2020-1 for software), post-market surveillance, and UKCA/CE marking considerations.

## 2. What a submission would need (either jurisdiction)

| Element | What it means here | Status in this prototype |
|---|---|---|
| **Intended use / indications for use** | Precise statement: adult frontal CXR, three findings, second-reader aid, clinician users, not for triage | Drafted in `docs/INTENDED_USE.md` — needs clinical review |
| **Quality management system** | ISO 13485 (EU: mandatory; FDA: 21 CFR 820 / QMSR) | **None.** Repository conventions only |
| **Software lifecycle (IEC 62304)** | Safety classification (likely class B/C), documented requirements, architecture, detailed design, verification, SOUP management, problem resolution | Partial: architecture docs, versioned configs, tests, traceability of model/data/config hashes; no formal requirements/traceability matrix, no SOUP list, no hazard classification |
| **Risk management (ISO 14971)** | Hazard analysis (missed findings, false positives, automation bias, PHI leakage, OOD inputs), risk controls, residual risk acceptance | Informal: failure modes listed in INTENDED_USE; anti-automation-bias and gating are risk controls, but there is no risk file |
| **Usability engineering (IEC 62366-1)** | Formative/summative studies with representative clinicians on the actual UI | None |
| **Cybersecurity** | FDA premarket cybersecurity guidance (SBOM, threat model, patching), EU MDCG 2019-16 | Partial: encryption at rest/in transit design, auth, audit; no SBOM, no threat model, secrets handling is dev-grade |
| **Data management** | Documented provenance and licensing of every training/tuning/test dataset; representativeness; de-identification validation | `datasets/registry.yaml` with licenses; **all radiologist-labelled sets are research-only licences — a commercial product would need different data agreements** |
| **Analytical / standalone performance** | Sensitivity, specificity, AUROC with CIs on an **independent** external test set, stratified by site, scanner, age, sex, view; calibration; failure analysis; reader-study design for CADe claims (MRMC) | Pipeline implemented (`secondreader cxr eval`); **no real-data results**; no MRMC study |
| **Clinical evaluation** | FDA: usually a retrospective standalone study plus, for CADe, an MRMC reader study showing clinician improvement. EU: clinical evaluation report with literature + own data; PMCF plan | None |
| **Labelling** | User manual, warnings, limitations, performance tables, version identification | Draft docs only |
| **Change control / PCCP** | FDA Predetermined Change Control Plan (2024 guidance) for model updates; EU significant-change assessment | Model versioning and content hashes support this; no PCCP |
| **Post-market surveillance** | Complaint handling, performance monitoring, vigilance/MDR reporting | Drift monitor + feedback capture are a technical basis; no process |
| **Privacy** | HIPAA (US) / GDPR (EU) compliance; DPIA; data processing agreements; breach procedures | Architecture supports de-identification, encryption, audit; no policies, no DPIA |

## 3. Where this prototype falls short (honest summary)

1. **No real data, no performance evidence.** The only model is trained on synthetic images.
2. **No QMS, no design history file, no risk file, no usability file.** The code is structured to make those tractable (versioning, traceability, tests) but they do not exist.
3. **Research-only data licences.** CheXpert, MIMIC-CXR, VinDr-CXR, PadChest, BRAX, CANDID-PTX are not licensed for commercial use; only NIH ChestX-ray14 (and small sets) are. A product would need commercially licensable, multi-site, radiologist-labelled data.
4. **De-identification is not validated** to a standard (e.g. against a burned-in-text corpus or a formal PS3.15 conformance check); the pixel heuristic is not OCR.
5. **Security is development-grade:** local password auth, no SSO, no key management service, no SBOM, no penetration testing, TLS only at a reverse proxy that is not configured here.
6. **No human-factors evidence** that read-first mode reduces automation bias in practice.
7. **No PCCP / change control** beyond content-hashed versions.
8. **Interoperability** is tested against Orthanc/OHIF only; no IHE AI Results (AIR) profile conformance, no HL7 FHIR.

## 4. Sensible next steps toward a regulated path

1. Lock intended use with clinicians; decide 510(k) (identify predicate) vs De Novo; pre-submission (Q-Sub) meeting with FDA.
2. Stand up a QMS (ISO 13485) and start IEC 62304 documentation from the existing architecture and DECISIONS log.
3. Obtain commercially licensable, multi-site data with radiologist ground truth; freeze an external test set under a data-use agreement.
4. Run standalone testing per subgroup, then an MRMC reader study.
5. Formal risk management (ISO 14971) and usability engineering (IEC 62366-1).
6. Cybersecurity: threat model, SBOM, secure deployment reference (TLS, KMS, SSO).
7. Post-market plan: drift thresholds tied to actions, complaint handling, vigilance.

## 5. Useful references (verify currency)

- FDA: *Clinical Decision Support Software* (2022); *Marketing Submission Recommendations for a PCCP for AI-Enabled Device Software Functions* (2024); *Cybersecurity in Medical Devices* (2023); *Content of Premarket Submissions for Device Software Functions* (2023); FDA AI-enabled device list.
- EU: MDR 2017/745 Annex VIII Rule 11; MDCG 2019-11 (software qualification & classification); MDCG 2020-1 (clinical evaluation of MDSW); MDCG 2019-16 (cybersecurity); EU AI Act obligations for high-risk AI in medical devices.
- Standards: IEC 62304, ISO 14971, IEC 62366-1, ISO 13485, IEC 81001-5-1 (health software security), IHE AI Results (AIR) profile, DICOM PS3.15 (de-identification), PS3.16 TID 1500.
