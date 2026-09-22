For research use only. Not for diagnostic use.

# Open-source platforms and components for running AI in a PACS workflow

> Survey for the Second Reader project. Every fact carries a numbered citation; the reference list at the end gives the URL and the access date (all accessed 2026-09-21). Stars, licences, last-push dates and latest-release dates were read from the GitHub REST API on 2026-09-21 unless another source is cited; where a web page and the API disagreed, the API value is used. Anything I could not confirm against a primary source is marked **unverified**.
>
> Licence legend used throughout: **adopt** = permissive (MIT / BSD / Apache-2.0 / EPL-2.0 dual), code may be copied into this repo with attribution; **ideas only** = GPL / AGPL / non-commercial / RSNA Public License, we may read it and talk to it over a network boundary but must not copy code into `src/`.

## 0. One-page summary

| Component | What it is | Licence (verified) | Adopt code? | Maturity (2026-09-21) | Relation to D-002/D-004 |
|---|---|---|---|---|---|
| MONAI Deploy App SDK + MAP | Python SDK + container packaging for inference apps | Apache-2.0 [1] | adopt | 138 stars; PyPI 4.0.0 (2026-06-13) [2]; GitHub "latest release" still 3.0.0 (2025-04-23) [1][3] | complement (packaging only); needs CUDA 13 runtime even without GPU [4] |
| MONAI Deploy Informatics Gateway + Workflow Manager (Express) | DICOM in/out gateway + .NET workflow engine (Argo, RabbitMQ, MinIO, MongoDB) | Apache-2.0 [5][6] | adopt | IG 0.5.2 (2025-06-28), 31 stars; WFM 0.2.0 (2024-06-13), 17 stars, last push 2025-03-26; Express 0.6.0 "NOT recommended for production" [5][6][7] | would replace; too heavy, slow-moving |
| mercure | DICOM router/processor with Docker modules | MIT [8] | adopt | 100 stars; v0.4.1 2026-05-30 (security-audit release); last push 2026-07-09 [8][9] | would replace queue+router; ideas worth borrowing |
| Kaapana | Kubernetes platform (dcm4chee, Airflow, OpenSearch, MinIO, Keycloak, OHIF) | AGPL-3.0 [10] | ideas only | 275 stars; 0.7.0 2026-07-15; very active [10][11] | would replace everything; wrong scale |
| Orthanc (+ Python plugin, AI samples) | Lightweight DICOM server | GPLv3+ core; Python plugin AGPL; REST/DICOM use from another process explicitly allowed [12][13] | REST only (what we do) | Orthanc 1.13.0 (2026-08-15) [14]; we pin `orthancteam/orthanc:24.10.1` | keep |
| OHIF Viewer v3 | Web DICOM viewer with SR/SEG extensions | MIT [15] | adopt | 4,344 stars; v3.12.18 released 2026-09-21; master 3.14.0-beta.35 [15][16]; we pin `ohif/app:v3.9.2` | keep (D-015) |
| dcm4chee-arc-light + Weasis | Java EE archive + Java viewer | dcm4chee: MPL 1.1 per README [17]; Weasis: EPL-2.0 / Apache-2.0 dual [18] | adopt (Weasis); dcm4chee MPL is file-level copyleft | dcm4chee 5.35.1 2026-08-24, 501 stars; Weasis 4.7.3 2026-08-25, 1,334 stars [17][18] | would replace Orthanc; no gain |
| XNAT + Container Service | Research imaging data platform; Docker launcher | BSD 2-clause [19] | adopt | xnat 1.10.1 (2026-08-28); container-service 3.8.2 (2026-09-15) [19][20] | would replace; wrong problem |
| NVIDIA Holoscan (Clara Deploy legacy) | GPU streaming SDK; base of MONAI Deploy 4.x | Apache-2.0 [21] | adopt | v4.6.0 2026-09-03; x86_64 dGPU / IGX / Jetson only [21] | not applicable to CPU-only images (D-014) |
| highdicom | SR (TID 1500), SEG, SC, GSPS, KOS, PM writers/parsers | MIT [22] | adopt (already D-005) | 241 stars; 0.28.1 2026-07-28; push 2026-09-20 [22][23] | keep |
| pydicom | Core DICOM I/O | MIT [24] | adopt (already) | 3.0.2 on PyPI 2026-03-19 [24] | keep |
| IHE AIR / AIRA / AIW-I | Result encoding / result assessment / UPS workflow profiles | IHE documents | n/a | AIR Rev 1.3 TI 2025-08-08; AIRA TI 2025-06-12; AIW-I Rev 1.1 TI 2020-08-06 [25][26][27] | conformance target for `writer.py` |
| DICOM WG-23 outputs | Sup 219 (JSON SR), Sup 224 (Service Discovery), Sup 251 (Application Request) | DICOM | n/a | 219 in trial implementation; 224 draft v4 2021, may be withdrawn; 251 draft v5 2026-07-14 [28][29][30][31] | watch |
| FHIR ImagingStudy / DiagnosticReport / Observation / ImagingSelection | HL7 resources for reports and image references | HL7 (CC0 spec) | n/a | Observation normative; ImagingStudy FMM4; DiagnosticReport FMM3; ImagingSelection FMM1 (R5) [32][33][34][35] | cheap export add-on |
| Diana / RSNA CTP / MIRC | Orthanc analytics glue; clinical-trial de-id pipeline | Diana MIT (dormant 2018); CTP RSNA Public License [36][37] | Diana adopt (dead); CTP ideas only | CTP last push 2025-10-31 [37] | CTP pixel anonymiser is the reference design for our burned-in-text gate |

### 0.1 Routing, results and de-identification at a glance

How each platform gets a study to a model, gets results back, and what it does about identifiers. "Ours" is the current `docker-compose.yml` topology.

| Platform | Study → model | Results → archive | De-identification | Queue / state |
|---|---|---|---|---|
| Ours (D-002/D-004) | Orthanc A Python `STABLE_STUDY` callback → HTTPS webhook → platform pulls `/instances/{id}/file` over REST [13] | `POST /instances` to Orthanc B (REST), best-effort in `inference.py` | PS3.15 allowlist + HMAC pseudonyms + fail-closed verifier before any write | Postgres `jobs` table, `SKIP LOCKED` |
| MONAI Deploy IG + WFM | DICOM SCP / STOW-RS into IG → RabbitMQ event → WFM → Argo runs MAP over a folder [42][7] | IG export via DICOM SCU or STOW-RS [42] | IG plug-in hooks for de-/re-identification (C#), no profile [42] | RabbitMQ + MongoDB + MinIO [7] |
| mercure | storescp receiver → rules on tags → processor runs Docker module on `MERCURE_IN_DIR` [8][44] | dispatcher: DICOM, DICOM+TLS, DICOMweb, folder, rsync, SFTP, S3, XNAT [47] | separate GPL tag-level anonymizer module; no pixel de-id [49][50] | folders on disk + PostgreSQL bookkeeper [8] |
| Kaapana | `dcmsend` to dcm4chee (11112), AE title → project, Airflow DAG per workflow (**unverified**) [53] | back into dcm4chee / MinIO; OHIF reads it [10] | not surveyed (docs pages unreachable) | Airflow + OpenSearch [10] |
| Orthanc-internal AI (mammography, OrthancAI) | model runs *inside* Orthanc's Python plugin on stable study / UI trigger [55][56] | SR written straight back into the same Orthanc [55] | none (identifiable data in-process) | none |
| XNAT + Container Service | Event Service on session archive → Docker/Swarm/K8s container with bind mounts [20][61] | outputs written back as XNAT resources [20] | DicomEdit scripts at receive time [19] | XNAT DB |
| dcm4chee-arc-light | forwarding / export rules, UPS-on-store, HL7 [17][59] | C-STORE / STOW / FHIR export modules [59] | attribute coercion; de-id not surveyed | WildFly + RDBMS |
| IHE AIW-I (profile) | UPS-RS workitems: pull / triggered-pull / push [27] | AIR store transactions RAD-43 / RAD-108 [25] | out of scope | Task Manager (UPS-RS) |
| DICOM Sup 251 (draft) | client posts an Application Processing Request naming input data and retrieval protocol [31] | request names the store protocol for outputs [31] | out of scope | health / status transactions [31] |

Reading across the rows: every mature orchestrator treats *export of results* as a retried, observable step and *de-identification* as either absent or a bolt-on; none makes "identifiable data never reaches the model host" a structural property. That is the part of D-002 no surveyed tool would give us for free.

## 1. Orchestration platforms

### 1.1 MONAI Deploy App SDK and the MAP format

* **What it does.** "A framework and associated tools to design, develop and verify AI-driven applications in the healthcare imaging domain"; built-in operators load DICOM, select series, convert to volume, run PyTorch or remote Triton inference, and write DICOM SEG, Basic Text SR and encapsulated PDF [1]. The operator module list is `dicom_data_loader_operator.py`, `dicom_series_selector_operator.py`, `dicom_series_to_volume_operator.py`, `dicom_seg_writer_operator.py`, `dicom_text_sr_writer_operator.py`, `dicom_encapsulated_pdf_writer_operator.py`, `monai_seg_inference_operator.py` [38].
* **The SR it writes is not TID 1500.** `DICOMTextSRWriterOperator` sets `sop_class_uid = "1.2.840.10008.5.1.4.1.1.88.11"` (Basic Text SR) and encapsulates a free-text result [39]. Our `writer.py` (Comprehensive SR, TID 1500 measurement groups) is strictly richer.
* **MAP specification.** A MAP is a single OCI image with `/etc/holoscan/app.json` (execution command, input/output formats, health probes) and `/etc/holoscan/pkg.json` (models, resources); code under `/opt/holoscan/app/`, models under `/opt/holoscan/models/`, I/O under `/var/holoscan/input` and `/var/holoscan/output`; env vars `HOLOSCAN_INPUT_PATH`, `HOLOSCAN_OUTPUT_PATH`, `HOLOSCAN_MODEL_PATH` [40]. The packager is `monai-deploy package <app> --config app.yaml --tag t --platform x86_64`; it "ONLY supports a set of platform-specific base images from nvcr.io", and the only platforms documented are `x86_64` and `igx-dgpu` [41].
* **Hardware prerequisites (the blocker for us).** The README for the current SDK states it depends on Holoscan SDK (CUDA 13), Ubuntu 22.04 / glibc 2.35+, "CUDA 13.0 or above is required along with a supported NVIDIA GPU with at least 8GB of video RAM", and "if inference is not used ... and a GPU is not installed, at least CUDA 13 runtime is required" [4]. That conflicts with D-014 (CPU-only Docker images, MPS on the dev Mac).
* **Maturity.** Apache-2.0, 138 stars, last push 2026-09-17 [1]; PyPI 4.0.0 released 2026-06-13 with Python 3.10–3.13 [2]; GitHub tags include `4.0.0`, `3.5.0`, `3.4.0` but the "latest release" object is still 3.0.0 (2025-04-23) [1][3].
* **Routing / results.** The SDK itself does not route; it consumes a folder and writes a folder. Routing is the Informatics Gateway's job (next section).

### 1.2 MONAI Deploy Informatics Gateway, Workflow Manager, MONAI Deploy Express

* **Informatics Gateway** "facilitates integration with DICOM compliant systems, enables ingestion of imaging data, helps triggering of workflows with the MONAI Deploy Workflow Manager and offers pushing the output to PACS systems" [5]. Release notes list DICOM SCP and DICOMweb STOW-RS ingestion, DICOM SCU and DICOMweb export, and (v0.4.0) a "lightweight plug-in engine" for de-identification / re-identification of in- and outbound data [42]. Apache-2.0, 31 stars, latest release 0.5.2 (2025-06-28), last push 2026-03-05 [5].
* **Workflow Manager** is a .NET 6 service; Express runs it with Task Manager, RabbitMQ (broker), MinIO (objects), MongoDB (state), optional Orthanc as PACS simulator and an ELK stack; MAPs are executed through Argo [6][7]. Apache-2.0, 17 stars, latest release 0.2.0 (2024-06-13), last push 2025-03-26 [6].
* **Express** `VERSION` is 0.6.0 and its README says "This is NOT recommended for production environments" [7][43].
* **Standards trail.** DICOM Sup 251 "Application Request" states it is "partially based on an API proposed by the American College of Radiology's Data Science Institute (ACR DSI) that was extended by the MONAI Deploy Informatics Gateway (MIG) for inference requests" [31] — the MIG inference-request API is the seed of a future DICOMweb service.
* **Relation to our design.** Replaces Orthanc A webhook + FastAPI ingest + Postgres queue with three extra services (RabbitMQ, MinIO, MongoDB) plus Argo and .NET. The parts we would keep (de-id, encrypted blobs, reveal gating, SR writer) are not provided by it. Complement value is low; the de-id "plug-in" is a C# hook, not a profile implementation.

### 1.3 mercure

* **What it does.** "A flexible DICOM routing and processing platform with web interface and monitoring", Python + DCMTK; components: receiver (storescp → JSON tag extraction), router (series/study completion + rules → outgoing/processing/discard), processor (Docker modules), dispatcher, cleaner, webgui, bookkeeper (PostgreSQL) [8]. MIT, 100 stars, last push 2026-07-09 [8]. Latest release v0.4.1 (2026-05-30) "addresses 10 vulnerabilities identified in a security audit" of 0.4.0-beta.9 [9].
* **Module contract** (question (d) below) [44]:
  * a Docker image with a `docker-entrypoint.sh`, started once per series (series-level rule) or once per study (study-level rule);
  * input in `MERCURE_IN_DIR`: files named `[series_UID]#[file_UID].dcm`, each with a sibling `.tags` JSON of extracted header values so a module need not parse DICOM;
  * settings read from `task.json` under `"process" > "settings"` (global module settings merged with per-rule overrides);
  * output DICOM written to `MERCURE_OUT_DIR`; an optional `result.json` is "archived ... regardless of the JSON structure", and `{"__mercure_notification": {"text": ...}}` triggers alerts.
  * UI-side, a module is a Docker tag with type "mercure" or "MONAI Application Package (MAP)"; MAPs need the "Requires Root User" switch; there is an "Enable NVIDIA GPU Support" switch and a Nomad orchestration tab [45]. The docs index states "mercure now supports MONAI Application Packages (MAP) as processing modules" [46].
* **Results back.** Dispatch targets: DICOM, DICOM+TLS, DICOMweb (basic or token auth), folder, rsync, SFTP, S3, XNAT [47]. Modules such as `mercure-exampleinference` (ONNX prostate segmentation), `mercure-totalsegmentator`, `mercure-monaisegment`, `mercure-anonymizer`, `mercure-decompressor` are published [48].
* **De-identification.** `mercure-anonymizer` is tag-level (remove / clear / replace with `@fake_name@`, `@random_uid@`, date truncation, per-AET project settings, "default"/"none" presets); pixel de-identification is not addressed [49]. Note its repository licence is **GPL-3.0** [50], as is **dcmannotate** (GPL-3.0, last push 2022-10-07) [51], mercure's helper that writes TID 1500 SR "compatible with OHIF", burned-in secondary capture, and Visage annotations [52]. Both are ideas only.
* **Relation to our design.** mercure is exactly the layer D-002/D-004 implement (receive → rule → container → dispatch → bookkeep in Postgres). Adopting it would replace `docker/orthanc-a/forward_to_platform.py`, `core/serving/queue.py`, `core/serving/worker.py` and the Orthanc B push in `core/serving/inference.py`, but it has no de-id-before-storage boundary, no PHI verifier, no encrypted blob store and no reveal gating — those would still be ours, running inside a mercure module. Net: not a replacement; its `task.json` / `result.json` contract is worth copying as an *out-of-process* plugin option (MIT, so we can lift code).

### 1.4 Kaapana (DKFZ)

* **What it does.** "Open-source toolkit for medical data analysis platforms" supporting AI workflows and federated learning; Kubernetes + Helm, dcm4chee (PACS), MinIO, OpenSearch (metadata), Apache Airflow (workflows), Keycloak (auth), Prometheus/Grafana, OHIF; ships nnU-Net and TotalSegmentator [10]. AGPL-3.0, 275 stars, releases 0.6.0 (2025-12-18: Task/Workflow/Data APIs, multi-node via Longhorn), 0.6.1 (2026-02-23), 0.7.0 (2026-07-15: Extension Manager, service-to-service auth, private datasets) [11]; last push 2026-09-21 [10].
* **Routing.** Images arrive by DICOM C-STORE (`dcmsend`) on port 11112; the called AE title (`kp-<project short_id>`) selects the project; the Airflow DAG `service-process-incoming-dcm` runs `dcm2json` and indexes metadata into OpenSearch; workflows are launched from the UI on datasets (search-result snippets of the Kaapana docs and 0.7.0 changelog — **unverified**: the FAQ page 404'd on both `latest` and `stable` paths) [53].
* **Relation to our design.** Whole-platform replacement, Kubernetes-only, AGPL (a hosted deployment must publish modifications). Wrong scale for a one-modality research prototype; the project-scoped datasets and Airflow DAG-per-model idea are useful for a later multi-site evaluation platform, not for serving.

### 1.5 Orthanc: licence position, Python plugin callbacks, AI recipes

* **Licence.** Orthanc core is GPLv3+; the scalability plugins (PostgreSQL, Web viewer) are AGPLv3+; the licensing FAQ table marks "Calling Orthanc from a third-party system (using REST API or DICOM protocol), even if some AGPL-licensed plugin is installed" as permitted for proprietary, GPL and AGPL callers, and notes the wording "third-party system" is deliberately broad [12]. Our platform only talks REST (`core/pacs/orthanc.py`), so the GPL stops at the process boundary. The **Python plugin is AGPL**, and the book says source disclosure is required "if you distribute Orthanc with Python scripts to clients or deploy it on web portals" [13]; `docker/orthanc-a/forward_to_platform.py` is the only code we run inside that boundary, it is 40 lines and already public in this repo.
* **Callbacks** available to a Python plugin [13]:
  * `RegisterOnChangeCallback` — resource lifecycle events including `ChangeType.STABLE_STUDY` / `STABLE_SERIES` (what `forward_to_platform.py` uses);
  * `RegisterOnStoredInstanceCallback` — per received instance;
  * `RegisterReceivedInstanceCallback` — modify an incoming DICOM before it is stored (a possible de-id-at-the-door hook, but it runs inside the identifiable PACS);
  * `RegisterRestCallback` and `RegisterIncomingHttpRequestFilter` — extend / authorise the REST API.
  * Constraint: because of the CPython GIL "only one [thread] can proceed at any given time" and "callbacks must be fast"; CPU-heavy work should go to a `multiprocessing` pool. The book's auto-routing sample forwards a study on `STABLE_STUDY` with `orthanc.RestApiPost('/modalities/sample/store', resourceId)`.
* **Built-in de-identification.** `/studies/{id}/anonymize` (and instance/series/patient/bulk variants) implement PS3.15 profiles `2008`, `2017c`, `2021b` (1.9.4+) and `2023b` (1.12.1+), default = highest supported; options `Replace`, `Keep`, `Remove`, `KeepPrivateTags`, `DicomVersion`; new UIDs are generated; pixel de-identification is not covered [54].
* **AI recipes and plugins.**
  * `jodogne/orthanc-mammography` — Python plugin that runs a deep-learning mass detector on MG studies from Orthanc Explorer, writes DICOM SR back into Orthanc, viewed in Stone Web Viewer; ~5 s per exam; AGPL-3.0, 11 stars, last push 2024-08-31 [55].
  * `NeuroSainteAnne/OrthancAI` — "wraparound around Orthanc and Orthanc Python plugin" with modules exposing `__init__` (model preload) and `process` (called at patient/study/series trigger), cleanup of temporary files; AGPL-3.0, 0 stars, last push 2024-04-09 [56].
  * `ImagingInformatics/orthanc-ai-orchestrator` — SIIM-sponsored plugin that creates UPS-style workitem tickets (SCHEDULED / IN PROGRESS / CANCELED / COMPLETED) 5 s after a study is uploaded; MIT, 15 stars, last push 2021-04-25 [57].
  * The 2026 JIIM paper "A Plugin-Based Architecture for Integrating AI Services in an Open-Source PACS" (Jesus et al., published 2026-02-19) concerns **Dicoogle**, not Orthanc [58].
* **Version drift.** Orthanc NEWS lists 1.13.0 released 2026-08-15 [14]; `docker-compose.yml` pins `orthancteam/orthanc:24.10.1`.
* **Relation to our design.** Keep. The AI plugins above all run the model *inside* Orthanc's Python process (GIL, AGPL, identifiable data in the same process), which is precisely what D-002 avoids.

### 1.6 dcm4chee-arc-light and Weasis

* **dcm4chee-arc-light** is a "DICOM Archive Java EE application running in WildFly" with LDAP configuration; the module tree includes DIMSE, STOW/QIDO/WADO, UPS, HL7, coercion and export modules for DICOM, FHIR, HL7 ORU, UPS and storage; 5.34.3 added "FHIR Imaging Study creation"; 5.35.x fixed several RCE/XXE advisories [17][59]. README licence: Mozilla Public License 1.1 [17] (the MPL/GPL/LGPL tri-licence often quoted for dcm4che — **unverified** for this repo). 501 stars, 5.35.1 (2026-08-24), last push 2026-09-21 [17].
* **Weasis** README: "Structured report viewer with hyperlinks to images and associated graphics", "DICOM Presentation States (GSPS) — applied with graphics rendered as overlays", SEG overlay in 2D/MPR/3D, RT structure/dose, parametric maps; a Java desktop application launched from the web, dual-licensed EPL-2.0 / Apache-2.0 (both licence files in the repo root) [18][60]. 1,334 stars, v4.7.3 (2026-08-25) [18].
* **Relation.** dcm4chee would replace Orthanc A/B at much higher operational cost (WildFly, LDAP) with no functional gain for us; its FHIR export is the one feature Orthanc lacks. Weasis cannot be embedded in a React iframe (D-015) but is the best open viewer for checking GSPS/SR rendering independently of OHIF.

### 1.7 XNAT and the Container Service

* **XNAT** (NrgXnat/xnat monorepo, 1.10.1, 2026-08-28) is BSD 2-clause; it bundles the `mizer` DICOM anonymiser with the DicomEdit 4/6 scripting DSLs [19]. The **Container Service** plugin (3.8.2, 2026-09-15) runs Docker/Swarm/Kubernetes containers against XNAT data via command/wrapper definitions and bind mounts, and can be triggered by the Event Service (e.g., on session archive) [20][61]. The XNAT ML plugin suite was built around NVIDIA Clara Train 2.0 and has been deprecated by the XNAT team (wiki, via search — **unverified**) [62].
* **Relation.** A research data-management platform, not a serving path; it would replace our Postgres/blob store and add a JVM stack. Its DicomEdit scripts are a useful cross-check corpus for `find_phi` rules.

### 1.8 NVIDIA Holoscan and Clara Deploy (legacy)

* Holoscan SDK is "the AI sensor processing SDK for low latency streaming workflows"; Apache-2.0, 222 stars; v4.6.0 (2026-09-03), v4.5.0, v4.4.0 all list supported platforms as x86_64 dGPU, IGX and Jetson via CUDA 12/13 containers; CPU-only is not listed [21][63]. MONAI Deploy App SDK 4.x is built on it and inherits the CUDA requirement [4].
* Clara Deploy SDK was consolidated into Clara Holoscan and NVIDIA directed Clara Deploy users to the MONAI Deploy Working Group (NVIDIA blog, via search) [64]; Clara Deploy docs are archived at 0.8.1 [65].
* **Relation.** Irrelevant to a CPU-first chest-X-ray classifier; relevant only if the ultrasound plugin ever needs real-time cine streaming on an IGX box.

### 1.9 Diana, RSNA CTP, MIRC, pixel checkers

* **Diana** (Derek Merck): "DICOM Analytics and Archive" — Orthanc for storage/proxy, Splunk for indexing, Celery "star" workers; MIT, 11 stars, last push 2018-11-30 (dormant) [36].
* **RSNA CTP**: stand-alone pipeline engine with `DicomImportService`, `DicomAnonymizer` (script-based), `DicomPixelAnonymizer` (blanks burned-in text regions), `DicomFilter` (quarantine), `IDMap`, `DicomExportService`; DICOM and HTTP(S) transport [66]. Source headers: "released under the terms of the RSNA Public License" — treat as **ideas only** [37]. 70 stars, last push 2025-10-31 [37]. MIRC is the RSNA project family that hosts CTP [66].
* **Burned-in-text tools**: `pydicom/dicom-cleaner` (OCR-based detection, "under development", no licence file, last push 2019) and `nroduit/image-ocr-identifier` (PaddleOCR FastAPI microservice returning mask coordinates for Karnak, v0.1.0 2026-09-08, licence not asserted on GitHub) [67][68]. I found no project literally named "pixel-checker" — **unverified/likely misremembered name**.

## 2. Viewers: how OHIF renders TID 1500 SR and Secondary Capture

Source read on `master` (3.14.0-beta.35) [16]; behaviour of the pinned `ohif/app:v3.9.2` is **unverified** and must be tested.

* **SOP classes handled by `@ohif/extension-cornerstone-dicom-sr`**: Basic Text, Enhanced, Comprehensive and Comprehensive 3D SR. The handler takes the measurement path only when the root concept name is `(126000, DCM, "Imaging Measurement Report")` *and* a `ContentSequence` exists; otherwise the SR is treated as plain text (to avoid a crash) [69].
* **Parsing**: `ImagingMeasurements (126010)` → `MeasurementGroup (125007)`; each group needs a `TrackingUniqueIdentifier` and `TrackingIdentifier`; SCOORD/SCOORD3D `GraphicType` POINT, POLYLINE, ELLIPSE, CIRCLE are read; `TEXT_ANNOTATION_POSITION` items are excluded; unsupported graphic types log a console warning and are skipped [69]. The renderer (`getRenderableData.ts`) draws POINT, MULTIPOINT, POLYLINE, CIRCLE and ELLIPSE [70].
* **Hydration into editable annotations** uses `MeasurementReport.generateToolState` from `@cornerstonejs/adapters` 5.10.3 / dcmjs 0.52.0 [71][72]. `isRehydratable` accepts a measurement group if either [73]:
  * its `TrackingIdentifier` text matches the Cornerstone convention `"<Cornerstone3DTools tag>:<ToolType>"` [74], or
  * `getAdaptersForTypes(graphicCode, graphicType, pointsLength)` finds an adapter registered for that combination.
  * `RectangleROI` registers `('DCM:111030', 'POLYLINE', 4)` and `('DCM:111030', 'POLYLINE', 5)`, so a five-point closed POLYLINE under `(111030, DCM, "Image Region")` hydrates as a rectangle even without the Cornerstone tracking string [75]; a generic closed contour goes to `PlanarFreehandROI` (closed if first and last points differ by < 1e-5) [76].
  * Adapters available: Angle, ArrowAnnotate, Bidirectional, CircleROI, CobbAngle, ControlPointPolyline, EllipticalROI, KeyImage, LabelData, Length, PlanarFreehandROI, Probe, RectangleROI, UltrasoundDirectional, plus Segmentation, RTStruct and ParametricMap sub-adapters [77].
* **Limitations stated in code**: "NO SUPPORT FOR SRs THAT HAVE MEASUREMENTS FROM MULTIPLE STUDIES"; image lookup is keyed `${ReferencedSOPInstanceUID}:${frameNumber}` and the authors note dcmjs/Cornerstone3D "has structural defect in supporting multi-frame files"; an `onBeforeSRHydration` hook allows filtering [78]. Components: `OHIFCornerstoneSRMeasurementViewport`, `OHIFCornerstoneSRTextViewport`, `OHIFCornerstoneSRContainer` [79].
* **Secondary Capture**: OHIF's `sopClassDictionary` defines `SecondaryCaptureImageStorage` (1.2.840.10008.5.1.4.1.1.7) and the multi-frame SC variants alongside CR and DX-for-presentation [80]; the default stack handler's SOP list begins with CR, DX (presentation/processing), MG, CT, MR ... [81] — the excerpt I read did not reach the SC entries, so "SC is in the default list" is **unverified** from source, though SC images are routinely shown by OHIF as ordinary image series.
* **Weasis** claims SR display with hyperlinks to referenced graphics and GSPS overlays [18] — the natural second viewer for a rendering cross-check.

## 3. Standards

### 3.1 IHE AI Results (AIR) — Rev 1.3, Trial Implementation, 2025-08-08

* **Status.** Supplement to IHE RAD TF V23.0, "published on August 8, 2025 for trial implementation", including the 2022 "AIR+" extensions [25]. (The IHE profiles index still lists it as Rev 1.2 revised 2025-08-08 — index metadata lags the PDF [82].)
* **Actors.** Evidence Creator, Image Manager / Image Archive, Image Display, Imaging Document Consumer [25 §49.1]. Options: Result Tree Option (Evidence Creator, Imaging Document Consumer). Required grouping: Evidence Creator with ITI Consistent Time (CT) Time Client [25 §49.2–49.3].
* **Transactions (Table 49.1-1).** Evidence Creator initiates at least one of Store Evidence Documents [RAD-43], Creator Images Stored [RAD-18], Key Image Note Stored [RAD-29], Store Instances Over the Web [RAD-108]. Image Manager/Archive responds to RAD-43, RAD-18, Query Analysis Results [RAD-137], Retrieve Evidence Documents [RAD-45], Retrieve Images [RAD-16], Retrieve Key Image Notes [RAD-31], QIDO-RS Query [RAD-129], WADO-RS Retrieve [RAD-107]. Image Display performs Display Analysis Result [RAD-136] [25 §49.1].
* **Result primitives (RAD TF-3 §6.5.3).** Qualitative Findings, Measurements, Locations, Regions, Parametric Maps, Tracking Identifiers, Image References, Result Trees [25 §6.5.3.2–6.5.3.9]. Secondary Capture was deliberately excluded ("Should we include a Secondary Capture primitive? A. No ... anyone that wants to store/display Secondary Capture doesn't need a profile") [25, closed issue 3].
* **Encoding rules that matter for us.**
  * Qualitative findings, locations and regions "shall be encoded in an instance of the DICOM **Comprehensive 3D SR** SOP Class using TID 1500 (Measurement Report) as the root template"; the profile answers "Should SR requirements describe each of the Enhanced, Comprehensive, and Comprehensive 3D IODs? A: No. Just Comprehensive 3D" [25 §6.5.3.2, closed issue 11].
  * Findings go in TID 1501 (whole image), TID 1410 (planar ROI) or TID 1411 (volumetric ROI); the "Qualitative Evaluations" container of TID 1500 "shall not be used" [25 §6.5.3.2].
  * Negative findings "shall" use a negative concept, e.g. `(373572006, SCT, "Clinical finding absent") = (finding)` (Vol 3) / `(444436002, SCT, "Clinical finding not suspected")` (Vol 1 text); the TID 1350 pattern `Finding = X` > `Presence of property = Absent` is described as "not adopted by this profile", and re-using a finding code as a concept with value Absent is "not appropriate" per SNOMED [25 §49.4.1.6, §6.5.3.2].
  * The Creator "shall describe each algorithm ... in the Contributing Equipment Sequence (0018,A001)" with Purpose of Reference `(109102, DCM, ...)`; values must match TID 4019 / TID 1004 items; "shall populate the Observation UID (0040,A171)" on finding content items and tracking identifiers [25 §6.5.3.1].
  * 2D point/line locations: SCOORD `(111030, DCM, "Image Region")` POINT / POLYLINE in TID 1501; planar regions: TID 1410 with SCOORD POLYLINE, CIRCLE or ELLIPSE; a bounding box is flagged with `(130400, DCM, "Geometric purpose of region")` [25 §6.5.3.4–6.5.3.5].
  * Confidence/probability: "another item for important future work"; "for now this is all implementation choice. The AI Model is free to provide whatever details"; `(111012, DCM, "Certainty of Finding")` exists in CAD templates [25, closed issue 34].
  * Result Trees (AIR+) are Basic Text SR instances using TID IHERADAIR1/2 that point at other result objects [25 §6.5.3.9].
* **Display [RAD-136].** An Image Display "shall be able to display each region in a way that the user can see the boundary", superimpose multiple regions, show region metadata and "allow the user to toggle the display of each region on/off" [25 §4.136.4.1.3.5].
* **AIRA** ("AI Result Assessment for Imaging", TI, published 2025-06-12) covers recording a clinician's verification of AI findings [26][82] — the standards home for our agree/disagree feedback (principle 8); content not reviewed here.
* **Open implementations.** I found none that claims AIR conformance; highdicom's TID 1500 classes cover all SR primitives (§4.1), and OHIF covers the display side for 2D SCOORD.

### 3.2 IHE AI Workflow for Imaging (AIW-I) — Rev 1.1, TI, 2020-08-06

Actors Task Requester, Task Manager, Task Performer, Watcher (+ Image Manager/Archive); all workflow transactions are DICOM UPS-RS: Create [RAD-80], Query [RAD-81], Claim [RAD-82], Get [RAD-83], Update [RAD-84], Complete [RAD-85], Manage Subscription [RAD-86], Send Notification [RAD-87], Request Cancelation [RAD-88], Open Event Channel [RAD-109]; results are stored by the Task Performer via RAD-8/RAD-18 and AIR; pull, triggered-pull and push workflows are all allowed [27]. Our Orthanc-A-webhook → platform-pull is a "triggered pull" without UPS; `orthanc-ai-orchestrator` [57] is the only open toy of the UPS ticket model I found.

### 3.3 DICOM WG-23 outputs and supplement titles (verified)

* WG-23 "Artificial Intelligence/Application Hosting" lists Sup 219 (JSON Representation of DICOM Structured Reports) as "out for trial implementation" and Sup 224 (Service Discovery and Control) as in progress and possibly to be withdrawn or revised; future work explicitly includes machine-consumable result formats and modernising Application Hosting via DICOMweb [28].
* **Sup 219** — "JSON Representation of DICOM Structured Reports": new PS3.22, TID 1500 as the model, human-readable "business names" for coded tuples, full-fidelity round trip; frozen draft 14 exists; the supplements page still says "will be further presented ... before Public Comment voting" (page metadata stale vs WG-23 page) [29][83].
* **Sup 224** — "Service Discovery and Control", WG-23, Version 4, 2021-08-03: application definitions in YAML (Open Application Model, Kubernetes, Dapr), ingest methods C-STORE / STOW / file path / API, egress C-STORE / file path, licence options, secrets [30].
* **Sup 251** — "Application Request", WG-23, Version 5, 2026-07-14, draft: adds an "Application Processing Request Service" to PS3.18 with request, completion-results, check-health (liveness/readiness) and check-processing-status transactions; interaction with UPS-RS "is not specifically defined" [31].
* **Sup 230** is "Update BCP Secure Communications Profiles" and **Sup 236** is "Waveform Presentation State" (WG-32) — neither is AI-related; there is no supplement titled "Structured Reporting for AI" [84][85].
* **Key Object Selection**: the KOS Document IOD flags significant images/objects; TID 2010 has Document Title (CID 7010), optional Observer Context, Key Object Description, ≥ 1 referenced IMAGE / WAVEFORM / COMPOSITE, optional Image Library; root-only, non-extensible [86][87]. IHE uses it as the Key Image Note (RAD-29) [25].

### 3.4 FHIR imaging resources

* **Observation** (normative since R4): `code`, `value[x]` (Quantity / CodeableConcept / …), `interpretation`, `bodySite`, `method`, `device`, `derivedFrom` (ImagingStudy, ImagingSelection …), `component`, `focus` [32].
* **DiagnosticReport** R5 (FMM 3, TU): `status` (registered | partial | preliminary | final), `code`, `subject`, `performer`, `result` → Observation, `study` → ImagingStudy (replaces R4 `imagingStudy`), `media.link` → DocumentReference, `conclusion` (markdown), `conclusionCode`, `presentedForm`, `composition` [33][88].
* **ImagingStudy** R5 (FMM 4, TU): DICOM Study UID in `identifier`, `series.uid`, `series.instance.uid/sopClass`, `endpoint` (WADO-RS) [34].
* **ImagingSelection** R5 (FMM 1, TU): `studyUid`, `seriesUid`, `instance.uid`, `imageRegion2D.regionType` ∈ {point, polyline, interpolated, circle, ellipse} with coordinate list, `imageRegion3D`, `derivedFrom` [35].
* dcm4chee has a FHIR export module and "FHIR Imaging Study creation" [59]; Orthanc has none.

## 4. Libraries for writing results

### 4.1 highdicom (MIT)

`highdicom.sr`: `MeasurementReport` (TID 1500), `MeasurementsAndQualitativeEvaluations` (TID 1501), `PlanarROIMeasurementsAndQualitativeEvaluations` (TID 1410), `VolumetricROIMeasurementsAndQualitativeEvaluations` (TID 1411), `ImageRegion` / `ImageRegion3D` / `VolumeSurface`, `ReferencedSegment(ationFrame)`, `Measurement` (TID 300), `QualitativeEvaluation`, `TrackingIdentifier`, `AlgorithmIdentification` (TID 4019), `ObservationContext`, document classes `EnhancedSR`, `ComprehensiveSR`, `Comprehensive3DSR` [89]. `highdicom.pr`: `GrayscaleSoftcopyPresentationState`, `ColorSoftcopyPresentationState`, `PseudoColorSoftcopyPresentationState`, `AdvancedBlendingPresentationState`, `GraphicAnnotation`, `GraphicObject`, `TextObject`, `GraphicLayer`, `GraphicGroup`, `GraphicTypeValues` (the docs page for `pr` is "under construction"; class list read from `src/highdicom/pr/__init__.py`) [90][91]. Also `highdicom.sc.SCImage`, `highdicom.ko.KeyObjectSelectionDocument`, `seg`, `pm`, `ann`, legacy converted enhanced images [22][23]. 0.28.1 (2026-07-28), Python ≥ 3.10 [23].

### 4.2 pydicom-only alternatives

`pydicom.sr` provides `Code`, `Collection` and the `codes` object for DICOM CIDs / SNOMED / UCUM; it has no template builder or content-tree validation [92]. dcmjs (JS, MIT, 350 stars, 0.52.0 2026-05-19) [72] and mercure's dcmannotate (GPL) [51] are the other TID 1500 writers; neither adds anything for a Python server.

## 5. Answers to the concrete questions

**(a) Does OHIF v3 display TID 1500 SR with POLYLINE bounding boxes out of the box?** Yes for read-only display and, in the current source, also for hydration: the SR extension parses TID 1500 groups with SCOORD POINT/POLYLINE/ELLIPSE/CIRCLE and renders them [69][70]; a five-point closed POLYLINE under `(111030, DCM, "Image Region")` matches the `RectangleROI` adapter by (code, type, point count) [75], so our `_box_region` output (closed 5-point POLYLINE from `hd.sr.ImageRegion`, `writer.py`) should both render and hydrate. Preconditions: root code 126000 with a `ContentSequence` [69]; `TrackingUniqueIdentifier` present (highdicom `TrackingIdentifier(uid=…)` supplies it); the referenced image must be in the same study in Orthanc B (the `${SOPInstanceUID}:${frame}` lookup) [78]. Limitations: SCOORD3D/multi-study SRs unsupported, TEXT annotations skipped, multi-frame lookups fragile [69][78]; behaviour of the pinned v3.9.2 image is **unverified**. Our gate-only SR (no region, `TrackingIdentifier="gate"`) will not hydrate and its display path (measurement vs text viewport) is **unverified** — needs a fixture test.

**(b) What would IHE AIR conformance take?** As an Evidence Creator:

1. emit **Comprehensive 3D SR** instead of Comprehensive SR [25 §6.5.3.2] — a one-line change from `hd.sr.ComprehensiveSR` to `hd.sr.Comprehensive3DSR`, since the 3D IOD is a superset;
2. add `ContributingEquipmentSequence` (0018,A001) with `PurposeOfReferenceCodeSequence` 109102 mirroring our `AlgorithmIdentification` name/version, and `ObservationUID` on finding items [25 §6.5.3.1];
3. encode absent findings with `(373572006, SCT, "Clinical finding absent") = finding` rather than only our `Presence = Absent` qualitative evaluation [25 §6.5.3.2];
4. add `(130400, DCM, "Geometric purpose of region")` = bounded-by on box regions [25 §6.5.3.5];
5. store via RAD-108 STOW-RS or RAD-43 C-STORE — we use Orthanc REST `POST /instances`, which is neither; a DICOMweb STOW-RS push to Orthanc B's `/dicom-web/studies` would be transaction-conformant [25 §49.1];
6. group with an ITI Consistent Time client [25 §49.3].

Our probability measurement (private 99SRD code) is consistent with AIR's "implementation choice" stance on confidence [25 issue 34]; SC overlays sit outside AIR by design [25 issue 3]. Otherwise our SR content (TID 1410 planar groups with finding type, TID 4019 algorithm identification, tracking identifiers, SCOORD POLYLINE) is close: the gap is IOD label, provenance attributes and the negative-finding idiom, not structure.

**(c) Is packaging our plugin as a MAP worth it?** Not now. MAP 4.x hard-requires the CUDA 13 runtime, Ubuntu 22.04 and nvcr.io base images even for CPU inference [4][41], contradicting D-014; the MAP I/O contract is a folder, so it would sit *outside* `ModalityPlugin` as a second packaging artefact, not replace it. Only the Informatics Gateway / Workflow Manager (slow-moving: WFM 0.2.0 from 2024, Express "not for production" [6][7]) and mercure [45] consume MAPs. Revisit if a partner site runs MONAI Deploy Express or mercure and asks for a MAP; the wrapper would call the existing CLI over `/var/holoscan/input` → `/var/holoscan/output` [40].

**(d) mercure's module interface vs `ModalityPlugin`.** mercure's contract is *process-level*: container + folder of DICOM files + `.tags` JSON + `task.json` settings → folder of DICOM + optional `result.json` [44]. Ours (`core/plugins/protocol.py`) is *in-process*: `accepts(header)` → `quality_gate(ImagingInput)` → `predict(ImagingInput) -> list[Finding]` with a mandatory calibrated probability and localisation, and the core writes SR/SC (`results/writer.py`). mercure has no gate concept, no typed findings, no calibration requirement and leaves SR/SC creation to each module (dcmannotate, GPL). The two compose: a mercure module could host our worker (mercure as PACS-facing router into `/v1/infer`), or we could add a "folder plugin" adapter that speaks mercure's contract for third-party models. The trade: our protocol guarantees principle 3 (never a finding without probability + localisation) at type level; mercure's guarantees isolation and language independence.

**(e) Is a FHIR DiagnosticReport export a cheap add-on?** Yes. One JSON-serialisable mapping from `ResultBundle.json` (`writer.py`):

* `DiagnosticReport.status = preliminary` (never `final`, mirroring `is_final=False` in the SR), `code` = a chest-radiograph report code, `conclusion` carrying the research-use label [33];
* `DiagnosticReport.study` → `ImagingStudy` identified by StudyInstanceUID (R5; `imagingStudy` in R4) [33][34][88];
* `result` → one `Observation` per finding: `code` = SNOMED finding, `valueQuantity` = calibrated probability, `interpretation` present/absent, `bodySite` chest, `device` = model version, `method` = algorithm [32];
* `Observation.derivedFrom` → `ImagingSelection` with `instance.uid` = source SOP and `imageRegion2D.regionType = polyline` for the box [35].

No new dependency and no server: emit a Bundle to the blob store and serve it from the results API. Caveats: ImagingSelection is FMM 1 and may change [35]; there is no FHIR server in our stack and Orthanc offers no FHIR façade (dcm4chee does [59]); D-021 gating must apply to the FHIR route exactly as to SR/SC.

## 6. Implications for Second Reader

1. **`src/secondreader/core/results/writer.py` — switch to Comprehensive 3D SR.** AIR requires the Comprehensive 3D SR SOP class and states the 3D IOD is a superset of Comprehensive/Enhanced [25 §6.5.3.2, issue 11]; highdicom ships `Comprehensive3DSR` [89]; OHIF handles both [69]. Update D-005/D-022 wording accordingly.
2. **`writer.py` — add AIR provenance attributes.** `ContributingEquipmentSequence` (0018,A001) with purpose 109102 and `ObservationUID` (0040,A171) on each finding/tracking item are "shall" requirements [25 §6.5.3.1]; our `AlgorithmIdentification(name, version=model_version_id)` already carries the values to mirror.
3. **`writer.py` / D-022 — negative-finding pattern.** AIR explicitly does not adopt the `Presence = Absent` modifier pattern and mandates a negative concept `(373572006, SCT, "Clinical finding absent") = finding` [25 §6.5.3.2]. Emit the AIR pattern for absent findings (keeping the probability measurement); keep the current Presence evaluation only if a consumer needs it.
4. **`writer.py` — bounding-box semantics.** Add `(130400, DCM, "Geometric purpose of region") = (75958009, SCT, "Bounded by")` to box regions so displays know the polyline is a box, not a contour [25 §6.5.3.5].
5. **`docker/ohif/app-config.js`, `docker-compose.yml` — pin and test.** Our SR hydration analysis is against OHIF master (3.14.0-beta.35) while compose pins `ohif/app:v3.9.2`; latest release is v3.12.18 (2026-09-21) [15][16]. Add an integration test that uploads a synthetic image + our SR to Orthanc B and asserts OHIF's SR handler marks it hydratable (RectangleROI by `('DCM:111030','POLYLINE',5)` [75]); then bump the image. Likewise Orthanc 1.13.0 vs the pinned 24.10.1 image [14].
6. **`docker/orthanc-a/forward_to_platform.py` — keep the boundary, shorten the callback.** The Orthanc licensing FAQ explicitly permits REST/DICOM callers of any licence [12], so D-002 is licence-clean; but the Python plugin doc says callbacks must be fast because of the GIL [13], and ours blocks for up to 10 s on `urlopen` inside `on_change`. Hand off to a thread/queue and return immediately; document that the script itself is under the AGPL plugin's terms.
7. **`core/serving/inference.py::_push_archive` — make the Orthanc B push a retried job.** The push to Orthanc B is best-effort and failures are only logged, so a result can exist in Postgres but never reach OHIF; every surveyed orchestrator (mercure dispatcher with retry [8], MONAI IG export [42], AIR RAD-108/RAD-43 store transactions [25]) treats export as a first-class, retried step. Add an `export` job kind to `queue.py`/`handlers.py` (D-004 already supports kinds and `max_attempts`), and consider STOW-RS to Orthanc B's `/dicom-web` for AIR transaction alignment.
8. **`core/serving/queue.py`, `worker.py` — keep D-004.** mercure's bookkeeper is also PostgreSQL [8]; MONAI WFM needs RabbitMQ + MongoDB + MinIO + Argo [7]; Kaapana needs Kubernetes + Airflow [10]. Nothing in the survey argues for a broker at our volume. Borrow mercure's `result.json`-archived-verbatim idea for per-job diagnostics (MIT) [44].
9. **`core/plugins/protocol.py` — no MAP refactor.** MAP/Holoscan requires CUDA 13 and nvcr.io base images [4][41][63], incompatible with D-014; the in-process protocol stays, and any MAP or mercure module is a wrapper around the CLI (§5c, §5d).
10. **`core/ingest/service.py` + `core/dicom` — independent de-id cross-check.** Orthanc's built-in `/anonymize` implements PS3.15 2023b [54]; running it on Orthanc A output in CI and diffing against our allowlist result gives a second, independently maintained implementation for the "fails closed" verifier. For burned-in text, CTP's `DicomPixelAnonymizer` (region-blanking scripts) [66] and the OCR microservices [67][68] are the reference designs — ideas only (RSNA licence / unasserted).
11. **`docs/PLAN.md`, `services/review-ui` (D-015) — viewer choice confirmed.** OHIF (MIT, embeddable, SR/SEG extensions) [15][69] is the only surveyed viewer that fits an iframe; Weasis (EPL/Apache, Java) is the right *second* viewer for checking SR/GSPS rendering [18]. Keep SC overlays as the AIR-acknowledged "fall-through rendering" for dumb displays [25 issue 3].
12. **Results API — add a FHIR `DiagnosticReport` bundle behind D-021.** Cheap (§5e) and gives an HL7 route for sites without DICOMweb consumers; gate it exactly like SR/SC in `core/serving/routes/review.py`.
13. **Standards watch (no code).** Track AIRA for the feedback loop (principle 8) [26], Sup 219 JSON-SR as a future replacement for our ad-hoc `findings` JSON [29], and Sup 251 as the eventual standard for the `POST /v1/infer` request/health/status shape [31].

### 6.1 What was verified how, and what was not

* **Verified against the GitHub REST API (2026-09-21):** licence SPDX id, star count, last push, latest release object for every repository in §0 [1][5][6][8][10][15][17][18][19][20][21][22][36][37][50][51][55][56][57][67][68][72]. Where a rendered web page reported a different *year* for the same release (mercure v0.4.1, dcm4chee 5.35.1, Weasis 4.7.3, Holoscan 4.6.0, MONAI IG 0.5.2), the API value is used.
* **Verified from primary source text:** IHE AIR and AIW-I (PDF text extracted locally) [25][27]; DICOM Sup 224 and Sup 251 (PDF text) [30][31]; OHIF / Cornerstone3D adapter source files [69][70][73]–[81]; highdicom, MONAI operator and packaging sources [38][39][41][91]; Orthanc licensing and Python plugin pages [12][13]; XNAT LICENSE.txt [19]; CTP source header [37].
* **Secondary or unverified (flagged inline):** Kaapana ingest mechanics (docs pages 404'd; search snippets only) [53]; the XNAT ML deprecation statement [62]; the Clara Deploy → MONAI Deploy transition blog [64]; dcm4chee's tri-licence beyond the README's MPL 1.1 [17]; behaviour of the pinned `ohif/app:v3.9.2` (source read is master 3.14.0-beta.35) [16]; whether SC Image Storage is in OHIF's default stack-handler list [81]; the display path OHIF takes for our region-less "unable to assess" SR; the existence of a tool named "pixel-checker".
* **Not reviewed:** the AIRA supplement body [26]; Sup 219 frozen-draft body [29]; Weasis's SR rendering in practice [18].

### Verdict

**Keep the D-002/D-004 design; complement, do not replace.** None of the surveyed platforms provides the three things that define this project — de-identify-before-storage as a physical boundary with a fail-closed verifier, server-side reveal gating, and typed findings that cannot exist without a calibrated probability and a localisation — and each would add heavier infrastructure (Kubernetes/Airflow, .NET + RabbitMQ + MongoDB + Argo, WildFly + LDAP) without supplying them. The permissively licensed pieces we already use (Orthanc via REST, OHIF, highdicom, pydicom) are current and active as of 2026-09-21, and the two adoptable ideas from elsewhere are small: mercure's folder/`task.json`/`result.json` contract as an optional out-of-process plugin adapter, and a retried export job modelled on every orchestrator's dispatcher. The substantive work the survey does point to is standards alignment inside `writer.py` (Comprehensive 3D SR, contributing-equipment provenance, AIR negative-finding pattern, box semantics) plus an OHIF hydration test against the pinned image — all additive, all within D-005's highdicom choice.

## References (all accessed 2026-09-21)

1. Project-MONAI/monai-deploy-app-sdk — https://github.com/Project-MONAI/monai-deploy-app-sdk (licence, stars, last push via GitHub API)
2. monai-deploy-app-sdk on PyPI — https://pypi.org/project/monai-deploy-app-sdk/
3. monai-deploy-app-sdk releases and tags — https://github.com/Project-MONAI/monai-deploy-app-sdk/releases ; https://api.github.com/repos/Project-MONAI/monai-deploy-app-sdk/tags
4. monai-deploy-app-sdk README (prerequisites) — https://github.com/Project-MONAI/monai-deploy-app-sdk/blob/main/README.md
5. Project-MONAI/monai-deploy-informatics-gateway — https://github.com/Project-MONAI/monai-deploy-informatics-gateway
6. Project-MONAI/monai-deploy-workflow-manager — https://github.com/Project-MONAI/monai-deploy-workflow-manager ; releases https://github.com/Project-MONAI/monai-deploy-workflow-manager/releases
7. MONAI Deploy Express README — https://github.com/Project-MONAI/monai-deploy/tree/main/deploy/monai-deploy-express
8. mercure-imaging/mercure — https://github.com/mercure-imaging/mercure
9. mercure releases / tags — https://github.com/mercure-imaging/mercure/releases ; https://github.com/mercure-imaging/mercure/tags
10. kaapana/kaapana — https://github.com/kaapana/kaapana
11. Kaapana releases — https://github.com/kaapana/kaapana/releases
12. Orthanc Book, "Licensing" FAQ — https://orthanc.uclouvain.be/book/faq/licensing.html
13. Orthanc Book, "Python plugin" — https://orthanc.uclouvain.be/book/plugins/python.html
14. Orthanc NEWS file — https://orthanc.uclouvain.be/hg/orthanc/raw-file/tip/NEWS
15. OHIF/Viewers — https://github.com/OHIF/Viewers ; releases https://github.com/OHIF/Viewers/releases
16. OHIF platform/app/package.json (master version) — https://github.com/OHIF/Viewers/blob/master/platform/app/package.json
17. dcm4che/dcm4chee-arc-light (README licence section) — https://github.com/dcm4che/dcm4chee-arc-light ; releases https://github.com/dcm4che/dcm4chee-arc-light/releases
18. nroduit/Weasis — https://github.com/nroduit/Weasis ; releases https://github.com/nroduit/Weasis/releases
19. NrgXnat/xnat (LICENSE.txt, BSD 2-clause) — https://github.com/NrgXnat/xnat ; https://github.com/NrgXnat/xnat/blob/main/LICENSE.txt
20. NrgXnat/container-service — https://github.com/NrgXnat/container-service
21. nvidia-holoscan/holoscan-sdk — https://github.com/nvidia-holoscan/holoscan-sdk
22. herrmannlab/highdicom — https://github.com/herrmannlab/highdicom
23. highdicom on PyPI — https://pypi.org/project/highdicom/
24. pydicom on PyPI — https://pypi.org/project/pydicom/ (GitHub "latest release" object is the v2.4.5 backport; tag v3.0.2 exists: https://api.github.com/repos/pydicom/pydicom/tags)
25. IHE RAD TF Supplement, AI Results (AIR), Rev 1.3 TI, 2025-08-08 — https://www.ihe.net/uploadedFiles/Documents/Radiology/IHE_RAD_Suppl_AIR.pdf (text extracted locally with pypdf; section numbers as cited)
26. IHE RAD TF Supplement, AI Result Assessment for Imaging (AIRA), TI 2025-06-12 — https://www.ihe.net/uploadedFiles/Documents/Radiology/IHE_RAD_Suppl_AIRA.pdf (not read in full)
27. IHE RAD TF Supplement, AI Workflow for Imaging (AIW-I), Rev 1.1 TI, 2020-08-06 — https://www.ihe.net/uploadedFiles/Documents/Radiology/IHE_RAD_Suppl_AIW-I.pdf
28. DICOM WG-23 page — https://www.dicomstandard.org/activity/wgs/wg-23
29. DICOM Sup 219 supplement page — https://www.dicomstandard.org/news/supplements/view/json-representation-of-dicom-structured-reports ; frozen draft https://dicom.nema.org/medical/dicom/supps/Frozen/sup219_fz_14_JSONSR.pdf
30. DICOM Sup 224 draft v4 (2021-08-03) — https://dicom.nema.org/medical/dicom/Supps/Drafts/sup224_04_Service_Discovery_and_Control.pdf
31. DICOM Sup 251 "Application Request", v5 2026-07-14 — https://www.dicomstandard.org/news-dir/current/docs/sups/sup251.pdf
32. HL7 FHIR R5 Observation — https://hl7.org/fhir/R5/observation.html
33. HL7 FHIR R5 DiagnosticReport — https://hl7.org/fhir/R5/diagnosticreport.html
34. HL7 FHIR R5 ImagingStudy — https://hl7.org/fhir/R5/imagingstudy.html
35. HL7 FHIR R5 ImagingSelection — https://hl7.org/fhir/R5/imagingselection.html
36. derekmerck/diana — https://github.com/derekmerck/diana
37. johnperry/CTP (licence header in source) — https://github.com/johnperry/CTP ; https://github.com/johnperry/CTP/blob/master/source/java/org/rsna/ctp/ClinicalTrialProcessor.java
38. monai-deploy-app-sdk operators directory — https://github.com/Project-MONAI/monai-deploy-app-sdk/tree/main/monai/deploy/operators
39. DICOMTextSRWriterOperator source — https://github.com/Project-MONAI/monai-deploy-app-sdk/blob/main/monai/deploy/operators/dicom_text_sr_writer_operator.py
40. MONAI Application Package specification — https://github.com/Project-MONAI/monai-deploy/blob/main/guidelines/monai-application-package.md
41. MONAI Deploy "Packaging app" guide — https://github.com/Project-MONAI/monai-deploy-app-sdk/blob/main/docs/source/developing_with_sdk/packaging_app.md
42. Informatics Gateway releases — https://github.com/Project-MONAI/monai-deploy-informatics-gateway/releases
43. MONAI Deploy Express VERSION — https://raw.githubusercontent.com/Project-MONAI/monai-deploy/main/deploy/monai-deploy-express/VERSION
44. mercure docs, "Module development" — https://mercure-imaging.org/docs/modules.html
45. mercure docs, "Modules" (usage) — https://mercure-imaging.org/docs/usage/modules.html
46. mercure docs index — https://mercure-imaging.org/docs/index.html
47. mercure docs, "Targets" — https://mercure-imaging.org/docs/usage/targets.html
48. mercure docs, "Existing modules" — https://mercure-imaging.org/docs/module_list.html
49. mercure docs, "Anonymizer" — https://mercure-imaging.org/docs/anonymizer.html
50. mercure-imaging/mercure-anonymizer (GPL-3.0 per GitHub API) — https://github.com/mercure-imaging/mercure-anonymizer
51. mercure-imaging/dcmannotate (GPL-3.0 per GitHub API) — https://github.com/mercure-imaging/dcmannotate
52. mercure docs, "DCMAnnotate" — https://mercure-imaging.org/docs/dcmannotate.html
53. Kaapana docs (search snippets; pages 404 on fetch): "How to send images to the platform" https://kaapana.readthedocs.io/en/latest/faq/how_to_send_images_to_the_platform.html ; changelog https://kaapana.readthedocs.io/en/latest/releases.html
54. Orthanc Book, "Anonymization and modification" — https://orthanc.uclouvain.be/book/users/anonymization.html
55. jodogne/orthanc-mammography — https://github.com/jodogne/orthanc-mammography
56. NeuroSainteAnne/OrthancAI — https://github.com/NeuroSainteAnne/OrthancAI
57. ImagingInformatics/orthanc-ai-orchestrator — https://github.com/ImagingInformatics/orthanc-ai-orchestrator
58. Jesus R. et al., "A Plugin-Based Architecture for Integrating AI Services in an Open-Source PACS", J Imaging Inform Med, 2026-02-19, PMID 41714575 — https://pubmed.ncbi.nlm.nih.gov/41714575/ (abstract via NCBI E-utilities)
59. dcm4chee-arc-light release notes 5.34.3 / 5.35.x — https://github.com/dcm4che/dcm4chee-arc-light/releases
60. Weasis licence files — https://github.com/nroduit/Weasis/blob/master/LICENSE.EPL-2.0 ; https://github.com/nroduit/Weasis/blob/master/LICENSE.Apache-2.0
61. XNAT Container Service documentation — https://wiki.xnat.org/container-service
62. XNAT ML documentation (via search) — https://wiki.xnat.org/ml/ ; https://wiki.xnat.org/ml/components-of-the-xnat-ml-release
63. Holoscan SDK releases — https://github.com/nvidia-holoscan/holoscan-sdk/releases
64. NVIDIA Technical Blog, "MONAI Bridges the Gap from Innovative Research to Clinical Production" (Clara Deploy → Clara Holoscan; MONAI Deploy WG) — https://developer.nvidia.com/blog/monai-bridges-the-gap-from-innovative-research-to-clinical-production/ (via search; secondary)
65. NVIDIA Clara Deploy SDK 0.8.1 archived docs — https://docs.nvidia.com/clara-holoscan/archive/clara-deploy-0.8.1/index.html
66. RSNA MIRC wiki, "CTP – The RSNA Clinical Trial Processor" — https://mircwiki.rsna.org/index.php?title=CTP-The_RSNA_Clinical_Trial_Processor
67. pydicom/dicom-cleaner — https://github.com/pydicom/dicom-cleaner
68. nroduit/image-ocr-identifier — https://github.com/nroduit/image-ocr-identifier
69. OHIF cornerstone-dicom-sr `getSopClassHandlerModule.ts` — https://github.com/OHIF/Viewers/blob/master/extensions/cornerstone-dicom-sr/src/getSopClassHandlerModule.ts
70. OHIF cornerstone-dicom-sr `utils/getRenderableData.ts` — https://github.com/OHIF/Viewers/blob/master/extensions/cornerstone-dicom-sr/src/utils/getRenderableData.ts
71. OHIF cornerstone-dicom-sr `package.json` (dcmjs 0.52.0, @cornerstonejs/adapters 5.10.3) — https://github.com/OHIF/Viewers/blob/master/extensions/cornerstone-dicom-sr/package.json
72. dcmjs-org/dcmjs — https://github.com/dcmjs-org/dcmjs
73. OHIF cornerstone-dicom-sr `utils/isRehydratable.ts` — https://github.com/OHIF/Viewers/blob/master/extensions/cornerstone-dicom-sr/src/utils/isRehydratable.ts
74. Cornerstone3D adapters `isValidCornerstoneTrackingIdentifier.ts` — https://github.com/cornerstonejs/cornerstone3D/blob/main/packages/adapters/src/adapters/Cornerstone3D/isValidCornerstoneTrackingIdentifier.ts
75. Cornerstone3D adapters `RectangleROI.ts` — https://github.com/cornerstonejs/cornerstone3D/blob/main/packages/adapters/src/adapters/Cornerstone3D/RectangleROI.ts
76. Cornerstone3D adapters `PlanarFreehandROI.ts` — https://github.com/cornerstonejs/cornerstone3D/blob/main/packages/adapters/src/adapters/Cornerstone3D/PlanarFreehandROI.ts
77. Cornerstone3D adapters directory — https://github.com/cornerstonejs/cornerstone3D/tree/main/packages/adapters/src/adapters/Cornerstone3D
78. OHIF cornerstone-dicom-sr `utils/hydrateStructuredReport.ts` — https://github.com/OHIF/Viewers/blob/master/extensions/cornerstone-dicom-sr/src/utils/hydrateStructuredReport.ts
79. OHIF cornerstone-dicom-sr `components/` — https://github.com/OHIF/Viewers/tree/master/extensions/cornerstone-dicom-sr/src/components
80. OHIF `platform/core/src/utils/sopClassDictionary.js` — https://github.com/OHIF/Viewers/blob/master/platform/core/src/utils/sopClassDictionary.js
81. OHIF `extensions/default/src/getSopClassHandlerModule.js` — https://github.com/OHIF/Viewers/blob/master/extensions/default/src/getSopClassHandlerModule.js
82. IHE Radiology profiles index — https://profiles.ihe.net/RAD/index.html
83. Canada Health Infoway note on Sup 219 trial use — https://infocentral.infoway-inforoute.ca/en/resources/docs/imagingdocs/dicom/dicom-2020/3175-supplement-219-trial-use-implementation-json-representation-of-dicom-structured-reports
84. DICOM Sup 230 "Update BCP Secure Communications Profiles" — https://www.dicomstandard.org/news-dir/progress/docs/sups/sup230.pdf
85. DICOM Sup 236 "Waveform Presentation State" slides (WG-32) — https://www.dicomstandard.org/news-dir/current/docs/sups/sup236-slides.pdf
86. DICOM PS3.3 A.35.4 Key Object Selection Document IOD — https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_A.35.4.html
87. DICOM PS3.16 TID 2010 Key Object Selection — https://dicom.nema.org/medical/dicom/current/output/chtml/part16/sect_TID_2010.html
88. HL7 FHIR R4 DiagnosticReport (for the R4 `imagingStudy` element) — https://hl7.org/fhir/R4/diagnosticreport.html
89. highdicom user guide, "The TID 1500 Measurement Report" — https://highdicom.readthedocs.io/en/latest/tid1500.html
90. highdicom user guide, presentation states (page "under construction") — https://highdicom.readthedocs.io/en/latest/pr.html
91. highdicom `src/highdicom/pr/__init__.py` — https://github.com/herrmannlab/highdicom/blob/master/src/highdicom/pr/__init__.py
92. pydicom reference, `pydicom.sr` — https://pydicom.github.io/pydicom/stable/reference/sr.html
