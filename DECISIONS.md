# Decisions log

Format: **D-nnn — title** · date · status. Context / decision / consequences. Newest at the bottom.

---

**D-001 — Research-only prototype** · 2026-09-21 · accepted
Every clinician-visible surface carries "For research use only. Not for diagnostic use." (constant `RESEARCH_USE_LABEL`). All datasets chosen are usable under research terms; commercial use would require re-training on `commercial_ok` datasets only.

**D-002 — Two-Orthanc topology** · 2026-09-21 · accepted
Orthanc A simulates the hospital PACS (identifiable). The platform pulls, de-identifies, and stores into its own encrypted blob store + Postgres, and pushes de-identified studies/results to Orthanc B, which OHIF views. Makes "de-identify before storage" a physical boundary that integration tests can verify. Cost: one extra container.

**D-003 — Modality plugin protocol with `frames × H × W` input** · 2026-09-21 · accepted
`core/` never imports `modalities/`. `ImagingInput` always carries a frame axis; CXR is `frames=1`. The quality gate is part of the protocol and runs before inference for every modality. This is what keeps ultrasound (cine loops, multi-frame DICOM, per-probe preprocessing, mandatory quality gate) from needing a core refactor.

**D-004 — Postgres-backed job queue** · 2026-09-21 · accepted
A `jobs` table + polling worker instead of Redis/Celery. One fewer service; volumes are tiny. `JobQueue` is an interface so it can be swapped.

**D-005 — `highdicom` for DICOM SR (TID 1500) and Secondary Capture output** · 2026-09-21 · accepted
Hand-rolling SR with pydicom is error-prone; highdicom is maintained, validates content trees, and aligns with IHE AI Results.

**D-006 — File-based, content-hashed model registry mirrored in DB** · 2026-09-21 · accepted
`model_id = sha256(weights ‖ config ‖ split manifest)[:16]`. Recorded in `model_versions`. Every prediction stores `model_version_id` and `input_sha256`. MLflow deferred.

**D-007 — Train from ImageNet init, never from public CXR-pretrained weights** · 2026-09-21 · accepted
torchxrayvision-style weights were trained on NIH/CheXpert/MIMIC/PadChest, which would contaminate our external test sets. Costs some accuracy at small scale; buys an honest external evaluation.

**D-008 — Patient-level hashed splits; external test locked by manifest hash** · 2026-09-21 · accepted
Tier = `sha256(salt ‖ source ‖ patient_id) mod 100` → train 80 / val 10 / internal-test 10 per source. External = VinDr-CXR + PadChest manual subset (+ BRAX optional). External manifest hash is recorded here at lock time and asserted by CI. NIH-derived re-releases (SIIM-ACR, RSNA, ChestX-Det) are never used as external.

**D-009 — App-level AES-256-GCM blob encryption** · 2026-09-21 · accepted
`EncryptedBlobStore` wraps any `BlobStore`. Key from `SR_BLOB_ENCRYPTION_KEY` env only. Disk-level encryption is additionally documented for deployment; app-level is what CI can test.

**D-010 — Subgroup reporting is mandatory in the eval report schema** · 2026-09-21 · accepted
`EvalReport` requires per-subgroup tables for source, vendor, age band, sex, view. Missing metadata yields an explicit "unavailable" row, never a silent drop. Bootstrap 95% CIs; `low_n` flag when positives < 100.

**D-011 — Dataset registry carries `license` and `commercial_ok`** · 2026-09-21 · accepted
`datasets/registry.yaml` is the only place datasets are declared. License text is re-verified and dated at download time.

**D-012 — Target finding "nodule/mass" is merged** · 2026-09-21 · accepted
NIH/PadChest split nodule (≤3 cm) from mass; CheXpert/MIMIC/VinDr merge them. A merged `nodule_mass` target is the only harmonizable definition. UI label: "Pulmonary nodule/mass".

**D-013 — PhysioNet-credentialed datasets are supported but not required** · 2026-09-21 · accepted
Download scripts for MIMIC-CXR / VinDr-CXR / BRAX read credentials from env and skip cleanly if absent. External test #1 is VinDr-CXR when available, otherwise PadChest-manual + CANDID-PTX. Nothing in the pipeline hard-depends on a credentialed set.

**D-014 — Local training target: Apple Silicon (MPS) at 320 px** · 2026-09-21 · accepted
Baseline is DenseNet-121 (timm, ImageNet init) at 320×320, batch 16, which fits an M1 Pro. Docker images are CPU-only; CUDA is an override via `SR_DEVICE`.

**D-015 — Review UI is a thin React app that embeds OHIF** · 2026-09-21 · accepted
Reveal-gating, priors panel, and feedback live in our app; images are rendered by OHIF (iframe pointed at Orthanc B via DICOMweb). An OHIF mode/extension is the later "inside the viewer" path; the results API is designed so that swap is additive.

**D-016 — Local auth (username/password + roles) with an `AuthProvider` seam** · 2026-09-21 · accepted
v0 uses argon2-hashed local users, JWT sessions, roles `clinician` / `admin`. Every study view and reveal is an `audit_events` row. OIDC/SSO is a second `AuthProvider` implementation, not a rewrite.

**D-017 — Uncertain (CheXpert "-1") labels are masked out of the loss** · 2026-09-21 · accepted
Not forced to 0 or 1. Reported as a sensitivity analysis in the eval report.

**D-018 — Homebrew Orthanc/Postgres for local integration when Docker is absent** · 2026-09-21 · accepted
Docker Compose is the canonical deployment; on the development Mac (no Docker) the same configs are run via `scripts/local_services.sh`. Integration tests skip when services are unreachable.
