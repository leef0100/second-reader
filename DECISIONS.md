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

**D-019 — OOD threshold is a policy knob, reported as a metric** · 2026-09-21 · accepted
The Mahalanobis gate threshold is the `gate.ood_percentile` (default 99th) percentile of internal-validation distances. On a shifted source this deliberately rejects more inputs ("unable to assess") rather than guessing; the rejection rate per source is printed in every evaluation report and tracked by the drift monitor. Ledoit-Wolf shrinkage keeps the covariance well-conditioned when n < D.

**D-020 — Model version id includes calibration** · 2026-09-21 · accepted
`model_version_id = sha256(weights ‖ config ‖ split manifest ‖ calibration)[:16]`. Temperatures and thresholds change what a clinician sees, so re-calibrating the same weights is a new version.

**D-021 — Reveal gating is enforced server-side** · 2026-09-21 · accepted
Findings, heatmaps, SR and SC are only served after a `ReviewSession.revealed_at` exists for that clinician and study; in `read_first` mode reveal requires a recorded pre-read impression. The UI cannot bypass this because the API refuses (403/409). `immediate` mode exists for bench use and is audited as such.

**D-022 — SR content: TID 1500 with per-finding probability, presence and box** · 2026-09-21 · accepted
Each finding is a measurement group with finding type (SNOMED CT), a numeric "calibrated probability" measurement (private code 99SRD), a Presence qualitative evaluation (SCT Present/Absent), an algorithm identification carrying the model version, and a POLYLINE image region for the box. Gated-out inputs produce an SR whose only group says "Unable to assess" (ContentLabel = UNABLE TO ASSESS). `VerificationFlag` is always UNVERIFIED — a clinician verifies, never the system. The TID 1600 image library is omitted because it requires Type-2C CR attributes many exports lack.

**D-023 — Secondary Capture overlays carry BurnedInAnnotation=YES** · 2026-09-21 · accepted
The overlay burns in the finding label, probability and the research-use banner (no PHI — inputs are de-identified). Setting the flag honestly means a downstream de-identifier treats them as pixel-annotated objects.

<<<<<<< HEAD
**D-024 — Real-data acquisition: NIH ChestX-ray14 first; licence text is recorded verbatim at download time** · 2026-09-21 · accepted
NIH ChestX-ray14 is the first real dataset on disk (the only ≥100k-image CXR set with no use restriction and no credentialing). Its 12 archive URLs come from NIH's own `batch_download_zips.py`, sizes were verified against the server and MD5s against NIH's FAQ; the metadata CSVs cannot be fetched by URL (Box blocks direct links) and were saved once through the Box UI, with SHA-256s pinned in `scripts/download_nih_cxr14.sh`. NIH has replaced `Data_Entry_2017.csv` with `Data_Entry_2017_v2020.csv` (`Patient Sex` column, ages corrected); the adapter accepts both and prefers v2020. Every dataset in `registry.yaml` now carries `license_file` (verbatim terms under `datasets/LICENSES/`) and `license_verified`; anything that needs a human (click-through, DUA, CITI) is listed in `datasets/ACQUISITION.md`. Manifests (`datasets/manifests/*.csv`) enumerate local paths and are git-ignored like split CSVs.
=======
**D-024 — Preprocessing cache as 8-bit letterboxed frames, keyed on manifest + config** · 2026-09-21 · accepted
`secondreader cxr cache` decodes every manifest image once (multiprocessing) and stores the windowed, letterboxed `[0,1]` frame as uint8 `(N, S, S)` in one memory-mapped `.npy` per tier, with `index.csv` (image_id → row + `input_statistics`) and `cache.json` carrying `sha256(tier manifest sha256 ‖ PreprocessConfig ‖ format)`. `FrameCache.open` refuses a cache whose key does not match the current manifest/config, so a re-split or a changed window can never be trained on silently. The dataset applies ImageNet normalization and augmentation on the cached tensor; the only difference from the decode path is 8-bit quantization (≤ 0.5/255 in `[0,1]`), which is below the source precision of NIH PNGs and CheXpert JPEGs (both 8-bit) — a test asserts exactly that bound. Serving never uses the cache: the plugin still decodes fresh pixels. Rows the cache lacks (`--limit` smoke caches, decode failures) fall back to decoding, with a warning giving the count.

**D-025 — Checkpoint every epoch and on Ctrl-C; resume continues the interrupted epoch** · 2026-09-21 · accepted
`<registry>/pending/checkpoint.pt` holds model, optimizer, OneCycle scheduler, early-stopping state, history, and the config hash (a different config refuses to resume). A KeyboardInterrupt saves the checkpoint with `step_in_epoch`; on `--resume` the seeded per-epoch permutation (`ResumableSampler`) skips the batches already consumed, so scheduler step counts stay consistent with `total_steps`. Augmentation RNG is not restored: a resumed run is equivalent, not bit-identical. Epoch metrics are appended to `pending/metrics.jsonl` and moved into the bundle.

**D-026 — Positive-class weighting is opt-in (`train.pos_weight: auto|list|null`)** · 2026-09-21 · accepted
NIH prevalence is low (pneumothorax ≈ 5 %). `auto` computes `n_neg / n_pos` per finding from the training manifest (labelled 0/1 rows only — uncertain labels stay masked per D-017) and passes it as `pos_weight` to the masked BCE. Default is off so the baseline's behaviour is unchanged; the weights used are recorded in the model card. Weighting changes logit scale, which is exactly what post-hoc temperature scaling on validation absorbs (D-020), so calibration is unaffected in kind.
>>>>>>> feat/fast-training
