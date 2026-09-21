# Second Reader — working rules for Claude and humans

**This is a research prototype. It is NOT a cleared medical device.**
Every output, report, log line that a clinician sees, and UI screen must carry:
`For research use only. Not for diagnostic use.`
The constant lives in `secondreader.core.constants.RESEARCH_USE_LABEL`; import it, never retype it.

## What this is
An AI "second reader" for medical imaging (chest X-ray first; ultrasound later as a plugin).
It helps clinicians catch missed findings. It supports their decision; it never makes it.

## Non-negotiable principles
1. **Generalization** — metrics are always reported per subgroup (source/site, vendor, age band, sex, view) with CIs, plus an external test set never used for training or tuning. An eval report without subgroup tables is invalid.
2. **Honest uncertainty** — outputs are calibrated probabilities. The quality/OOD gate runs before inference and returns `unable_to_assess` rather than guessing.
3. **Anti-automation-bias** — the review UI supports "read first, then reveal AI". Every finding is shown with its confidence and a heatmap/box. Never render a finding without both.
4. **Prior comparison** — studies are linked to a pseudonymized patient; priors are shown side by side.
5. **Standards-based** — DICOM in; DICOM SR + Secondary Capture + JSON out; Orthanc + OHIF for testing, no custom viewer.
6. **Privacy & security** — de-identify before storage (two-Orthanc boundary), encrypt at rest and in transit, audit who viewed what. No real data, no secrets in the repo. Ever.
7. **Traceability** — every model, split, and config is versioned by content hash; every prediction records model version + input hash.
8. **Monitoring** — log predictions and clinician agree/disagree; run drift checks on inputs and outputs.

## Architecture rules
- `src/secondreader/core/` is modality-agnostic and must never import from `modalities/`.
- Modalities implement `core.plugins.ModalityPlugin`. Inputs are always `frames × H × W` (CXR = 1 frame) so multi-frame/cine (ultrasound) needs no core change.
- Ingest → de-id → encrypted blob store + Postgres metadata → job → plugin gate → plugin model → results writer → Orthanc B / DB.
- Orthanc A = simulated hospital PACS (identifiable). Orthanc B = platform archive (de-identified only). Nothing identifiable crosses into platform storage.

## Working conventions
- Python 3.12, `uv`, `ruff`, `mypy --strict` on `core/`, `pytest`. Run `make check` before committing.
- Tests use synthetic DICOM only (`tests/fixtures/synthetic.py`). Fake PHI is fine; real PHI is never.
- Log decisions in `DECISIONS.md` (D-nnn). Update `docs/INTENDED_USE.md` when scope changes.
- Config is pydantic-validated YAML under `configs/`; runtime settings come from env (`SR_*`), see `.env.example`.
- Never add a dataset without an entry in `datasets/registry.yaml` including `license` and `commercial_ok`.
- Do not use CXR-pretrained public weights (e.g. torchxrayvision) — they contaminate external test sets (D-007).

## Phases
1 Plan → 2 Foundation → 3 Baseline model → 4 Serving + review → 5 Documentation. See `docs/PLAN.md`.
