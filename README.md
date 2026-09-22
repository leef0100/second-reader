# Second Reader

> **For research use only. Not for diagnostic use.**
> Research prototype. Not a cleared or approved medical device.

An AI "second reader" for medical imaging that helps clinicians catch missed findings. Chest X-ray (frontal PA/AP) is the first modality: **pneumothorax, pleural effusion, pulmonary nodule/mass**. It fits into a PACS workflow, shows priors side by side, explains itself with heat-maps, says when it is uncertain, and never makes the decision itself.

Read `CLAUDE.md` for the principles, `DECISIONS.md` for why things are the way they are, `docs/PLAN.md` for the architecture, `docs/INTENDED_USE.md` and `docs/REGULATORY_NOTES.md` before drawing any conclusions from outputs.

## What is here

```
second-reader/
├── src/secondreader/core/        modality-agnostic platform
│   ├── dicom/                    de-identification (allowlist profile, HMAC pseudonyms, verifier), rendering
│   ├── ingest/                   raw DICOM → de-id → verify → encrypt → index → queue
│   ├── storage/                  AES-256-GCM blob store
│   ├── db/                       SQLAlchemy schema (+ alembic/), prior-study linkage
│   ├── plugins/                  ModalityPlugin protocol, registry, loader
│   ├── results/                  DICOM SR (TID 1500) + Secondary Capture + JSON writer
│   ├── serving/                  FastAPI app, routers, job worker, inference runner
│   ├── eval/                     metrics, calibration, subgroup report
│   ├── monitoring/               drift (PSI) + feedback summary
│   ├── security/, audit/, pacs/  local auth, audit log, Orthanc client
├── src/secondreader/modalities/cxr/   the chest X-ray plugin: preprocess, gate, model, train, evaluate, saliency, adapters
├── services/review-ui/           clinician review UI (React) — read-first / reveal, priors, feedback
├── docker/                       Orthanc A plugin, OHIF config, gateway nginx, API image
├── configs/                      de-id profile, experiment configs (content-hashed)
├── datasets/                     registry.yaml (licences!), split manifests (hashes only)
├── docs/                         PLAN, INTENDED_USE, REGULATORY_NOTES, reports/
└── tests/                        unit (synthetic DICOM only) + integration (live Orthanc/Postgres)
```

## Quick start (local, Docker Compose)

Requirements: Docker (Compose v2), Python 3.12, [`uv`](https://docs.astral.sh/uv/), Node 20 (only to develop the UI).

```bash
git clone <this repo> second-reader && cd second-reader
cp .env.example .env            # then set real secrets (see below)
uv sync --extra dev             # Python env
make check                      # ruff + mypy + unit tests (synthetic data only)
```

Generate secrets (never commit `.env`):

```bash
python3 -c "import secrets; print('SR_PSEUDONYM_SECRET=' + secrets.token_hex(32))"
python3 -c "import secrets; print('SR_BLOB_ENCRYPTION_KEY=' + secrets.token_hex(32))"
python3 -c "import secrets; print('SR_AUTH_SECRET=' + secrets.token_hex(32))"
python3 -c "import secrets; print('SR_WEBHOOK_TOKEN=' + secrets.token_hex(16))"
echo "SR_BOOTSTRAP_ADMIN_PASSWORD=<choose one, >=10 chars>"
```

Train the synthetic smoke model (the only model in the repo — it proves the pipeline, nothing else):

```bash
uv run secondreader cxr synth --out data/synthetic
uv run secondreader cxr split data/synthetic/synth-a/manifest.csv data/synthetic/synth-b/manifest.csv --version smoke-v1
uv run secondreader cxr train --config configs/experiments/cxr_smoke.yaml --splits datasets/splits/smoke-v1
uv run secondreader cxr eval <model_version_id> --tier external --splits datasets/splits/smoke-v1
echo <model_version_id> > models/ACTIVE_MODEL_ID
```

Bring the stack up:

```bash
docker compose up -d --build
```

| URL | What |
|---|---|
| http://localhost:3000/review/ | Clinician review UI (login: `admin` / `SR_BOOTSTRAP_ADMIN_PASSWORD`) |
| http://localhost:3000/ | OHIF viewer over Orthanc B (de-identified archive) |
| http://localhost:3000/api/docs | API (OpenAPI) |
| http://localhost:8042 | Orthanc A — simulated hospital PACS (identifiable). DICOM AET `ORTHANC_A`, port 4242 |
| http://localhost:8043 | Orthanc B — platform archive (de-identified only) |

Send a study to the "hospital PACS" and watch it flow:

```bash
# any DICOM (or a synthetic one: uv run python -c "from tests.fixtures.synthetic import *; open('x.dcm','wb').write(make_bytes(cxr_spec()))")
curl -u orthanc:orthanc -X POST http://localhost:8042/instances --data-binary @x.dcm -H 'Content-Type: application/dicom'
docker compose logs -f api worker      # ingested → prediction → pushed to Orthanc B
```

Orthanc A's Python plugin notifies the platform when a study becomes stable; the platform pulls it, de-identifies it (fails closed if the verifier finds anything), stores it encrypted, queues inference, and pushes the de-identified image + DICOM SR + Secondary Capture overlays to Orthanc B, where OHIF and the review UI show them.

Synchronous API alternative: `POST /api/v1/infer` (multipart `file`) with a bearer token from `POST /api/v1/auth/login`.

## Developing without Docker

```bash
scripts/dev_up.sh                                   # postgres + orthanc A/B via compose, migrations
uv run uvicorn secondreader.core.serving.app:app --reload --port 8000
uv run secondreader worker run                      # in another shell
cd services/review-ui && npm install && npm run dev # UI on :5173, proxies /api → :8000
```

`make test` runs unit tests (SQLite, synthetic DICOM only); `make test-int` runs the live Orthanc/Postgres integration tests (skipped when services are down).

## Real data

Nothing in this repository downloads or contains patient data. To train on public datasets:

1. Read `datasets/registry.yaml` — every dataset's licence and `commercial_ok` flag. Most radiologist-labelled sets are **research-only**. Verbatim terms live in `datasets/LICENSES/`; `datasets/ACQUISITION.md` lists what a human must do (PhysioNet credentialing, Stanford/BIMCV click-throughs) and what an agent can fetch unattended. Re-verify the licence text at download time.
2. Fetch NIH ChestX-ray14 (no credentials, ~45 GB): `nohup scripts/download_nih_cxr14.sh > data/raw/nih_cxr14/download.log 2>&1 &` — resumable, verifies sizes + MD5s, extracts to `data/raw/nih_cxr14/images/`.
3. Build manifests: `uv run secondreader cxr manifest nih data/raw/nih_cxr14 --out datasets/manifests/nih_cxr14.csv` (also `chexpert`, `vindr`, `padchest`). Manifests are git-ignored.
4. Split (patient-level, hashed; external sets locked): `uv run secondreader cxr split datasets/manifests/*.csv --version v1 --external vindr_cxr,padchest`. Record the printed manifest hash in `DECISIONS.md`.
5. Build the preprocessing cache once per split version (decodes every image once, ~30 min for 112k NIH PNGs on an M1 Pro; the cache is refused if the manifest or preprocess config changes): `uv run secondreader cxr cache --splits datasets/splits/v1 --size 320 --out data/cache/v1`. Add `--limit 500` for a smoke cache.
6. Train with `configs/experiments/cxr_baseline.yaml --cache data/cache/v1` and evaluate on `--tier external`. A checkpoint is written every epoch and on Ctrl-C; rerun the same command with `--resume` to continue. Per-epoch metrics land in `models/<id>/metrics.jsonl`. The report (`runs/<model>/external/report.md`) has overall and per-subgroup metrics with CIs and reliability diagrams; it refuses to render without subgroup tables.
7. Throughput on this machine (loader and DenseNet forward/backward, with the exact commands) is in `docs/reports/throughput_m1pro.md`; `uv run secondreader cxr bench loader|model` reproduces it.

## Adding a modality (e.g. ultrasound)

Implement `secondreader.core.plugins.ModalityPlugin` in `src/secondreader/modalities/<id>/`, register it under the `secondreader.plugins` entry-point group, keep your preprocessing/model/eval/intended-use inside the package. Inputs arrive as `frames × H × W` with per-frame metadata; the quality gate runs before `predict` and may return `unable_to_assess`. Nothing in `core` needs to change.

## Security notes for anyone deploying this beyond a laptop

Terminate TLS at the gateway; put Orthanc B and the API behind it; replace basic-auth injection in `docker/gateway/nginx.conf` with real identity propagation; keep `SR_*` secrets in a secrets manager; enable disk encryption in addition to the app-level blob encryption; review `docs/REGULATORY_NOTES.md` — this is not a product.
