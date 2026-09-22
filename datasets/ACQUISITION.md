# Dataset acquisition — what an agent can do vs. what a human must do

Status as of **2026-09-21**. Every dataset below has an entry in `registry.yaml`
(with `license_file`, `license_verified`, `commercial_ok`) and verbatim terms under
`LICENSES/`. `data/` and `datasets/manifests/*.csv` are git-ignored — nothing here is
ever committed. All data is **for research use only**.

## Summary

| Dataset | Role | Size | Credentialing | Agent can fetch unattended? | Status |
|---|---|---|---|---|---|
| NIH ChestX-ray14 | train | 45.1 GB | none | **yes** — `scripts/download_nih_cxr14.sh` | **downloading** (see below) |
| Kermany paediatric | ood_negative (gate should reject) | 8.4 GB (incl. OCT) | none (CC BY 4.0) | **yes** — `scripts/download_ood_sets.sh` | **downloading 2026-09-22** (zip verified; `chest_xray/` extracted, OCT skipped) |
| NLM TB Shenzhen + Montgomery | gate_eval (gate should accept) | 3.8 GB + 0.6 GB | none | **yes** — `scripts/download_ood_sets.sh` | **downloading 2026-09-22** |
| Open-i IU CXR | ood / report text | ~1–2 GB | none, but CC BY-NC-ND | yes once NLM's site is back up | **site down** (US federal funding-lapse notice); see `LICENSES/openi_iu_cxr.md` |
| CheXpert | train | 471 GB | Stanford AIMI account + click-through RUA | **no** — human registers, then agent runs AzCopy | waiting on human |
| PadChest | external_test | ~1.02 TB | BIMCV Google Form + RUA | **no** — human submits form, receives private link | waiting on human |
| MIMIC-CXR (+ -JPG) | train (optional) | 4.7 TB DICOM / 570 GB JPG | PhysioNet credentialing | **no** | waiting on human |
| VinDr-CXR | external_test | 200 GB DICOM | PhysioNet credentialing | **no** | waiting on human |
| BRAX | external_test | ~ 70 GB DICOM | PhysioNet credentialing | **no** | waiting on human |
| CANDID-PTX | localization_eval | ~ 20 GB | Figshare DUA (form) | **no** | waiting on human |

## Human-action checklist

### 1. PhysioNet credentialing (MIMIC-CXR, VinDr-CXR, BRAX) — one process unlocks all three
1. Create a PhysioNet account at <https://physionet.org/register/> (institutional e-mail, real name, supervisor/reference for students).
2. Complete the CITI Program course **"Data or Specimens Only Research"** (Massachusetts Institute of Technology affiliate, free): <https://physionet.org/about/citi-course/>. Upload the completion report (PDF) to your PhysioNet profile.
3. Apply for **credentialed access** in the profile ("Credentialing"). Review typically takes days to a few weeks.
4. For each dataset, open its page and **sign the Data Use Agreement** (PhysioNet Credentialed Health Data License 1.5.0 — research only, no redistribution, no re-identification, report breaches):
   * MIMIC-CXR <https://physionet.org/content/mimic-cxr/> and MIMIC-CXR-JPG <https://physionet.org/content/mimic-cxr-jpg/>
   * VinDr-CXR <https://physionet.org/content/vindr-cxr/>
   * BRAX <https://physionet.org/content/brax/>
5. Then an agent can download with your credentials via `wget -r -N -c -np --user <you> --ask-password https://physionet.org/files/<dataset>/<version>/` or the Google Cloud / AWS mirrors — **never put the password in a file or in the repo**; run it interactively or with a short-lived env var.

### 2. Stanford AIMI click-through (CheXpert)
1. Go to <https://stanfordaimi.azurewebsites.net/datasets/8cbd9ed4-2eb9-4565-affc-111cf4f7ebe2>, click **Login → Sign up** (name, e-mail, institution).
2. Accept the **Stanford University Dataset Research Use Agreement** (non-commercial research only; no redistribution). Copy the agreement text into `LICENSES/chexpert.md` (section "Licence text") — Stanford only shows it at registration.
3. Click **Export Dataset** → you receive an Azure Blob SAS URL. Hand the URL (it expires) to the agent: `azcopy copy "<SAS URL>" data/raw/chexpert/ --recursive`. 471 GB — plan ~1 day at 100 Mbit/s and 500 GB free disk (currently ~480 GB free after NIH: a smaller `CheXpert-v1.0-small` mirror is no longer offered by Stanford, so budget disk or stream-resize).

### 3. BIMCV click-through (PadChest)
1. Fill the request form linked from <https://bimcv.cipf.es/bimcv-projects/padchest/> (Google Form: name, institution, e-mail, purpose) and tick acceptance of the **PADCHEST Dataset Research Use Agreement** (`LICENSES/padchest.md`).
2. BIMCV e-mails a personal download link (do **not** share it outside the organisation — clause 6). Hand the link to the agent; ~1.02 TB (PNG16 + labels CSV). If disk is tight, download the labels CSV first and only the `MethodLabel == Physician` image folders we actually use for the external test.

### 4. Figshare DUA (CANDID-PTX)
Open <https://figshare.com/articles/dataset/CANDID-PTX/14173982>, request access and accept the custom non-commercial research agreement; the download button appears after approval.

### 5. Nothing needed (agent-only)
* **NIH ChestX-ray14** — running now: `scripts/download_nih_cxr14.sh` (resumable; log `data/raw/nih_cxr14/download.log`; PID in `download.pid`). Metadata CSVs were saved once through the Box UI because Box refuses direct links for them; their SHA-256s are pinned in the script.
* **Kermany paediatric + NLM TB sets** — `scripts/download_ood_sets.sh` (resumable; size-checked
  against the pinned zip size / each folder's `index.json`; log `data/raw/ood_download.log`, PID in
  `data/raw/ood_download.pid`). Started 2026-09-22. It re-reads the Mendeley licence field on the
  day of download and fetches the NLM README files (the licence terms) next to the data. Then build
  the manifests with `uv run python -m secondreader.modalities.cxr.cli_datasets manifest kermany
  data/raw/kermany_pediatric` and `… manifest nlm-tb data/raw/nlm_tb`; how they are used is in
  `datasets/OOD_EVAL_PLAN.md`.
* **Open-i IU CXR** — retry when <https://openi.nlm.nih.gov> stops serving the funding-lapse notice; then re-verify the FAQ licence text before use (CC BY-NC-ND 4.0 as last published → `commercial_ok: false`, `ood`/report-text role only).

## Rules that apply to every download
* Verify size and hash against the values recorded in the licence file; re-verify the licence text on the day of download and update `license_verified`.
* Keep raw data under `data/raw/<registry id>/`; build manifests with `uv run secondreader cxr manifest <id> data/raw/<id> --out datasets/manifests/<id>.csv`.
* Never commit images, manifests, links with tokens, or credentials. Never redistribute.
