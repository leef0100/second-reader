# CheXpert — licence and access

| | |
|---|---|
| Registry id | `chexpert` |
| Official source | Stanford AIMI Shared Datasets: <https://stanfordaimi.azurewebsites.net/datasets/8cbd9ed4-2eb9-4565-affc-111cf4f7ebe2> (project page: <https://stanfordmlgroup.github.io/competitions/chexpert/>; CheXpert Plus, with reports, lives on Redivis: <https://stanford.redivis.com/datasets/5yyj-1a9f6ap0x>) |
| Verified on | **2026-09-21** (AIMI dataset page read in a browser without logging in) |
| Licence | "Stanford University Dataset Research Use Agreement" (AIMI page, field `License`) |
| `commercial_ok` | **false** — research only, no redistribution |
| Credentialing | **Human click-through required.** The AIMI page shows the file list without login but says: "Please login in order to export this dataset." A free AIMI account (name, email, institution) and acceptance of the Research Use Agreement at registration are needed; the download is then an Azure Storage link (AzCopy). No CITI course, no signed DUA, no institutional approval. |
| Size | `CheXpert-v1.0.zip` **471.12 GB** (original resolution) + `train_cheXbert.csv` 23.13 MB + `train_visualCheXbert.csv` 29.86 MB + `README.md` 3.29 KB (file explorer on the AIMI page; created 2021-04-08, last modified 2021-07-27). The older "small" (~11 GB, 320-px) build is no longer listed on AIMI. |
| Contents | 224,316 chest radiographs of 65,240 patients, Stanford University Medical Center, Oct 2002 – Jul 2017; train labels from three NLP labelers (CheXpert, CheXbert, VisualCheXbert); validation/test labels from board-certified radiologists |

## Licence text

Stanford does not publish the agreement as a standalone page; it is shown and accepted at AIMI registration. The dataset card says only:

> License: Stanford University Dataset Research Use Agreement

The terms as summarised by Stanford's own materials (competition page, CheXpert Plus paper, Garbin et al. datasheet): permission to use the data **for non-commercial research purposes only**; you **may not distribute, publish or reproduce a copy** of the dataset; you may not attempt to re-identify individuals; Stanford retains all rights; provided "as is". The verbatim agreement must be copied into this file by whoever registers (see `datasets/ACQUISITION.md`).

## Obligations

* Non-commercial research only → `commercial_ok: false`; a model trained on CheXpert cannot be the basis of a commercial retrain (D-0xx policy in `registry.yaml`).
* Do not redistribute images or the download link. Keep them under `data/raw/chexpert/` (git-ignored).
* Cite: Irvin J. et al., "CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison", AAAI 2019 (arXiv:1901.07031).

## Notes for the adapter

`chexpert_manifest()` expects `train.csv` / `valid.csv` with columns `Path, Sex, Age, Frontal/Lateral, AP/PA, Pneumothorax, Pleural Effusion, Lung Lesion` and keeps `-1` (uncertain) for masking (D-017). `train_cheXbert.csv` / `train_visualCheXbert.csv` are alternative label sets, not used by default.
