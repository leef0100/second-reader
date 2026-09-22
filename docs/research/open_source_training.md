For research use only. Not for diagnostic use.

# Open-source chest X-ray classification pipelines: survey and benchmark for the Second Reader NIH baseline

> Due-diligence research for the Second Reader project (targets: `pneumothorax`, `pleural_effusion`, `nodule_mass`). Every number below is transcribed from the cited primary source (arXiv/journal PDF, GitHub README/config, model card) and was accessed on **2026-09-21** unless stated. Numbers we could not confirm against a primary source are marked **unverified**. Licence classes: **ADOPT** = permissive (MIT/Apache/BSD; code may be copied with attribution); **IDEAS-ONLY** = GPL/AGPL/non-commercial/no licence file (default all-rights-reserved) — we may learn from the design but must not copy code or weights. Independently of licence, rule **D-007** forbids using any weights pre-trained on NIH / CheXpert / MIMIC / PadChest for the externally evaluated model, so every "CXR-pretrained" checkpoint below is **ideas only for weights** even when its code is MIT.

Our pipeline, for reference (do not edit from this doc): `src/secondreader/modalities/cxr/train.py` (masked BCE over labelled entries, optional per-finding `pos_weight = n_neg/n_pos`, AdamW lr 1e-4 wd 1e-4, OneCycleLR pct_start 0.1, grad-clip 5, batch 16, early stop on mean val AUROC patience 3, view-head auxiliary CE ×0.3, per-finding temperature scaling + Youden threshold on val, Mahalanobis OOD on train features); `model.py` (timm `densenet121` ImageNet init, `num_classes=0`, drop 0.2, linear heads); `preprocess.py` (rescale/MONOCHROME1 → 0.5/99.5 percentile window → [0,1] → letterbox 320² → 3-ch ImageNet mean/std); `data.py` (RandomAffine 7°, translate 0.05, scale 0.9–1.1; 80 % brightness/contrast ±0.1; **no flips**); `cache.py` (uint8 memmap at 320 px); `evaluate.py` (AUROC, sens/spec at threshold, ECE, Brier, bootstrap CIs, subgroups). Split `nih-v1` = hashed **random patient-level 80/10/10** (90,036 / 10,637 / 11,447 images), **not** the official `train_val_list.txt` / `test_list.txt`.

---

## 0. Executive summary

1. **Two incompatible NIH benchmarks exist.** Papers that use the official patient-wise `test_list.txt` (Wang 2017, Guendel 2018, Baltruschat 2019 §3.1, Ark 2023) report mean-14 AUROC of **0.745–0.83**; papers that draw their own random patient split (Yao 2017, CheXNet 2017 and its reproductions, CheXclusion 2020, Guendel's own "random" column) report **0.80–0.85**. Guendel 2018 measured the gap on one model: mean 0.807 (official) vs 0.841 (random) [R4]; Baltruschat 2019 measured 0.795–0.806 (official) vs 0.822 (5× resampled random) for the same architecture [R5]. **Our `nih-v1` is a random patient split, so it is comparable only to the random-split rows** (§6).
2. **Typical random-split AUROC for our three targets** (verified in §6): pneumothorax **0.86–0.89**, effusion **0.86–0.89**, nodule **0.73–0.80**, mass **0.82–0.87**. The claim "pneumothorax ~0.85–0.90, effusion ~0.87–0.90, nodule ~0.75–0.80, mass ~0.80–0.86" is broadly right; the effusion lower bound is nearer 0.86 and Baltruschat's nodule numbers dip to 0.73 (§6.1). On the **official** split subtract roughly 0.03–0.06: pneumothorax 0.80–0.85, effusion 0.76–0.83, nodule 0.67–0.76, mass 0.69–0.82 (§6.2).
3. **NIH labels are noisy in exactly our classes.** Oakden-Rayner's visual audit found label PPV (clinical rule) of 60 % for pneumothorax without a chest drain, 64 % nodule, 46 % mass, 70 % effusion, and that **80 % of "pneumothorax" positives have a chest drain** [R29]. Google/Majkowska released radiologist-adjudicated labels for 4,374 NIH images (pneumothorax, nodule/mass, airspace opacity, fracture) — two of our three targets — on which their model reached AUROC 0.94 (pneumothorax) and 0.91 (nodule/mass) [R28].
4. **Nothing off-the-shelf can be used for weights** (D-007): torchxrayvision, CheXzero, KAD, RAD-DINO, Ark/Ark+ and BiomedCLIP (research-only) are all trained on NIH/CheXpert/MIMIC/PadChest or carry non-commercial terms. Permissive **code** we may adopt: torchxrayvision (Apache-2.0), jfhealthcare/Chexpert (Apache-2.0), sneddy SIIM-ACR (MIT), i-pan RSNA (MIT), jrzech/reproduce-chexnet (BSD-3), CheXzero/KAD (MIT, code only), MONAI (Apache-2.0).
5. Our recipe (DenseNet-121 ImageNet, 320 px, masked weighted BCE, Adam-family lr 1e-4, batch 16, few epochs) is essentially the CheXpert/Irvin 2019 recipe [R7] and is well within the literature; the main levers the literature suggests are **resolution ≥448 for nodule/mass**, **checkpoint/seed ensembling**, an **official-split evaluation run**, and an **adjudicated-label test set** (§8).

---

## 1. Classic NIH ChestX-ray14 papers

### 1.1 Wang et al. 2017 — ChestX-ray8 / ChestX-ray14 (CVPR) [R1]
- **Source / licence**: arXiv 1705.02315 (paper only; no training code released). IDEAS-ONLY by default (paper).
- **Data**: 112,120 frontal images, 30,805 patients; labels text-mined from reports (NegBio-style NLP); images exported from DICOM as 1024×1024 8-bit PNG.
- **Model**: ImageNet-pretrained AlexNet/GoogLeNet/VGG-16/ResNet-50 with a transition layer + global pooling; ResNet-50 best.
- **Loss**: weighted cross-entropy (W-CEL) balancing positives/negatives per class.
- **Split**: paper text says "randomly shuffled ... training (70 %), validation (10 %), testing (20 %)" and fine-tuned with SGD. The patient-level `train_val_list.txt` (86,524) / `test_list.txt` (25,596) shipped with the dataset are what Guendel 2018 [R4] and Baltruschat 2019 [R5] call the **official split** and pair with Wang's ChestX-ray14 column below.
- **AUROC (ResNet-50, ChestX-ray14 column)**: Atelectasis 0.7003, Cardiomegaly 0.8100, **Effusion 0.7585**, Infiltration 0.6614, **Mass 0.6933**, **Nodule 0.6687**, Pneumonia 0.6580, **Pneumothorax 0.7993**, Consolidation 0.7032, Edema 0.8052, Emphysema 0.8330, Fibrosis 0.7859, Pleural Thickening 0.6835, Hernia 0.8717; mean **0.7451** (mean computed by Guendel [R4]).
- **Caveat**: a second set of "Wang" numbers circulates (Effusion 0.784, Mass 0.706, Nodule 0.671, Pneumothorax 0.806, mean 0.738) — quoted as "Wang et al." by Yao [R2] and CheXNet [R3]; we could not trace which revision they come from (**unverified**). Use the 0.7585/0.6933/0.6687/0.7993 set when comparing to official-split work.

### 1.2 Yao et al. 2017 — "Learning to diagnose from scratch by exploiting dependencies among labels" [R2]
- **Source**: arXiv 1710.10501 (paper only). IDEAS-ONLY.
- **Model**: DenseNet-style encoder trained **from scratch (no ImageNet)**, LSTM decoder over labels; **512×512** input.
- **Augmentation**: random translate ±25 px, rotate ±15°, scale 0.8–1.2 ("crucial in combatting overfitting").
- **Optimiser**: Adam lr 1e-3, ×0.9 on validation plateau; early stop after 10,000 updates without improvement.
- **Split**: own **random 70/10/20** ("there is no standard split for this dataset").
- **AUROC (model a, independent labels)**: **Effusion 0.859, Mass 0.792, Nodule 0.717, Pneumothorax 0.841**, average 0.798.

### 1.3 Rajpurkar et al. 2017 — CheXNet (arXiv v3) [R3]
- **Source**: arXiv 1711.05225. Stanford did not release CheXNet training code (only the CheXpert labeler is public, MIT [R14]). IDEAS-ONLY (paper).
- **Model**: DenseNet-121, ImageNet init, final FC replaced, sigmoid; **224×224**, ImageNet mean/std normalisation.
- **Augmentation**: random horizontal flip only.
- **Loss**: weighted BCE with w+ = |N|/(|P|+|N|), w− = |P|/(|P|+|N|).
- **Optimiser**: Adam (β 0.9/0.999), lr 1e-3 decayed ×10 when val loss plateaus after an epoch; **batch 16**; pick lowest val loss. Mixed precision: not mentioned (2017).
- **Split**: **random patient-level 70/10/20** for the 14-class comparison ("following previous work ... no patient overlap"). (The pneumonia radiologist study used a 420-image test set.)
- **AUROC**: Atelectasis 0.8094, Cardiomegaly 0.9248, **Effusion 0.8638**, Infiltration 0.7345, **Mass 0.8676**, **Nodule 0.7802**, Pneumonia 0.7680, **Pneumothorax 0.8887**, Consolidation 0.7901, Edema 0.8878, Emphysema 0.9371, Fibrosis 0.8047, Pleural Thickening 0.8062, Hernia 0.9164; mean **0.841** (our arithmetic; matches [R10]).

### 1.4 Guendel et al. 2018 — "Learning to recognize abnormalities in chest X-rays with location-aware dense networks" (DNetLoc) [R4]
- **Source**: arXiv 1803.04565 (Siemens; no code). IDEAS-ONLY.
- **Model**: DenseNet-121, ImageNet normalisation (3 channels), **1024×1024 input** via two extra stride-2 3×3 conv layers initialised as Gaussian down-sampling; global average pool enlarged to 8×8.
- **Loss**: per-class BCE with **batch-frequency weights** wP = (P+N)/P, wN = (P+N)/N (same idea as our `pos_weight = n_neg/n_pos`).
- **Optimiser**: Adam (β 0.9/0.999, ε 1e-8), lr 1e-3, ÷10 on val plateau; **batch 128**.
- **Split**: **official patient-wise split** for Table 1, plus their own random patient splits; they note the official test set has "on average 3 times more images per patient" than train and "large label inconsistency".
- **AUROC official split (DNet)**: **Effusion 0.828, Mass 0.821, Nodule 0.758, Pneumothorax 0.846**, mean **0.807**. **Same model, random patient split**: Effusion 0.885, Mass 0.854, Nodule 0.774, Pneumothorax 0.872, mean **0.841**. This is the cleanest single-model quantification of the split effect: **+0.034 mean AUROC** from official → random.

### 1.5 Baltruschat et al. 2019 — "Comparison of deep learning approaches for multi-label chest X-ray classification" (Scientific Reports) [R5]
- **Source**: arXiv 1803.02315; Sci Rep 9:6381 (Sci Rep articles are normally CC BY — we could not read the licence page through the publisher's redirect, **unverified**). No code. IDEAS-ONLY.
- **Models**: ResNet-38/50/101; off-the-shelf (OTS) vs fine-tuned (FT) ImageNet at **224**, vs **from scratch at 448×448×1** ("large"); optional non-image features (view position, age, sex) concatenated before the FC ("-meta").
- **Preprocessing/aug**: resize to 256 (small) / 480 (large) then centre crop; geometric augmentation rotation **±7°** and horizontal flip; no dropout.
- **Optimiser**: Adam (β 0.9/0.999), lr 1e-3 (transfer) / 1e-2 (scratch), halved on val plateau; batch 16 (transfer) / 8 (large); CNTK on GTX 1080.
- **Split**: **5× resampling, patient-wise 70/10/20**, results averaged; plus a separate run on the **official split** (Table 5).
- **AUROC, 5× resampled (mean ± std, ×100)** — ResNet-50 FT 224 without meta: Pneumothorax 87.0±0.8, Effusion 87.1±0.2, Mass 82.2±1.0, Nodule 72.6±0.9, avg 81.7±1.0. ResNet-50-large-meta (448, scratch, +meta): Pneumothorax 85.9±1.1, Effusion 87.3±0.3, Mass 83.2±0.3, Nodule 75.8±1.4, avg **82.2±1.1** (best). Going 224→448 from scratch improved nodule by ≈0.018 and mass by ≈0.006.
- **AUROC, official split** — ResNet-50-large-meta: Pneumothorax 0.819, Effusion 0.818, Mass 0.810, Nodule 0.736, avg **0.795**; ResNet-38-large-meta: 0.840 / 0.822 / 0.820 / 0.747, avg **0.806**. Same architecture family loses ≈0.02–0.03 mean AUROC on the official split; the authors warn that "a fair comparison between groups without the same splitting might be non conclusive".
- **Label-noise finding**: Grad-CAM shows the network attends to **chest drains** for "pneumothorax" (citing Oakden-Rayner); they conclude a ChestX-ray14-trained model "would also respond to cases with a drain".

### 1.6 Seyyed-Kalantari et al. 2020 — CheXclusion (PSB 2021) [R6]
- **Source**: arXiv 2003.00827; code https://github.com/LalehSeyyed/CheXclusion — **no LICENSE file** (GitHub API `license: null`) → IDEAS-ONLY.
- **Model**: DenseNet-121 ImageNet init; images resized to **256×256**, ImageNet mean/std, centre crop; random horizontal flip + random rotation (10° for NIH, 15° for CheXpert/MIMIC, tuned).
- **Loss**: plain multi-label BCE (no class weights reported). CheXpert **uncertain → negative** (aggregated with "not mentioned").
- **Optimiser**: Adam default; initial lr **5e-4** (NIH) / 1e-4 (CheXpert); halve on 3-epoch plateau; stop after 10 epochs without improvement; **batch 32** (NIH) / 48; 5 seeds, 95 % CI reported.
- **Split**: **random patient-level 80/10/10** (same design as our `nih-v1`).
- **AUROC NIH (mean over 5 seeds)**: Effusion **0.875±0.002**, Mass **0.829±0.006**, Nodule **0.779±0.006**, Pneumothorax **0.879±0.005**, average **0.840±0.001**.
- **AUROC CheXpert (random 22,274-image test subset, not the official 200/500-study test)**: Pleural Effusion 0.885, Pneumothorax 0.893, Lung Lesion 0.780. **MIMIC-CXR**: Effusion 0.933, Pneumothorax 0.903, Lung Lesion 0.772.

### 1.7 Irvin et al. 2019 — CheXpert (AAAI) [R7]
- **Source**: arXiv 1901.07031; labeler code MIT [R14]; model code not released.
- **Model/training**: DenseNet-121, **320×320**, Adam default β, **lr 1e-4**, **batch 16**, **3 epochs**, checkpoint every 4,800 iterations, ensemble of the best 10 checkpoints × 3 runs = 30 checkpoints.
- **Uncertain-label strategies (validation AUROC, Pleural Effusion)**: U-Ignore 0.928, U-Zeros 0.931, U-Ones 0.934, U-SelfTrained 0.932, U-MultiClass 0.936 — all CIs overlap for effusion; only Atelectasis (U-Ones 0.858 vs U-Zeros 0.811, p = 0.03) and Cardiomegaly (U-MultiClass) showed significant differences. Masking uncertain labels (our D-017 = U-Ignore) is therefore a defensible default for effusion. Pneumothorax and Lung Lesion are not among the 5 competition tasks; no AUROC is given for them in the paper.
- **Test set**: 500 studies, ground truth by consensus of 5 board-certified radiologists.

---

## 2. Open-source pipelines and reproductions

### 2.1 torchxrayvision (mlmed) [R8, R9]
- **Repo/licence**: https://github.com/mlmed/torchxrayvision — **Apache-2.0** (LICENSE file; GitHub API reports NOASSERTION because of a custom header). Code → ADOPT. **Weights → forbidden by D-007** (trained on NIH, PadChest, CheXpert, MIMIC-CXR, RSNA, etc.).
- **Models**: `densenet121-res224-{all,nih,pc,chex,mimic_nb,mimic_ch,rsna}` at **224×224**; `resnet50-res512-all` at 512; 18 output pathologies (NaN for labels a dataset lacks).
- **Normalisation**: pixels scaled from the image's **full possible bit-depth range** (not min/max of the image) to **[-1024, 1024]** — deliberately avoids per-image contrast stretching. (Our percentile window does the opposite; see §8.)
- **Augmentation**: random rotation up to **45°**, translation up to **15 %**, scale ±10 % (from Cohen 2019 tuning).
- **Missing labels**: loss computed only on available labels (masked BCE) with task-frequency weighting [R9] — the same design as our `train.py`.
- **Calibration**: per-disease operating point that maximises TPR − FPR on held-out data (i.e. Youden), then a piecewise-linear map so that 0.5 is the decision boundary — equivalent to our Youden threshold [R8 §"calibrate"].
- **Reported NIH per-pathology AUROC**: not tabulated in the paper or the models page (only cross-dataset figures in [R9]) — **unverified / n/a**.

### 2.2 arnoweng/CheXNet (PyTorch reimplementation) [R10]
- **Licence**: **no LICENSE file** (GitHub API `null`) → IDEAS-ONLY. Repo ships `model.pth.tar`.
- **Split**: random 70/10/20 per README.
- **AUROC (README table: paper → implementation → "improved")**: Effusion 0.8638 → 0.8870 → 0.8891; Mass 0.8676 → 0.8597 → 0.8627; Nodule 0.7802 → 0.7873 → 0.7883; Pneumothorax 0.8887 → 0.8726 → 0.8844; mean 0.841 → 0.847. Preprocessing (resize 256 + TenCrop 224, ImageNet norm) is what the code does per community reports — **unverified** here.

### 2.3 zoogzog/chexnet [R11]
- **Licence**: **no LICENSE file** → IDEAS-ONLY.
- **Recipe**: DenseNet-121/169/201; resize 256 → crop 224; **TenCrop TTA** at test; random 70/10/20.
- **AUROC**: Effusion **0.8860**, Mass **0.8653**, Nodule **0.8037**, Pneumothorax **0.8857**, mean **0.8508** (highest "CheXNet-style" mean we found; TenCrop helps).

### 2.4 jrzech/reproduce-chexnet [R12]
- **Licence**: **BSD-3-Clause** → ADOPT.
- **Recipe**: torchvision DenseNet-121, **SGD + momentum** ("better than Adam"), batch 16; ships a 621-image enriched test sample. Split proportion not stated in README (**unverified**; random).
- **AUROC**: Pneumothorax **0.8776**, Effusion **0.8827**, Nodule **0.7756**, Mass **0.8377** (vs CheXNet 0.8887 / 0.8638 / 0.7802 / 0.8676).

### 2.5 jfhealthcare/Chexpert (JF Healthcare, CheXpert leaderboard #1 at the time) [R13]
- **Licence**: **Apache-2.0** (`LICENSE-2.0.txt`) → ADOPT (code for FPA attention, AVG_MAX pooling, PCAM).
- **Config (`config/example.json`)**: DenseNet-121 pretrained; **512×512 with fixed aspect ratio** (`fix_ratio`, border padded with pixel mean); pixel mean 128 / std 64; **histogram equalisation on**; augmentation type "Aug" (details in code); Gaussian blur k=3; BatchNorm; global pool `AVG_MAX`; attention `FPA`; FC BatchNorm; dropout 0.
- **Training**: **Adam lr 1e-4, ×0.1 at epoch 2, 3 epochs, batch 56**, weight decay 0; criterion BCE, `pos_weight` [1,1,1,1,1] with `batch_weight: true`, `enhance_index [2,6]` ×1 (oversampling hooks); target metric AUC; top-3 checkpoints saved. Mixed precision: not in config. Uncertain-label policy: not visible in config (**unverified**).
- **Reported CheXpert (200-patient test) AUROC**: Cardiomegaly 0.8703, Edema 0.9436, Consolidation 0.9334, Atelectasis 0.9029, **Pleural Effusion 0.9166**, mean 0.9154.

### 2.6 stanfordmlgroup [R14]
- Only `chexpert-labeler` (MIT) is public; CheXNet/CheXpert training code is not. Nothing to adopt for training.

### 2.7 MONAI tutorials and model zoo [R15]
- **Licence**: tutorials and model-zoo are **Apache-2.0** → ADOPT.
- **Model zoo (37 bundles, `models/` on `dev`)**: the only CXR bundle is `cxr_image_synthesis_latent_diffusion_model` (512² generation). Classification bundles are `breast_density_classification`, `endoscopic_inbody_classification`, `pathology_nuclei_classification`, and a generic `classification_template`. **No chest X-ray classification bundle exists.**
- **Tutorials `2d_classification/`**: `mednist_tutorial.ipynb`, `monai_101.ipynb`, `monai_201.ipynb` — MedNIST (64×64 toy set that includes some NIH thumbnails), DenseNet-121 — **not** a CXR pathology pipeline. Useful only as a framework reference.

### 2.8 NVIDIA Clara `clara_xray_classification_chest_no_amp` (NGC) [R16]
- **Terms**: NVIDIA Clara Train EULA (proprietary; model marked compatible only with Clara Train SDK v2.0 and the Clara Deploy CXR container is deprecated) → IDEAS-ONLY.
- **Recipe**: DenseNet-121, **256×256**, 16-bit PNG input, trained on **PLCO** (not NIH), 15 outputs incl. Nodule, Mass, Pleural Effusion; average AUC over categories **0.797** on PLCO validation; an `_amp` sibling exists (AMP = automatic mixed precision), 12 GB GPU required; "This is an example, not to be used for diagnostic purposes".

### 2.9 Field-by-field recipe comparison

Consolidated from §1–§2 and §5 so each requested field can be read across projects. "—" = not stated in the source we read; "n/a" = not applicable. Our own pipeline is the last row for side-by-side reading.

| Project | Backbone (init) | Input res. | Normalisation / windowing | Augmentation | Uncertain labels | Imbalance | Loss | Optimiser / LR schedule | Epochs | Batch | AMP | Eval split | Ref |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Wang 2017 | ResNet-50 (ImageNet) | 224 from 1024 PNG | ImageNet | — | n/a (NIH) | class-weighted CE | W-CEL | SGD | — | — | — | random 70/10/20 (paper); official lists shipped | [R1] |
| Yao 2017 | DenseNet-like (scratch) | 512 | — | translate ±25 px, rot ±15°, scale 0.8–1.2 | n/a | — | BCE (+LSTM) | Adam 1e-3, ×0.9 on plateau; early stop 10k updates | — | — | — | random 70/10/20 | [R2] |
| CheXNet 2017 | DenseNet-121 (ImageNet) | 224 | ImageNet mean/std | h-flip | n/a | w+/w− class weights | weighted BCE | Adam 1e-3, ÷10 on plateau | until plateau | 16 | — | random 70/10/20 | [R3] |
| Guendel 2018 | DenseNet-121 (ImageNet) + 2 stride-2 convs | 1024 | ImageNet, 3-ch | — | n/a | batch-frequency wP/wN | weighted BCE ×14 | Adam 1e-3, ÷10 on plateau | — | 128 | — | official + own random | [R4] |
| Baltruschat 2019 | ResNet-38/50/101 (ImageNet or scratch) | 224 / 448×448×1 | resize 256/480, centre crop | rot ±7°, h-flip | n/a | — (BCE) | multi-label BCE | Adam 1e-3 (FT) / 1e-2 (scratch), ÷2 on plateau | — | 16 / 8 | — (CNTK, GTX 1080) | 5× random 70/10/20 + official | [R5] |
| CheXclusion 2020 | DenseNet-121 (ImageNet) | 256 → centre crop | ImageNet mean/std | h-flip, rot 10° (NIH) | uncertain → negative | none | BCE | Adam 5e-4 (NIH), ÷2 on 3-epoch plateau, stop after 10 | variable | 32 (NIH) | — | random 80/10/10 | [R6] |
| Irvin 2019 (CheXpert) | DenseNet-121 (ImageNet) | 320 | — | — | U-Ignore/Zeros/Ones/SelfTrained/MultiClass | — | BCE | Adam 1e-4 | 3 (ckpt every 4,800 it; 30-ckpt ensemble) | 16 | — | official val 200 / test 500 studies | [R7] |
| torchxrayvision | DenseNet-121 (ImageNet) / ResNet-50 | 224 / 512 | full bit-depth → [−1024, 1024] | rot ≤45°, translate ≤15 %, scale ±10 % | n/a | task-frequency weights | masked BCE | — | — | — | — | own splits (unstated) | [R8, R9] |
| jfhealthcare | DenseNet-121 (ImageNet) + FPA, AVG_MAX pool | 512, fixed ratio, pad with mean | (x−128)/64, hist-eq, blur k3 | "Aug" (code) | — | batch weighting, `enhance_index` oversampling | BCE, pos_weight 1 | Adam 1e-4, ×0.1 at epoch 2, wd 0 | 3 | 56 | — | CheXpert official test | [R13] |
| NVIDIA Clara | DenseNet-121 | 256 | 16-bit PNG | — | n/a | — | — | — | — | — | `_amp` sibling exists | PLCO val | [R16] |
| SIIM-ACR 1st (seg.) | U-Net: ResNet-34/50, SE-ResNeXt-50 | 512 → 1024 | — | h-flip, contrast/gamma/brightness p0.3, elastic/grid/optical p0.3, ShiftScaleRotate | n/a | negative sampling 0.8→0.4 | BCE+Dice+Focal | ReduceLROnPlateau → cosine (warm restarts) | 10–12 + up-train stages | 2–4 | — | 5-fold CV + LB | [R17] |
| RSNA 1st | InceptionResNetV2 cls (1-ch) + RetinaNet / Def. R-FCN / Def. RelationNet | 256 cls / 384 det | grayscale DICOM | — | n/a | — | — | — | — | — | — | fold CV, AUC+F1 selection | [R18] |
| CXR-LT 2023 T1 | ConvNeXt-S (ImageNet → NIH/CheXpert/VinDr) | 1024 | — | — | n/a (MIMIC labels) | weighted ASL | ASL | — | two-stage | — | — | 70/10/20 challenge split, mAP | [R20] |
| CXR-LT 2024 top teams | ConvNeXt, ViT-L, MaxViT | 224–1024 multi-res | — | multi-scale crops, flips, diffusion synth | — | ASL / weighted BCE / focal | ASL etc. | — | — | — | — | challenge test + gold subset | [R21] |
| CheXzero | CLIP ViT-B/32 (OpenAI CLIP) | 224 | CLIP | — | n/a (reports) | — | contrastive | SGD 1e-4 mom 0.9 | 4 | 64 | — | CheXpert test 500 (zero-shot) | [R23] |
| KAD | ResNet-50 | 224 (512/1024) | — | — | n/a | — | contrastive + BCE | AdamW 1e-4, 5 warm-up | 50 | 64 | — (A100) | NIH / CheXpert / PadChest zero-shot | [R27] |
| Ark (MICCAI 2023) | Swin-B (ImageNet) | 224 | — | random crop + rotation; brightness/contrast/gamma | dataset-native labels | — | BCE / CE + MSE consistency | SGD 0.3, cosine, EMA teacher | 200 rounds | 200 | — (4× V100) | NIH official (75,312 / 11,212 / 25,596) | [R26] |
| RAD-DINO | ViT-B/14 (DINOv2) | 518 | resize (B-spline) then min-max [0,255] | DINOv2 dual-view, larger crops, less blur | n/a | — | DINOv2 | — | 60k steps | 640 | — | VinDr / CANDID-PTX / RSNA (AUPRC) | [R25] |
| **Second Reader (ours)** | timm DenseNet-121 (ImageNet), drop 0.2 | 320 letterbox | percentile 0.5/99.5 → [0,1] → ImageNet | RandomAffine 7°, 5 %, 0.9–1.1; jitter ±0.1 (80 %); **no flip** | masked (D-017) | optional pos_weight n_neg/n_pos | masked BCE + 0.3 view CE | AdamW 1e-4 wd 1e-4, OneCycle pct 0.1, clip 5 | early stop patience 3 | 16 | fp32 on MPS | `nih-v1` random 80/10/10 | — |

---

## 3. Kaggle winning solutions

### 3.1 SIIM-ACR Pneumothorax Segmentation 2019 — 1st place (sneddy / Aimoldin Anuar) [R17]
- **Licence**: **MIT** → ADOPT (augmentation and sampling code).
- **Models**: U-Net variants — AlbuNet (ResNet-34), ResNet-50, SCSE-U-Net (SE-ResNeXt-50); progressive **512 → 1024** with encoder frozen for early up-train epochs.
- **Augmentation (albumentations)**: `HorizontalFlip`, `OneOf([RandomContrast, RandomGamma, RandomBrightness], p=0.3)`, `OneOf([ElasticTransform, GridDistortion, OpticalDistortion], p=0.3)`, `ShiftScaleRotate`, `Resize`.
- **Imbalance**: fraction of non-pneumothorax images sampled per epoch reduced from **0.8 → 0.4** over training.
- **Loss**: combo BCE + Dice + Focal with weights (3,1,4) / (1,1,1) / (2,1,2).
- **Schedule**: ReduceLROnPlateau, then CosineAnnealing(WarmRestarts) in up-train stages; batch 2–4; 10–12 epochs in stage 0; 1–2× GTX 1080 Ti. Mixed precision: not stated.
- **Inference**: hflip TTA; top-3 checkpoint averaging per fold per pipeline; post-processing triplet (top score 0.75, min area 2000 px, bottom score 0.3).

### 3.2 RSNA Pneumonia Detection 2018 — 1st place (Ian Pan, Alexandre Cadrin-Chênevert) [R18]
- **Licence**: **MIT** → ADOPT. Paper: Pan et al., AJR 2019.
- **Approach**: classifier + detector ensemble — InceptionResNetV2 classifier on **256×256 grayscale (1-channel)** DICOM; detectors RetinaNet (384), Deformable R-FCN, Deformable Relation Networks; Keras/TF; fold-based CV; model selection on **AUC + F1**; stage-1 data only. "Simple" variant (1 classifier + 5 relation nets) scored 0.253 on private LB. LR/epochs/augmentation not in README (**unverified**).

### 3.3 VinBigData Chest X-ray Abnormalities Detection 2021 [R19]
- The official **1st-place write-up** on Kaggle could not be read (page requires client-side rendering) — details **unverified**.
- **3rd place (Scu-sen)**: no LICENSE → IDEAS-ONLY. Multi-label classifier trained at **1024 / 1024-fixed-aspect / 1280-fixed-aspect**, 5-fold ×3 resolutions; 11 YOLOv5 detectors (30 epochs @512 batch 16, 60 epochs @1024 batch 4); dedicated aortic-enlargement detector (100 epochs); post-processing **multiplies detector confidence by the classifier's abnormal probability** and uses DICOM metadata to flag likely-normal studies; 8× V100 32 GB.
- **2nd place (ZFTurbo)**: no LICENSE → IDEAS-ONLY; three member pipelines ensembled; details not in README.
- Take-away for us: the top detection solutions all gate detections with an image-level classifier trained at ≥1024 px.

---

## 4. CXR-LT long-tailed challenges (MIMIC-CXR based)

### 4.1 CXR-LT 2023 (ICCV CVAMD) [R20]
- 377,110 CXRs, 26 findings, 70/10/20 split, metric mAP. Top-5: T1 **0.372**, T2 0.354, T3 0.354, T4 0.351, T5 0.349.
- **T1**: ConvNeXt-S at **1024×1024**, ImageNet → NIH/CheXpert/VinDr pretraining, **weighted asymmetric loss (ASL)**, ML-Decoder head, cross-view transformer, two-stage training, ensemble.
- **T2**: EfficientNetV2-S and ConvNeXt-S at 512/768, weighted focal loss, heavy **mosaic** augmentation, TTA + stratified-CV ensemble.
- **T3**: ConvNeXt-B at 448, ImageNet-21k → NIH, weighted CE, head/tail expert ensemble, Lion optimiser.
- **Organisers' take-aways**: high resolution (>300 px), modern CNNs (ConvNeXt, EfficientNetV2), large-scale CXR pretraining, strong augmentation + ensembling, **loss re-weighting for tail classes** (used by all top-5), text-label representations.

### 4.2 CXR-LT 2024 (MICCAI) [R21]
- 377,110 CXRs, 45 labels (19 new rare). Task 1 (noisy test) top mAP **0.281 / 0.279 / 0.277**; Task 2 (gold-standard subset) **0.526 / 0.511 / 0.511**; Task 3 zero-shot 0.129.
- ConvNeXt most common (4 of 9 top teams), ViT-L/MaxViT rising; resolutions **224–1024 with multi-resolution ensembles**; **ASL** in 5 of 9 teams, weighted BCE, focal; 8 of 9 use ensembles; one team added 100 diffusion-synthesised images per rare class; GPT-4 relabelling precision 0.786 vs 0.711 rule-based on the gold set. Eight of nine relied on supervised pretraining / transfer.

---

## 5. Foundation / vision-language models (all weights blocked by D-007)

| Model | Repo / licence | Backbone, input | Pretraining data | Key numbers | Use for us |
|---|---|---|---|---|---|
| **CheXpert Plus** [R22] | github.com/Stanford-AIMI/chexpert-plus (no repo licence); data under Stanford Research Use Agreement, **non-commercial** | — (dataset) | 223,462 report–image pairs, 187,711 studies, 64,725 patients, DICOM + PNG, demographics, RadGraph | Paper reports labeler F1 (CheXbert 0.93), **no classification AUROC** | Data for a CheXpert arm only; no weights |
| **CheXzero** [R23] | MIT (code) | CLIP ViT-B/32, **224** | MIMIC-CXR 377,110 image–impression pairs; 4 epochs, batch 64, SGD lr 1e-4 mom 0.9 | Zero-shot CheXpert test mean **0.889**; Pleural Effusion **0.932**, Atelectasis 0.816, Cardiomegaly 0.906, Consolidation 0.892, Edema 0.897; "−0.042 below best supervised" | Code ADOPT; weights blocked (MIMIC) |
| **BiomedCLIP** [R24] | MIT; model card: "intended solely for research ... any deployed use case ... out of scope" | ViT-B/16, 224 + PubMedBERT | PMC-15M figure–caption pairs | No CXR AUROC on card | Research-only statement conflicts with a clinical product path; ideas only |
| **RAD-DINO** [R25] | MIT (HF card) | ViT-B/14 (DINOv2), **518** | MIMIC-CXR 368,960 + CheXpert 223,648 + **NIH 112,120** + PadChest 136,787 + BRAX 41,260 = 882,775; 60k steps, batch 640 | VinDr-CXR macro AUPRC 52.8; CANDID-PTX pneumothorax AUPRC 80.1; RSNA 71.0 | Blocked (NIH in pretraining) |
| **Ark / Ark+** [R26] | **ASU non-commercial licence** → IDEAS-ONLY | Swin-B, **224** (MICCAI 2023); Ark+ repo shows `img_size=768` | ChestX-ray14, CheXpert, MIMIC-II, VinDr-CXR, RSNA, Shenzhen (335k / 704k); SGD lr 0.3, batch 200, cosine, 200 rounds, 4× V100; aug: random crop + rotation; brightness/contrast/gamma | Fine-tune AUROC **NIH official split (test 25,596): Ark-6 83.05±0.09**; ImageNet-supervised Swin-B baseline **81.73±0.14**; CheXpert (234-image val) 89.14 | Blocked; but the Swin-B ImageNet baseline is a valid official-split reference |
| **KAD** [R27] | MIT (code) | ResNet-50, **224** (512/1024 variants) | MIMIC-CXR; 50 epochs, batch 64, lr 1e-4, A100 | Zero-shot ChestX-ray14 mean **0.789**: Pneumothorax 0.860, Effusion 0.824, Nodule 0.698, Mass 0.754; CheXpert mean 0.905, Pleural Effusion 0.949; fine-tune 100 % labels 0.825 vs ImageNet-init 0.804 (split **unverified**) | Code ADOPT; weights blocked |

---

## 6. Benchmark table — NIH ChestX-ray14 AUROC

Columns: split (**OFF** = official `train_val_list`/`test_list`; **RND** = author's random patient-level split), resolution, backbone, then per-finding AUROC. "mean-14" is over the 14 NIH labels. All values transcribed from the cited source; ± are std over resamples/seeds where reported.

### 6.1 Random patient splits (comparable to our `nih-v1`)

| Paper / repo | Split | Res. | Backbone (init) | Pneumothorax | Effusion | Nodule | Mass | mean-14 | Ref |
|---|---|---|---|---|---|---|---|---|---|
| Yao 2017 | RND 70/10/20 | 512 | DenseNet-like (scratch) | 0.841 | 0.859 | 0.717 | 0.792 | 0.798 | [R2] |
| CheXNet 2017 (v3) | RND 70/10/20 | 224 | DenseNet-121 (ImageNet) | **0.889** | 0.864 | 0.780 | **0.868** | 0.841 | [R3] |
| Guendel 2018 DNet, random column | RND | 1024 | DenseNet-121 (ImageNet) | 0.872 | **0.885** | 0.774 | 0.854 | 0.841 | [R4] |
| Baltruschat 2019 ResNet-50 FT | RND 5× 70/10/20 | 224 | ResNet-50 (ImageNet) | 0.870±0.008 | 0.871±0.002 | 0.726±0.009 | 0.822±0.010 | 0.817±0.010 | [R5] |
| Baltruschat 2019 ResNet-50-large-meta | RND 5× 70/10/20 | 448 | ResNet-50 (scratch) + meta | 0.859±0.011 | 0.873±0.003 | 0.758±0.014 | 0.832±0.003 | 0.822±0.011 | [R5] |
| CheXclusion 2020 | RND 80/10/10 | 256→crop | DenseNet-121 (ImageNet) | 0.879±0.005 | 0.875±0.002 | 0.779±0.006 | 0.829±0.006 | 0.840±0.001 | [R6] |
| arnoweng/CheXNet (impl. / improved) | RND 70/10/20 | 224 (TenCrop) | DenseNet-121 (ImageNet) | 0.873 / 0.884 | 0.887 / 0.889 | 0.787 / 0.788 | 0.860 / 0.863 | 0.841→0.847 | [R10] |
| zoogzog/chexnet | RND 70/10/20 | 224 (TenCrop) | DenseNet-121 (ImageNet) | 0.886 | 0.886 | **0.804** | 0.865 | **0.851** | [R11] |
| jrzech/reproduce-chexnet | RND (unverified ratio) | 224 | DenseNet-121 (ImageNet), SGD | 0.878 | 0.883 | 0.776 | 0.838 | n/a | [R12] |

**Observed random-split ranges**: pneumothorax **0.86–0.89**, effusion **0.86–0.89**, nodule **0.73–0.80**, mass **0.82–0.87**, mean-14 **0.82–0.85** (Yao's 0.798 is scratch-trained). The caller's claimed ranges (0.85–0.90 / 0.87–0.90 / 0.75–0.80 / 0.80–0.86) are consistent, with effusion's floor nearer 0.86 and nodule's floor nearer 0.73 in ResNet-50 work.

### 6.2 Official NIH split (`test_list.txt`, 25,596 images)

| Paper | Split | Res. | Backbone (init) | Pneumothorax | Effusion | Nodule | Mass | mean-14 | Ref |
|---|---|---|---|---|---|---|---|---|---|
| Wang 2017 ResNet-50 | OFF | 224 (from 1024) | ResNet-50 (ImageNet) | 0.799 | 0.759 | 0.669 | 0.693 | 0.745 | [R1, R4] |
| Yao 2017 as re-run by Baltruschat on OFF (Table 5) | OFF | 512 | DenseNet-like | 0.805 | 0.806 | 0.724 | 0.777 | 0.761 | [R5] |
| Guendel 2018 DNet | OFF | 1024 | DenseNet-121 (ImageNet) | **0.846** | **0.828** | **0.758** | **0.821** | **0.807** | [R4] |
| Baltruschat 2019 ResNet-38-large-meta | OFF | 448 | ResNet-38 (scratch) + meta | 0.840 | 0.822 | 0.747 | 0.820 | 0.806 | [R5] |
| Baltruschat 2019 ResNet-50-large-meta | OFF | 448 | ResNet-50 (scratch) + meta | 0.819 | 0.818 | 0.736 | 0.810 | 0.795 | [R5] |
| Ark 2023 — ImageNet-supervised Swin-B baseline (fine-tuned) | OFF (test 25,596) | 224 | Swin-B (ImageNet) | n/a | n/a | n/a | n/a | 0.817±0.001 | [R26] |
| Ark 2023 — Ark-6 (CXR-pretrained, D-007-blocked) | OFF | 224 | Swin-B (Ark) | n/a | n/a | n/a | n/a | 0.831±0.001 | [R26] |
| KAD fine-tune 100 % labels (split **unverified**) | ? | 224 | ResNet-50 (KAD) / ImageNet | n/a | n/a | n/a | n/a | 0.825 / 0.804 | [R27] |

**Observed official-split ranges**: pneumothorax **0.80–0.85**, effusion **0.76–0.83**, nodule **0.67–0.76**, mass **0.69–0.82**, mean-14 **0.745–0.83**. Same-model deltas official→random: Guendel **+0.034** mean (+0.026 pneumothorax, +0.057 effusion, +0.016 nodule, +0.033 mass) [R4]; Baltruschat ≈ +0.02–0.03 mean [R5].

### 6.3 CheXpert AUROC (pleural effusion / pneumothorax / lung lesion)

| Paper | Test set | Res. | Pleural Effusion | Pneumothorax | Lung Lesion | Ref |
|---|---|---|---|---|---|---|
| Irvin 2019 (validation, 200 studies; U-Ignore … U-MultiClass) | official val | 320 | 0.928–0.936 | n/a | n/a | [R7] |
| jfhealthcare (leaderboard #1 at the time) | official 200-patient test | 512 | 0.9166 | n/a | n/a | [R13] |
| CheXclusion 2020 | own random 22,274-image split | 256 | 0.885 | 0.893 | 0.780 | [R6] |
| CheXzero (zero-shot, MIMIC-trained) | official test (500 studies) | 224 | 0.932 | n/a (0.764 reported in a secondary source, **unverified**) | n/a | [R23] |
| KAD (zero-shot, MIMIC-trained) | official test | 224 | 0.949 | n/a | n/a | [R27] |

---

## 7. NIH label noise — what is known and what relabelled data exists

### 7.1 Oakden-Rayner 2020, "Exploring large-scale public medical image datasets" (Academic Radiology 27(1):106–112; arXiv 1907.12720) [R29]
- One board-certified radiologist visually reviewed **50 images per class × 15 class groups ≈ 700 CXR14 images** and scored the NIH label's **positive predictive value** under a permissive and a "clinical-style" rule (NPV could not be assessed).
- PPV (permissive / clinical / Wang's text-mining claim): **Nodule 76 % / 64 % / 96 %**; **Mass 64 % / 46 % / 75 %**; **Pneumothorax 90 % / 90 % / 90 %, but only 60 % for images without a chest drain**; **Pleural effusion 74 % / 70 % / 93 %**; Emphysema 14 % / 10 % (86 % were subcutaneous emphysema); Cardiomegaly 70 % / 52 % / 100 %.
- **80 % of "pneumothorax" positives contain chest drains**, often with no visible pneumothorax (treated cases) — a shortcut Baltruschat later confirmed with Grad-CAM [R5]. Recommendation: visual inspection is "a necessary component of understanding large image datasets".

### 7.2 Majkowska et al. 2020, Radiology 294(2):421–431 — radiologist-adjudicated NIH labels [R28]
- Google trained models for **pneumothorax, nodule/mass, airspace opacity, fracture** on 657,954 images (DS1 759,611 + ChestX-ray14) and evaluated on **radiologist-adjudicated** test sets: 1,818 (DS1) and **1,962 (ChestX-ray14)** images; reference standard by adjudicated review of three radiologists (up to five consensus rounds), four further radiologists read for comparison.
- **AUROC on the adjudicated ChestX-ray14 test subset**: **pneumothorax 0.94 (95 % CI 0.93–0.96)**, **nodule/mass 0.91 (0.89–0.93)**, airspace opacity 0.94 (0.93–0.95), fracture 0.81 (0.75–0.86). (DS1 population-adjusted: 0.95 / 0.72 / 0.91 / 0.86.)
- **Data release**: labels for **2,412 validation + 1,962 test = 4,374 ChestX-ray14 images** (`four_findings_expert_labels/`: `individual_readers.csv`, `validation_labels.csv`, `test_labels.csv`; values YES / NO / HEDGE), obtained via the Google form linked from the Cloud Healthcare NIH dataset page; citation required. A second set, `all_findings_expert_labels/` (**810 PA test images, all 14 findings + normal/abnormal, five ABR-certified radiologists**, Nabulsi et al. 2021), is on the same page. Note these adjudicated images are drawn from the **official NIH test/val lists**, so they overlap our `nih-v1` train partition unless excluded.

### 7.3 Consequences
- NIH AUROC measures agreement with NLP labels whose PPV for our classes is 46–90 %; AUROC against adjudicated labels can be *higher* (Majkowska 0.94 pneumothorax vs 0.85–0.89 against NLP labels) because label noise depresses apparent AUROC.
- The chest-drain confound means a high NIH pneumothorax AUROC does not certify detection of untreated pneumothorax; CXR-LT 2024 likewise found rule-based labels have precision ≈0.71 on its gold set [R21].

---

## 8. Implications for Second Reader (file-mapped)

1. **Compare `nih-v1` only to random-split rows, and add an official-split run** (`config.py` split registry, `evaluate.py` report). Expected `nih-v1` targets from §6.1: pneumothorax 0.86–0.89, effusion 0.86–0.89, nodule 0.73–0.80, mass 0.82–0.87 (a merged `nodule_mass` head should land between the two). Register a second split built from `train_val_list.txt`/`test_list.txt` so we can also quote Guendel 0.807 / Baltruschat 0.806 / ImageNet-Swin-B 0.817 official-split numbers; Guendel shows the same model drops ≈0.03 mean AUROC on the official split, so a report that mixes the two is meaningless [R4, R5].
2. **Resolution is the main lever for nodule/mass** (`preprocess.py`, `cache.py`, `config.py`). Every official-split leader uses ≥448 px (Guendel 1024, Baltruschat 448) and CXR-LT winners use 448–1024; Baltruschat measured +0.018 nodule AUROC from 224→448 [R5]; CheXpert's 320 recipe [R7] is our current point. On the cloud GPU, build a 448 or 512 memmap cache variant and re-run; keep 320 for the M1 (27 img/s ≈ 69 min/epoch already).
3. **Backbone**: DenseNet-121 ImageNet init is exactly what CheXNet, CheXclusion, Irvin and Guendel use, and D-007 rules out CXR-pretrained encoders anyway [R3, R6, R7]. ConvNeXt-S/B is the CXR-LT 2023/2024 consensus [R20, R21]; since `model.py` already uses timm, expose `backbone` as a config value and try `convnext_tiny` at 448 on the GPU. The Ark table shows ImageNet-supervised Swin-B reaches 0.817 mean-14 on the official split at 224 [R26], i.e. modern backbones give ≈+0.01 over DenseNet-121 (Guendel 0.807) without CXR pretraining.
4. **Augmentation policy** (`data.py`). Our RandomAffine(±7°, 5 %, 0.9–1.1) matches Baltruschat's ±7° exactly [R5] and is milder than Yao (±15°, ±25 px, 0.8–1.2) [R2] or torchxrayvision (45°, 15 %, 10 %) [R8]. Almost every NIH pipeline adds **horizontal flip** (CheXNet, Baltruschat, CheXclusion, SIIM winner) [R3, R5, R6, R17]; we deliberately omit it because laterality matters for a second reader — keep that, but record it as a decision. Cheap additions with precedent: random gamma/contrast (SIIM `OneOf(RandomContrast, RandomGamma, RandomBrightness)` p=0.3 [R17]; Ark's τ2 brightness/contrast/gamma [R26]) and rotation up to 10° (CheXclusion tuned 10° for NIH [R6]).
5. **Normalisation**: torchxrayvision scales by the image's **bit-depth range** rather than per-image percentiles, arguing that contrast stretching "could be removing information" [R8]; jfhealthcare instead applies histogram equalisation [R13]. Our 0.5/99.5 percentile window is a middle ground; log it as an ablation candidate in `preprocess.py` (fixed-range vs percentile) before external evaluation, since windowing interacts with domain shift.
6. **Loss and imbalance** (`train.py`). Masked BCE + `pos_weight = n_neg/n_pos` is the Cohen/TXRV masked loss [R9] combined with Guendel's batch-frequency weights [R4] / CheXNet's w+/w− [R3] — standard. CXR-LT winners uniformly used **loss re-weighting** and mostly **asymmetric loss (ASL)** or weighted focal for rare labels [R20, R21]; the SIIM winner instead down-sampled negatives from 0.8→0.4 [R17]. Add ASL as a selectable criterion for `nodule_mass`, the rarest of our targets.
7. **LR schedule and epochs** (`train.py`). AdamW 1e-4 + OneCycle(pct 0.1), batch 16 is the Irvin recipe (Adam 1e-4, batch 16, **3 epochs**, checkpoint every 4,800 it) [R7] and jfhealthcare (Adam 1e-4, ×0.1 at epoch 2, 3 epochs, batch 56) [R13]; CheXNet used 1e-3 with ÷10-on-plateau [R3]; CheXclusion 5e-4 with halve-on-3-epoch-plateau and stop after 10 [R6]. Two conclusions: (a) 3–6 epochs at 320 px are enough for DenseNet-121 on ~100k images, so the 69-min/epoch M1 budget is workable; (b) early-stop patience 3 on mean AUROC is stricter than CheXclusion's 10 — on the GPU allow patience 5 and use `OneCycleLR` total steps = planned epochs.
8. **Checkpoint/seed ensembling and TTA** (`train.py`, `evaluate.py`). Irvin averages the best 10 checkpoints × 3 runs [R7]; zoogzog's TenCrop TTA gives the highest CheXNet-style mean (0.851) [R11]; SIIM uses hflip TTA + top-3 checkpoint averaging [R17]; 8 of 9 CXR-LT 2024 top teams ensemble [R21]. Save top-k checkpoints by val AUROC and average logits at evaluation; hflip TTA is *not* appropriate for us (laterality), but multi-crop/scale TTA is.
9. **Mixed precision on the cloud GPU** (`train.py`). Use `torch.autocast("cuda", dtype=torch.bfloat16 or float16)` + `torch.amp.GradScaler` for fp16; `binary_cross_entropy_with_logits` autocasts safely to float32 whereas plain `binary_cross_entropy` raises inside autocast regions [R30] — our masked loss must stay on the `_with_logits` form. MPS autocast supports fp16 and (macOS 14+) bf16 per the PyTorch issue tracker [R30, secondary; the stable AMP docs list only cuda/cpu/xpu/hpu/mtia/maia] — treat MPS AMP as **unverified** and benchmark before relying on it. Ark trained Swin-B with 4× V100 [R26]; a single modern GPU at 448 px with AMP should give >150 img/s for DenseNet-121 (our estimate, unverified).
10. **Label noise handling** (`data.py`, `evaluate.py`, `docs/reports`). (a) Request Google's `four_findings_expert_labels` (4,374 images; pneumothorax and nodule/mass map directly onto two of our three targets) and `all_findings_expert_labels` (810 images, includes effusion) [R28]; **exclude those image IDs from `nih-v1` training** and report AUROC against adjudicated labels separately — Majkowska's reference points are 0.94 pneumothorax / 0.91 nodule-mass. (b) Add a **chest-drain/support-device subgroup** to `evaluate.py` (e.g. via the Google labels or a simple device detector) because 80 % of NIH pneumothorax positives have drains [R29] and networks attend to them [R5]. (c) D-012 (merge nodule+mass) is consistent with the adjudicated "nodule/mass" label and with Oakden-Rayner's finding that mass PPV is only 46 % clinical — the merged class is less noisy than either alone [R29].
11. **Uncertain CheXpert labels (D-017)**. Irvin's validation numbers for Pleural Effusion are U-Ignore 0.928 vs U-Ones 0.934 vs U-MultiClass 0.936 with overlapping CIs [R7], so masking is fine; CheXclusion maps uncertain→negative [R6]. Keep D-017 but add a U-Ones ablation flag in `train.py` for effusion/pneumothorax only.
12. **Metadata as input, not just as auxiliary target** (`model.py`, `train.py`). Baltruschat's "+meta" (view position, age, sex concatenated before the FC) gave +0.5 mean AUROC and the largest gains on nodule/mass [R5]. Our view-head predicts view as an auxiliary loss; consider also feeding view (and age/sex where available) as input features, which is cheap and has published precedent.
13. **Licence/adoption ledger** for the repo (`docs/`): ADOPT code from torchxrayvision (Apache-2.0: dataset loaders, masked-loss utilities), jfhealthcare (Apache-2.0: FPA/AVG_MAX/PCAM pooling), sneddy (MIT: albumentations policy, negative-sampling schedule), i-pan (MIT), jrzech (BSD-3), CheXzero/KAD (MIT, code only), MONAI (Apache-2.0). IDEAS-ONLY: Ark/Ark+ (ASU non-commercial), NVIDIA Clara (EULA), arnoweng, zoogzog, CheXclusion, Scu-sen, ZFTurbo (no licence file), CheXpert Plus data (non-commercial DUA), BiomedCLIP (research-only intended use). **No third-party CXR weights at all** (D-007) — every pretrained checkpoint surveyed here was trained on NIH, CheXpert, MIMIC or PadChest [R8, R23, R25, R26, R27].

---

## 9. Could not verify / open items

Items a reader should not cite from this document without going back to the source:

1. **Baltruschat journal licence** (CC BY assumed for Scientific Reports) — publisher page redirected to a cookie gate; numbers were taken from arXiv v2, which is the accepted manuscript.
2. **Wang 2017's second number set** (0.784 / 0.706 / 0.671 / 0.806) quoted by Yao and CheXNet — revision of origin not traced.
3. **arnoweng/CheXNet preprocessing** (256 → TenCrop 224) — from community knowledge of `read_data.py`, not read here.
4. **jrzech/reproduce-chexnet split ratio** — README does not state it.
5. **jfhealthcare uncertain-label policy** and augmentation contents of `"Aug"` — in code, not in the config we read.
6. **torchxrayvision per-pathology NIH AUROC** — not tabulated in the paper or docs; models page has no table.
7. **VinBigData 1st-place write-up** — Kaggle page not readable without a browser session; 2nd/3rd-place repos lack licence files.
8. **RSNA 1st place LR/epochs/augmentation** — not in README; the AJR paper [R18] may contain them.
9. **KAD fine-tuning split** (official vs random) for the 0.825 vs 0.804 comparison.
10. **CheXzero pneumothorax AUROC 0.764** — appears only in a secondary blog/search snippet.
11. **MPS autocast support** — PyTorch stable AMP docs do not list `mps`; the GitHub issue tracker says fp16/bf16 work (bf16 needs macOS 14+). Benchmark before relying on it.
12. **Ark+ (Nature 2025) per-dataset AUROC** — full text behind the publisher gate; only the abstract (PubMed) and the MICCAI 2023 tables were read.
13. **GPU throughput estimate** (>150 img/s at 448 with AMP) in §8.9 is our own guess.

---

## 10. References (all accessed 2026-09-21)

- **[R1]** Wang X. et al., "ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases", CVPR 2017. https://arxiv.org/abs/1705.02315 (PDF read locally: split text §"CNN Setting", Table "ResNet-50 ChestX-ray8 / ChestX-ray14").
- **[R2]** Yao L. et al., "Learning to diagnose from scratch by exploiting dependencies among labels", arXiv 2017. https://arxiv.org/abs/1710.10501 (§4.1 split, §4.3 training, Table 2).
- **[R3]** Rajpurkar P. et al., "CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning", arXiv 1711.05225 v3. https://arxiv.org/abs/1711.05225 (§2.2, §3.1, §5, Table 2).
- **[R4]** Guendel S. et al., "Learning to recognize Abnormalities in Chest X-Rays with Location-Aware Dense Networks", arXiv 1803.04565 (CIARP 2018). https://arxiv.org/abs/1803.04565 (§3.1 loss/optimiser, §3.2 1024 input, Table 1).
- **[R5]** Baltruschat I. M. et al., "Comparison of Deep Learning Approaches for Multi-Label Chest X-Ray Classification", Scientific Reports 9:6381 (2019). arXiv: https://arxiv.org/abs/1803.02315 ; journal: https://www.nature.com/articles/s41598-019-42294-8 (journal page unreadable through the publisher redirect; arXiv v2 PDF read locally: §3 Implementation, Table 3, Table 5, §4 Grad-CAM/drains).
- **[R6]** Seyyed-Kalantari L. et al., "CheXclusion: Fairness gaps in deep chest X-ray classifiers", arXiv 2003.00827 (PSB 2021). https://arxiv.org/abs/2003.00827 (§4.1 models, Table 2). Code: https://github.com/LalehSeyyed/CheXclusion (no LICENSE file per GitHub API).
- **[R7]** Irvin J. et al., "CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison", AAAI 2019. https://arxiv.org/abs/1901.07031 (§"Uncertainty approaches" table; training details §"Training procedure").
- **[R8]** Cohen J. P. et al., "TorchXRayVision: A library of chest X-ray datasets and models", arXiv 2111.00595 (MIDL 2022). https://arxiv.org/abs/2111.00595 ; repo https://github.com/mlmed/torchxrayvision ; LICENSE https://raw.githubusercontent.com/mlmed/torchxrayvision/master/LICENSE (Apache-2.0) ; models page https://mlmed.org/torchxrayvision/models.html.
- **[R9]** Cohen J. P. et al., "On the limits of cross-domain generalization in automated X-ray prediction", MIDL 2020. https://arxiv.org/abs/2002.02497 (masked loss, augmentation, 224 input).
- **[R10]** arnoweng/CheXNet. https://github.com/arnoweng/CheXNet (README AUROC table; no LICENSE per GitHub API).
- **[R11]** zoogzog/chexnet. https://github.com/zoogzog/chexnet (README AUROC table; no LICENSE per GitHub API).
- **[R12]** jrzech/reproduce-chexnet. https://github.com/jrzech/reproduce-chexnet (BSD-3-Clause; README AUROC table).
- **[R13]** jfhealthcare/Chexpert. https://github.com/jfhealthcare/Chexpert ; licence https://raw.githubusercontent.com/jfhealthcare/Chexpert/master/LICENSE-2.0.txt (Apache-2.0) ; config https://raw.githubusercontent.com/jfhealthcare/Chexpert/master/config/example.json.
- **[R14]** stanfordmlgroup/chexpert-labeler. https://github.com/stanfordmlgroup/chexpert-labeler (MIT per GitHub API).
- **[R15]** Project-MONAI/model-zoo https://github.com/Project-MONAI/model-zoo (Apache-2.0; `models/` listing on `dev`) ; Project-MONAI/tutorials https://github.com/Project-MONAI/tutorials (Apache-2.0; `2d_classification/`).
- **[R16]** NVIDIA NGC, `clara_xray_classification_chest_no_amp`. https://catalog.ngc.nvidia.com/orgs/nvidia/teams/med/models/clara_xray_classification_chest_no_amp.
- **[R17]** sneddy/pneumothorax-segmentation (SIIM-ACR 2019 1st place). https://github.com/sneddy/pneumothorax-segmentation (MIT; README).
- **[R18]** i-pan/kaggle-rsna18 (RSNA 2018 1st place). https://github.com/i-pan/kaggle-rsna18 (MIT) ; Pan I. et al., AJR 2019, https://www.ajronline.org/doi/10.2214/AJR.19.21512.
- **[R19]** VinBigData 2021: 1st-place write-up https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection/writeups/s-1st-place-solution (not readable without JS) ; 3rd place https://github.com/Scu-sen/VinBigData-Chest-X-ray-Abnormalities-Detection (README; no LICENSE) ; 2nd place https://github.com/ZFTurbo/2nd-place-solution-for-VinBigData-Chest-X-ray-Abnormalities-Detection (no LICENSE).
- **[R20]** Holste G. et al., "Towards long-tailed, multi-label disease classification from chest X-ray: Overview of the CXR-LT challenge", arXiv 2310.16112 (Med. Image Anal. 2024). https://arxiv.org/abs/2310.16112.
- **[R21]** Holste G. et al., "CXR-LT 2024: A MICCAI challenge on long-tailed, multi-label, and zero-shot disease classification from chest X-ray", arXiv 2506.07984. https://arxiv.org/abs/2506.07984.
- **[R22]** Chambon P. et al., "CheXpert Plus", arXiv 2405.19538. https://arxiv.org/abs/2405.19538 ; https://github.com/Stanford-AIMI/chexpert-plus.
- **[R23]** Tiu E. et al., "Expert-level detection of pathologies from unannotated chest X-ray images via self-supervised learning", Nat. Biomed. Eng. 2022. PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC9792370/ ; code https://github.com/rajpurkarlab/CheXzero (MIT).
- **[R24]** microsoft/BiomedCLIP model card. https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 (MIT; intended-use statement).
- **[R25]** microsoft/rad-dino model card https://huggingface.co/microsoft/rad-dino (MIT; training-data table) ; Pérez-García F. et al., arXiv 2401.10815 https://arxiv.org/abs/2401.10815 (Nat. Mach. Intell. 2025).
- **[R26]** Ma D. et al., "Foundation Ark", MICCAI 2023, arXiv 2310.09507 https://arxiv.org/abs/2310.09507 (Table 1 splits, Table 2 AUROC, Table 5 setup) ; Ma D. et al., "A fully open AI foundation model applied to chest radiography", Nature 643:488–498 (2025), https://www.nature.com/articles/s41586-025-09079-8 (abstract via PubMed 40500447) ; repo https://github.com/jlianglab/Ark ; licence https://raw.githubusercontent.com/jlianglab/Ark/main/LICENSE ("solely for non-commercial use").
- **[R27]** Zhang X. et al., "Knowledge-enhanced visual-language pre-training on chest radiology images", Nat. Commun. 14:4542 (2023); arXiv https://arxiv.org/abs/2302.14042 ; code https://github.com/xiaoman-zhang/KAD (MIT).
- **[R28]** Majkowska A. et al., "Chest Radiograph Interpretation with Deep Learning Models: Assessment with Radiologist-adjudicated Reference Standards and Population-adjusted Evaluation", Radiology 294(2):421–431 (2020). https://pubs.rsna.org/doi/10.1148/radiol.2019191293 (abstract via Europe PMC / PubMed 31793848) ; label release: https://docs.cloud.google.com/healthcare-api/docs/resources/public-datasets/nih-chest (four_findings_expert_labels; all_findings_expert_labels, Nabulsi et al. Sci Rep 2021).
- **[R29]** Oakden-Rayner L., "Exploring Large-scale Public Medical Image Datasets", Academic Radiology 27(1):106–112 (2020); arXiv 1907.12720 https://arxiv.org/abs/1907.12720 (PDF read locally: Table 2, §3.1).
- **[R30]** PyTorch AMP documentation https://docs.pytorch.org/docs/stable/amp.html (autocast op lists; `binary_cross_entropy` unsafe) ; MPS autocast bf16 issue https://github.com/pytorch/pytorch/issues/139386 (secondary).
- **[R31]** Licence fields were read from the GitHub REST API (`https://api.github.com/repos/<owner>/<repo>`, field `license.spdx_id`) on 2026-09-21; `null` means no recognised LICENSE file.
