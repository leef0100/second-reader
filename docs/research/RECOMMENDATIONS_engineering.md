For research use only. Not for diagnostic use.

# Engineering recommendations from the open-source survey

> Prioritised, file-mapped changes for this repository, derived from
> `open_source_training.md`, `open_source_deployment.md` and
> `open_source_calibration_ood_xai.md` (same folder). Every number below was
> checked against a primary source on 2026-09-21; anything that could not be
> verified is marked **unverified**. Licence verdicts: MIT / Apache-2.0 / BSD =
> code may be adopted; GPL / AGPL / CC-NC / "research only" = ideas only.

## 0. Where the repo stands today (so the deltas are concrete)

| Aspect | Current implementation | File |
|---|---|---|
| Backbone / init | `timm` DenseNet-121, ImageNet init, `drop_rate 0.2`, linear finding + view heads | `src/secondreader/modalities/cxr/model.py`, `config.py` |
| Input | percentile window 0.5/99.5 → letterbox 320² → 3-ch ImageNet mean/std | `preprocess.py`, `configs/experiments/cxr_nih_baseline.yaml` |
| Augmentation | `RandomAffine(7°, translate 5 %, scale 0.9–1.1)` + brightness/contrast jitter ±0.1 (p = 0.8); no flip, no crop | `data.py` (`ManifestDataset.__init__`, `__getitem__`) |
| Loss / imbalance | masked BCE (D-017) + optional `pos_weight = n_neg/n_pos` (D-026, on in `cxr_nih_baseline.yaml`) + 0.3 × view CE | `train.py` (`masked_bce`, `view_ce`) |
| Optimiser | AdamW 1e-4, wd 1e-4, OneCycle (pct_start 0.1), grad-clip 5, batch 16, 8 epochs, early stop patience 3 on mean val AUROC | `train.py`, `config.py::TrainConfig` |
| Precision / device | fp32 on MPS (autocast measured slower on M1 Pro, D-025); ~27 img/s → ~69 min/epoch at 320 px | `docs/reports/throughput_m1pro.md` |
| Split | `nih-v1`: hashed patient-level 80/10/10 (train 90,036 / val 10,637 / test 11,447) — **not** NIH's official `train_val_list.txt` / `test_list.txt` (which are on disk: 86,523 / 25,595 lines in `data/raw/nih_cxr14/`) | `splits.py`, `runs/nih-v1/pipeline.log` |
| Calibration | per-finding 1-D temperature (bounded 0.25–4) + Youden / sens-90 threshold on val; 15 equal-width-bin ECE | `core/eval/metrics.py`, `train.py` |
| Gate | header → pixel → view head + class-agnostic Mahalanobis (Ledoit-Wolf) at val 99th percentile (D-019) | `gate.py` |
| Saliency | hand-rolled Grad-CAM on `forward_features`; box = largest component ≥ 0.5 × max | `saliency.py` |
| Serving | Orthanc A → webhook → FastAPI → Postgres job queue → plugin → highdicom SR/SC → Orthanc B / OHIF | `core/serving/*`, `core/results/writer.py` (D-002/D-004/D-005) |

## 1. The benchmark our NIH numbers must be read against

The single most important fact from the survey: **NIH ChestX-ray14 AUROCs are
not comparable across papers unless the split is the same**, and the official
split is systematically harder than random patient splits. The cleanest
evidence is Guendel et al. 2018, who ran *the same* location-aware DenseNet-121
(1024 px) on both:

| Finding | Wang 2017, official split (ResNet-50) | Guendel 2018, official split | Guendel 2018, **random patient split, same model** | Baltruschat 2019, official (ResNet-38-large-meta, 448 px) | Baltruschat 2019, random 5-fold (ResNet-50 FT, 224 px) | CheXNet 2017, random split (DenseNet-121, 224 px) | torchxrayvision `resnet50-res512-all`, own 20 % split |
|---|---|---|---|---|---|---|---|
| Pneumothorax | 0.799 | 0.846 | 0.872 | 0.840 | 0.870 ± 0.8 | 0.889 | 0.84 |
| Effusion | 0.759 | 0.828 | 0.885 | 0.822 | 0.871 ± 0.2 | 0.864 | 0.86 |
| Nodule | 0.669 | 0.758 | 0.774 | 0.747 | 0.726 ± 0.9 (0.751 at 448 px) | 0.780 | 0.76 |
| Mass | 0.693 | 0.821 | 0.854 | 0.820 | 0.822 ± 1.0 (0.839 at 448 px) | 0.868 | 0.82 |
| Mean (14) | 0.745 | 0.807 | 0.841 | 0.806 | 0.817 | 0.841 | — |

Sources: Wang et al. 2017 Table 17 (arXiv 1705.02315 PDF, "AUCs … using
published data split"); Guendel et al. 2018 Table 1 (arXiv 1803.04565 PDF —
also reproduces Wang's official numbers); Baltruschat et al. 2019 Tables 3 and 5
(arXiv 1803.02315 PDF / Sci Rep 9:6381); CheXNet numbers as reproduced in
https://github.com/arnoweng/CheXNet (paper column: 0.8887 / 0.8638 / 0.7802 /
0.8676; mean 0.841); torchxrayvision
https://github.com/mlmed/torchxrayvision/blob/master/BENCHMARKS.md (validated on
"20 % of each dataset", not the official split). All accessed 2026-09-21.

Reading of the table:

* The "typical" ranges in the task brief (pneumothorax ~0.85–0.90, effusion
  ~0.87–0.90, nodule ~0.75–0.80, mass ~0.80–0.86) are **confirmed for random
  patient splits** (CheXNet, Baltruschat 5-fold, Guendel-random). On the
  **official split** the bar is lower: pneumothorax ~0.84–0.85, effusion
  ~0.82–0.83, nodule ~0.75–0.76, mass ~0.82 (Guendel, Baltruschat).
* Our `nih-v1` split is a random patient split, so a DenseNet-121 at 320 px
  should land near the CheXNet / Baltruschat-FT column: roughly **0.86–0.88
  pneumothorax, 0.86–0.88 effusion, 0.73–0.78 nodule, 0.82–0.86 mass**. Note
  our `nodule_mass` target is the union (D-012), which has no literature
  reference row; expect it between the nodule and mass numbers.
* Guendel attribute the official-split gap to "large label inconsistency" and
  "on average 3 times more images per patient in the test set" (arXiv
  1803.04565 §4) — i.e. the official test set is heavy on follow-up exams of
  sick in-patients. That is closer to a deployment population than a random
  split is.

## 2. Prioritised recommendations (12)

Effort: S ≤ 1 day, M ≤ 1 week, L > 1 week. "Effect" is the expected direction,
not a promise; the repo's own eval report is the arbiter.

### R1. Add the official NIH split as a first-class tier and report it alongside `nih-v1` — **Priority 1, effort S**

* **What.** Add `--official nih` to `secondreader cxr split` that assigns rows
  listed in `data/raw/nih_cxr14/test_list.txt` to `test` and splits
  `train_val_list.txt` into train/val by the existing patient hash. Write the
  resulting manifest hash into `DECISIONS.md` like any other locked split.
* **Why.** Without it, no number we publish can be compared to Wang/Guendel/
  Baltruschat/CheXclusion, and the official test set is the only NIH subset for
  which radiologist-adjudicated labels exist (R5). Baltruschat: "performance of
  a network depends significantly on the selected split" (arXiv 1803.02315 §1).
* **Files.** `src/secondreader/modalities/cxr/splits.py` (`build_splits`: new
  `fixed_test_ids: set[str] | None` argument), `cli.py` (flag), `adapters.py`
  (no change — `image_id` already equals the NIH file name), `datasets/
  registry.yaml` (`official_split_files:`), `DECISIONS.md` (new D-0xx).
* **Effect.** Comparability; expect *lower* absolute AUROC (~-0.03 mean, up to
  -0.06 on effusion per Guendel Table 1) — say so in the report notes so it is
  not read as a regression. Keep `nih-v1` for internal model selection; use the
  official tier as a reported benchmark, never for tuning.
* **Licence.** NIH split files are part of the NIH release ("usage … is
  unrestricted", registry entry). Nothing to adopt.

### R2. Next run at 512 px (same DenseNet-121), on the cloud GPU — **Priority 2, effort S config / M compute**

* **What.** New `configs/experiments/cxr_nih_512.yaml` with
  `preprocess.image_size: 512`, rebuild the cache with `--size 512`
  (≈ 2.56 × the 320-px cache, ~30 GB for NIH), everything else unchanged so the
  comparison is one-variable.
* **Why.** Our two hardest targets are the resolution-sensitive ones. Sabottke &
  Spieler 2020 (Radiology: AI, NIH, ResNet-34/DenseNet-121): nodule AUC 0.689 at
  64 px vs 0.854 at 320 px and mass 0.767 → 0.886; maxima "between 256 × 256 and
  448 × 448" (https://pmc.ncbi.nlm.nih.gov/articles/PMC8017385/). Two 2025–26
  studies push the sweet spot to 512: "Sharpness matters" (PMC13365246;
  SIIM-ACR pneumothorax, DenseNet-121/ResNet-152) found external pneumothorax
  AUROC best at 512 (DenseNet 0.91) with low-resolution models dropping > 20 %
  externally, and recommends "512 × 512 … as a reasonable balance"; the DINOv3
  resolution-scaling study (arXiv 2510.07191, 816 k radiographs) finds gains
  "concentrated on small focal and boundary-dependent abnormalities" at 512 and
  that 1024 "rarely improved performance". CXR-LT 2023 top-3 used 448–1024
  (PMC11365790). Baltruschat's own 224 → 448 change moved nodule 0.726 → 0.751.
* **Files.** `configs/experiments/cxr_nih_512.yaml`; no code change.
  `gate.min_short_side` stays 256 (a gate, not a model constraint).
* **Effect.** Accuracy: expect the largest gain on `nodule_mass` (+0.01–0.03
  AUROC per the citations), smaller on pneumothorax; effusion ~flat. Speed:
  ~2.6 × slower per epoch — on the M1 Pro that is ~3 h/epoch, so run it on the
  cloud GPU with R7. Robustness: the external-generalisation argument in
  "Sharpness matters" is the reason this outranks a backbone change.
* **Licence.** n/a.

### R3. Backbone trial: ConvNeXt-Tiny (ImageNet-1k/22k) as the second run, DenseNet-121 stays the reference — **Priority 3, effort S**

* **What.** `model.backbone: convnext_tiny.fb_in22k_ft_in1k` (timm name;
  weights from Meta's ConvNeXt release, MIT) at 512 px. Checked in the
  installed timm 1.0.29 (`.venv/.../timm/models/convnext.py`): the tag exists
  (`timm.list_pretrained('convnext_tiny*')` also lists
  `convnext_tiny.dinov3_lvd1689m` — natural-image SSL, licence **unverified**),
  `forward_features` / `forward_head(pre_logits=True)` exist, so
  `CxrNet.feature_maps` / `head_from_maps` work unchanged; the ConvNeXt head is
  `pool → LayerNorm → fc` (`NormMlpClassifierHead`), whereas DenseNet's is
  `pool → dropout → fc` — see R9 for why that matters for the CAM.
* **Why.** All CXR-LT 2023 top-3 teams used CNNs, two of them ConvNeXt
  (ConvNeXt-S at 1024, ConvNeXt-B at 448; PMC11365790); the DINOv3 study found
  "ConvNeXt-B remained superior to ViT-B/16 under both full and
  parameter-efficient adaptation" (arXiv 2510.07191). Against over-sizing:
  Bressem et al. 2020 (CheXpert, 15 architectures) — "smaller networks such as
  ResNet-34 … have the potential to classify chest radiographs as precisely as
  … DenseNet-201 or ResNet-151" (arXiv 2002.08991), and Baltruschat found a
  *shallower* ResNet-38 beat ResNet-50/101 on the official split (0.806 vs
  0.795 / 0.785). So: Tiny, not Base.
* **Files.** `configs/experiments/cxr_nih_512_convnext.yaml`; possibly
  `model.py` if `forward_head` differs; `saliency.py` unchanged (Grad-CAM on the
  last stage's map is fine; see R9 for the CAM variant).
* **Effect.** Accuracy +0.00–0.02 mean AUROC (**unverified** for our targets);
  throughput similar to DenseNet-121 at the same resolution on CUDA; MPS
  throughput **unverified** (run `secondreader cxr bench model`).
* **Licence.** timm code Apache-2.0
  (https://github.com/huggingface/pytorch-image-models); ConvNeXt weights MIT
  (https://github.com/facebookresearch/ConvNeXt). Do **not** pick timm weights
  tagged `fb_swsl`/`fb_ssl`/`wsl` — those are CC-BY-NC 4.0 per timm's docs.

### R4. Augmentation policy: modest geometric + photometric set, torchvision v2 only — **Priority 4, effort S**

* **What.** In `data.py` replace the single `RandomAffine` with:
  `RandomResizedCrop(size, scale=(0.80, 1.0), ratio=(0.9, 1.1))` (applied to the
  letterboxed frame), `RandomAffine(degrees=7, translate 0.03)`, horizontal flip
  p = 0.5, brightness/contrast ±0.15, random gamma 0.8–1.25, optional light
  Gaussian blur p = 0.1. Keep augmentation off for val/OOD/drift passes (already
  the case). Add `train.augment_policy: none|basic|strong` to `TrainConfig` so
  the policy is in the content hash.
* **Why.** Every strong open pipeline uses crop + rotation + flip + intensity:
  Baltruschat (Szegedy-style crops 8–100 % area, ±7°, horizontal flip; arXiv
  1803.02315 §3), CheXNet (horizontal flip, arnoweng README), SIIM-ACR 1st place
  ("HorizontalFlip, OneOf[RandomContrast, RandomGamma, RandomBrightness],
  OneOf[Elastic, GridDistortion, OpticalDistortion], ShiftScaleRotate";
  https://github.com/sneddy/pneumothorax-segmentation, MIT), CXR-LT winners
  ("standard" augmentation; team 2 used mosaic). Horizontal flip is safe for a
  classifier trained on a laterality-agnostic target (all three of ours), and the
  SR box is produced from the *un-flipped* inference image. Elastic/grid
  distortions are skipped: they were tuned for segmentation and can fabricate
  pleural-line-like edges.
* **Files.** `src/secondreader/modalities/cxr/data.py` (`_aug`), `config.py`.
* **Effect.** Small accuracy gain, larger robustness gain on external sets
  (evidence is indirect — Kaggle/CXR-LT ablations are not per-finding).
* **Licence.** torchvision BSD-3 — fine. **Do not** move to `albumentations`:
  the MIT repo was archived on 2025-07-10 and its successor AlbumentationsX is
  AGPL-3.0 / commercial (https://github.com/albumentations-team/albumentations
  README) — incompatible with this Apache-2.0 repo.

### R5. NIH label noise: add the Google adjudicated labels as a clean-label evaluation subset and a `label_quality` subgroup axis — **Priority 5, effort M**

* **What.** (a) Request the "four findings" expert labels (validation 2,412 +
  test 1,962 images; pneumothorax, nodule/mass, airspace opacity, fracture;
  three-radiologist adjudication with HEDGE values) via the Google form linked
  from https://docs.cloud.google.com/healthcare-api/docs/resources/public-datasets/nih-chest,
  store under `data/raw/nih_cxr14/four_findings_expert_labels/`, register in
  `datasets/registry.yaml` (`nih_cxr14_adjudicated`, inherits NIH's unrestricted
  terms per that page, `commercial_ok: true`). (b) New adapter
  `nih_adjudicated_manifest()` mapping YES/NO/HEDGE → 1/0/-1 for
  `pneumothorax` and `nodule_mass` (effusion is not covered). (c) In
  `evaluate.py` add a subgroup axis `label_quality ∈ {nlp, adjudicated}` and
  print the adjudicated rows as their own table. (d) Do **not** change the
  training loss yet — masking (D-017) already handles HEDGE.
* **Why.** Oakden-Rayner 2020: ChestX-ray14 label PPVs "mostly between 10 % and
  30 % lower" than documented, and "in the pneumothorax class, 80 % of the
  positive cases have chest drains" with PPV 60 % for drain-free cases
  (Academic Radiology 27:106, arXiv 1907.12720). Majkowska et al. 2020 built
  the adjudicated set precisely for this and report AUROC 0.94 pneumothorax /
  0.91 nodule-or-mass on it (Radiology 294:421, PubMed 31793848). Because the
  adjudicated images are drawn from the official validation/test lists, R1 is a
  prerequisite. The chest-drain shortcut also argues for the CANDID-PTX
  external set (chest-tube labels) already planned in `docs/PLAN.md`.
* **Files.** `adapters.py`, `evaluate.py`, `datasets/registry.yaml`,
  `datasets/ACQUISITION.md`, `core/eval/report.py` (new axis is generic).
* **Effect.** No accuracy change; honest evaluation. A large NLP-vs-adjudicated
  gap on pneumothorax is the expected signature of the drain shortcut and should
  be reported, not hidden.
* **Licence.** Labels inherit NIH's no-restriction policy (stated on the Google
  page); citation of Majkowska 2020 required.

### R6. D-007 clarification: define "contaminating weights" by dataset overlap, not by modality — **Priority 6, effort S (decision text only)**

* **What.** Amend D-007 to: *"The externally evaluated model may not initialise
  from weights pre-trained (supervised or self-supervised) on any dataset that
  appears in `datasets/registry.yaml` as `external_test`, nor on any dataset
  whose images are re-released inside one (SIIM-ACR, RSNA, ChestX-Det). Weights
  trained only on natural images (ImageNet, LVD-142M/DINOv2) are permitted.
  `model_card.init` must name the weight source and its training corpora."*
* **Why — with the concrete verdicts:**
  * **RAD-DINO** (Microsoft, MIT licence): trained on MIMIC-CXR 368,960 +
    CheXpert 223,648 + NIH-CXR 112,120 + **PadChest 136,787 + BRAX 41,260**
    (882,775 images) and "shared for research purposes only … not meant to be
    used for clinical practice" (https://huggingface.co/microsoft/rad-dino).
    PadChest and BRAX are our external tests #2/#3 → **not allowed** under
    D-007 as amended, independent of its permissive code licence.
  * **Ark / Ark+** (JLiangLab, Nature 2025): pretrained on ChestX-ray14, RSNA,
    CheXpert, **VinDr-CXR**, Shenzhen, MIMIC (PubMed 41319624;
    https://www.nature.com/articles/s41586-025-09079-8) → **not allowed**
    (VinDr is external test #1).
  * **torchxrayvision `*-all`** (Apache-2.0): NIH + PadChest + CheXpert + MIMIC
    (+ others) → **not allowed** (already the D-007 example).
  * **BiomedCLIP** (MIT): trained on PMC-15M PubMed-Central figure/caption pairs,
    not on any CXR dataset as such (https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224).
    Letter-of-the-rule clean, but PubMed figures do reproduce sample images from
    NIH/CheXpert/PadChest papers (**unquantified**), and a 2026 study found
    frozen CLS embeddings of RAD-DINO/DINOv2/BiomedCLIP/MedSigLIP "sat at the
    chance floor (AUC 0.500–0.524)" for small-lesion perturbations unless
    patch-level probes are used (arXiv 2606.11606). **Not recommended** as an
    encoder for our nodule/mass target; acceptable for gate-only experiments.
  * **DINOv2** (Meta): code and standard weights Apache-2.0, pre-trained on
    "142 M images without using any labels" — curated natural images, no
    medical data (https://github.com/facebookresearch/dinov2). **Allowed**; the
    `XRay-DINO` variant in the same repo is FAIR non-commercial → ideas only.
    **DINOv3** licence and corpus **unverified** here (custom "DINOv3 License");
    check before use.
  * Evidence that in-domain pretraining helps is real (CXR-LT 2023: "multi-stage
    pretraining (ImageNet → domain CXRs) outperformed ImageNet-only") — the
    honest route is **our own** self-supervised pretraining on NIH train +
    CheXpert train (both internal), which D-007 already permits.
* **Files.** `DECISIONS.md` (D-007 amendment), `model.py::save_bundle`
  (`init` currently "imagenet"/"random" → free-text provenance string),
  `docs/INTENDED_USE.md` if wording refers to D-007.
* **Effect.** Robustness of the evaluation claim; no accuracy change.
* **Licence.** Decision text only.

### R7. Mixed precision and larger batches on CUDA only; keep fp32 on MPS — **Priority 7, effort S**

* **What.** In `train.py` wrap forward/loss in
  `torch.autocast("cuda", dtype=torch.bfloat16)` when `device.type == "cuda"`
  and the GPU supports bf16 (Ampere+); no `GradScaler` is needed for bf16 (only
  fp16). Add `train.amp: auto|off` to `TrainConfig` (part of the config hash).
  With the 512-px run use batch 32–48 and scale `lr` linearly, keep OneCycle.
  Enable `channels_last` on CUDA only.
* **Why.** PyTorch AMP docs: wrap the forward pass in `torch.autocast`, run
  `backward` outside; bf16 "only uses torch.autocast" (no scaler)
  (https://docs.pytorch.org/docs/2.14/amp.html). Our own bench showed
  autocast/`channels_last` are *slower* on MPS (D-025,
  `docs/reports/throughput_m1pro.md`), so gate the flag on device type rather
  than re-adding a global knob. Guendel trained at batch 128 to keep positives
  in every batch — relevant to `pos_weight` stability at 5 % prevalence.
* **Files.** `train.py`, `config.py`, `bench.py` (add the CUDA case to the
  throughput report).
* **Effect.** Speed 1.5–3 × on CUDA (**unverified** on the chosen cloud GPU);
  no accuracy change expected with bf16.
* **Licence.** n/a.

### R8. Optimiser details: discriminative LR for heads, 12–15 epochs at 512, select on the three targets — **Priority 8, effort S**

* **What.** Two param groups (`backbone` 1e-4, `findings`/`view` heads 1e-3),
  `epochs: 12`, `early_stop_patience: 3`, keep AdamW + OneCycle. Keep
  `pos_weight: auto`. Keep the 0.3 view-CE weight. Leave asymmetric/focal loss
  as an A/B only if pneumothorax sensitivity at the sens-90 operating point is
  unsatisfactory.
* **Why.** The classic recipes are Adam 1e-3 with reduce-on-plateau for 10–20
  epochs (CheXNet; Guendel: "initialized with 1e-3 and reduced 10 × when the
  validation loss plateaus", arXiv 1803.04565 §3.1; Baltruschat: Adam 1e-3
  transfer, halve on plateau). Ours is 10 × lower for the whole net, which is
  right for the pretrained trunk but slow for randomly initialised heads. CXR-LT
  winners used weighted asymmetric / focal losses for 26 long-tailed classes
  (PMC11365790); for three findings at 5–12 % prevalence `pos_weight` is the
  same mechanism with fewer knobs, and temperature scaling absorbs the logit
  shift (D-026).
* **Files.** `train.py` (`torch.optim.AdamW` param groups), `config.py`
  (`head_lr_multiplier: float = 10.0`), `configs/experiments/*.yaml`.
* **Effect.** Faster convergence; small accuracy gain (**unverified** for our
  setup).
* **Licence.** n/a.

### R9. Saliency: switch the CAM to HiResCAM (or offer it beside Grad-CAM) and evaluate boxes against NIH `BBox_List_2017.csv` — **Priority 9, effort S–M**

* **What.** (a) Make the CAM variant explicit: replace the
  `weights.mean(dim=(2,3)) * fmap` step in `saliency.py::grad_cam` with the
  element-wise HiResCAM product `(grads * fmap).sum(dim=1)` (one line). For the
  current DenseNet-121 path (feature map → global average pool → linear head)
  the two are mathematically the same explanation, so this is a no-op today;
  it matters the moment anything non-linear sits between the map and the head —
  timm's ConvNeXt head applies a LayerNorm *after* pooling (R3), where plain
  Grad-CAM is no longer guaranteed to reflect the computation. (b) Add a
  localisation evaluation (`secondreader cxr eval --boxes`) that computes
  hit-rate / IoU of `box_from_heatmap` against NIH's 984 boxes
  (`BBox_List_2017.csv`, already on disk; NIH-derived, so evaluation only,
  never a train signal). Alternatives (pytorch-grad-cam, MIT: GradCAM,
  HiResCAM, GradCAM++, ScoreCAM, LayerCAM, … with ViT support —
  https://github.com/jacobgil/pytorch-grad-cam) and the Saporta/Arun evidence
  are in `open_source_calibration_ood_xai.md`.
* **Why.** Draelos & Carin, "Use HiResCAM instead of Grad-CAM for faithful
  explanations of convolutional neural networks" (arXiv 2011.08891): HiResCAM
  "is guaranteed to highlight only the locations the model used to make each
  prediction", Grad-CAM is not in general. Saporta et al. 2022 (Nat Mach
  Intell) show CXR saliency localisation lags radiologist boxes substantially,
  so the box we put into the SR must be measured, not assumed.
* **Files.** `saliency.py`, `evaluate.py`, `adapters.py` (box loader),
  `core/eval/report.py` (localisation table).
* **Effect.** Faithfulness of the heatmap the clinician sees; no AUROC change.
* **Licence.** In-house change; `pytorch-grad-cam` is MIT if adopted instead
  (confirmed in the XAI survey).

### R10. OOD gate: keep Mahalanobis, add a second cheap score and validate the gate on labelled OOD sets — **Priority 10, effort M**

* **What.** Keep the D-019 Mahalanobis gate but (a) fit it on the *frontal
  training* features already extracted (done) with **relative Mahalanobis**
  (subtract a background Gaussian) or add the **energy** / **ViM** score from
  the logits, and combine by "either exceeds its 99th percentile"; (b) add an
  OOD evaluation command that reports rejection rate on Kermany paediatric
  (registry `ood_negative`), NIH laterals (there are none — use PadChest
  laterals when available), and synthetic corruptions (MedMNIST-C style blur /
  contrast / noise), so `gate.ood_percentile` is chosen with a measured
  false-reject / true-reject trade-off rather than a fixed 99.
* **Why.** Single-Gaussian penultimate-feature Mahalanobis is the standard
  baseline but is known to be weak on *near*-OOD (e.g. lateral views, portable
  AP with devices); OpenOOD benchmarks consistently rank ViM / KNN / relative
  Mahalanobis above plain Mahalanobis on near-OOD. Evidence and licences are in
  `open_source_calibration_ood_xai.md` (OpenOOD MIT, pytorch-ood Apache-2.0 —
  confirmed there).
* **Files.** `gate.py::OodModel` (+ `energy` field), `train.py` (fit second
  score), `evaluate.py`/`cli.py` (`ood-eval`), `DECISIONS.md` (D-019 update).
* **Effect.** Robustness: fewer confident predictions on inputs the model has
  never seen; a measurable false-reject rate to report per source.
* **Licence.** Ideas from MIT/Apache-2.0 code; implement in-house (few lines).

### R11. Calibration: keep per-finding temperature scaling, add equal-mass-bin ECE and a shift-aware caveat — **Priority 11, effort S**

* **What.** In `core/eval/metrics.py` add an adaptive (equal-mass) ECE next to
  the 15 equal-width bins, and report both; keep temperature scaling on
  internal validation (D-020) but print in the external report that
  temperatures were *not* refit on the external site (and offer
  `--recalibrate-on-site` for the PCCP-style per-site recalibration discussed in
  `docs/research/SUMMARY.md` rec 5).
* **Why.** Equal-width ECE at 5 % prevalence puts almost all mass in the first
  bin and hides miscalibration at the operating point; temperature scaling is
  known not to survive distribution shift (Ovadia et al. 2019, "Can you trust
  your model's uncertainty?"). Reference implementations (netcal, torch-
  uncertainty) and their licences are in `open_source_calibration_ood_xai.md`.
* **Files.** `core/eval/metrics.py`, `core/eval/report.py`, `evaluate.py`.
* **Effect.** Honest calibration reporting; no accuracy change.
* **Licence.** In-house; netcal/torch-uncertainty Apache-2.0 if adopted
  (confirmed in the survey).

### R12. Deployment: keep the Orthanc A/B + FastAPI + Postgres-queue design; borrow three things from MONAI Deploy / mercure / IHE AIR rather than replacing anything — **Priority 12, effort S–M**

* **What.** (a) Package the CXR plugin as a MONAI Application Package (MAP) *in
  addition to* the in-process plugin, so the same model can run under MONAI
  Deploy Express or mercure at a partner site; (b) adopt mercure's
  `task.json`-style per-job manifest (study UIDs, routing rule, model version,
  timestamps) as the payload of our `jobs` row for auditability; (c) emit a
  GSPS (Grayscale Softcopy Presentation State) with the box graphic beside the
  SR so viewers that ignore TID 1500 regions still show the box, moving us
  toward IHE AI Results (AIR) conformance. Verdicts per platform (MONAI Deploy
  Apache-2.0; mercure MIT; Kaapana AGPL = ideas only; Orthanc GPLv3 but used
  only over REST at a process boundary, which is fine) are in
  `open_source_deployment.md`.
* **Why.** The survey found no open platform that replaces our de-identify-
  before-storage boundary (D-002) — mercure and MONAI Deploy route DICOM to
  containers but leave de-identification and the review UI to you; Kaapana's
  licence blocks adoption. The gaps they expose in our design are packaging
  portability, job-manifest richness and viewer-independent result rendering.
* **Files.** `docker/` (MAP build), `core/serving/queue.py` (payload schema),
  `core/results/writer.py` (GSPS via highdicom, MIT), `DECISIONS.md` (D-002 /
  D-004 "complement, not replace" note).
* **Effect.** Portability and standards conformance; no accuracy change.
* **Licence.** highdicom MIT, MONAI Deploy App SDK Apache-2.0, mercure MIT
  (verified in the deployment survey).

## 3. Suggested run order

1. R1 (official split) + R6 (D-007 text) — half a day, no compute.
2. R7 (bf16 on CUDA) + R2 (512 px DenseNet-121) as one cloud run
   (`cxr_nih_512.yaml`), reporting `nih-v1` test **and** the official tier.
3. R4 (augmentation) + R8 (LR groups) as the next run, one change set, same
   split — compare to step 2 on the official tier.
4. R3 (ConvNeXt-T) only after 2–3, so the backbone is the only variable.
5. R5 (adjudicated labels) as soon as the Google form is answered — it changes
   what "good" means for pneumothorax.
6. R9–R12 in parallel with training runs; none of them changes the model.

## 4. Five highest-impact recommendations

1. **R1 — official NIH split as a reported tier.** Without it our numbers are
   incomparable and the adjudicated labels (R5) cannot be used.
2. **R2 — 512 px next run.** The best-supported accuracy lever for
   nodule/mass and for external pneumothorax generalisation.
3. **R6 — D-007 by dataset overlap.** RAD-DINO and Ark+ both saw our external
   sets (PadChest/BRAX, VinDr); DINOv2 does not. Write it down before someone
   "just tries" a foundation encoder.
4. **R5 — adjudicated-label evaluation subset.** 80 % of NIH pneumothorax
   positives carry a chest drain; the NLP-vs-adjudicated gap is the honest
   pneumothorax number.
5. **R4 + R8 — augmentation and LR groups.** Cheapest changes with the most
   consistent support across every strong open pipeline.

## 5. Uncertainty notes

* The literature ranges are for *NIH-only* training at 224–1024 px; adding
  CheXpert (as `cxr_baseline.yaml` intends) typically raises NIH AUROC further
  but no clean citation was found for our three targets — **unverified**.
* No paper reports a merged `nodule_mass` target on NIH; the row in §1 is an
  interpolation.
* ConvNeXt on MPS throughput, bf16 gains on the cloud GPU, and the timm
  `forward_head(pre_logits=True)` path for ConvNeXt were not measured here.
* DINOv3's licence/corpus and the exact contents of PMC-15M were not verified.
