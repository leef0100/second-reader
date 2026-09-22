For research use only. Not for diagnostic use.

# Open-source calibration, OOD detection, quality gates and explainability: what exists, what the evidence says, what we should adopt

> Due-diligence research for the Second Reader project. Every URL was accessed on 2026-09-21; bracketed numbers point to the reference list at the end. Licence verdicts use the repo's own LICENSE file or licence badge wherever it could be fetched; "unverified" means the primary source could not be read and the claim rests on a secondary source or on memory. Nothing here changes a decision on its own — see the "Implications" section for proposed follow-ups to `DECISIONS.md`.

**Licence legend.** *Adoptable* = MIT / Apache-2.0 / BSD (we may vendor or depend on code). *Ideas only* = GPL / AGPL / non-commercial / non-standard licence (we may read it and re-implement the idea, we may not copy code). *Data terms* = dataset licences, which bind how we may use the data, not the code.

**Files this maps onto.** `src/secondreader/core/eval/metrics.py` (per-output temperature scaling, 15 equal-width-bin ECE, Brier, bootstrap CIs, Youden and sensitivity-target thresholds), `src/secondreader/modalities/cxr/gate.py` (header → pixel → learned checks; view head; single class-agnostic Mahalanobis on penultimate features with Ledoit-Wolf covariance, 99th-percentile validation threshold, D-019), `src/secondreader/modalities/cxr/saliency.py` (hand-rolled Grad-CAM on `backbone.forward_features`, box = largest connected component above 0.5 × max), `src/secondreader/modalities/cxr/train.py` (where calibration and the OOD model are fitted on validation; D-020), `src/secondreader/modalities/cxr/model.py`, `src/secondreader/modalities/cxr/evaluate.py`, `src/secondreader/core/monitoring/drift.py` (PSI on input statistics and output distributions).

---

## 1. Calibration

### 1.1 Reference implementations

| Library | Repo | Licence (verified from) | Maturity | What it implements | Adopt? |
|---|---|---|---|---|---|
| **netcal** (calibration-framework) | github.com/EFS-OpenSource/calibration-framework [1] | Apache-2.0 (README badge) [1] | last release r1.3.6, 2024-08-08; ~380 stars [1][2] | Temperature scaling, Platt/logistic, beta calibration, histogram binning, isotonic, BBQ; metrics ECE/MCE/ACE/MMCE; regression calibration (variance scaling, GP methods); multi-class handled one-vs-all by default, no explicit multi-label API [1] | Adoptable. Heavy dependency set (PyTorch ≥ 2.3, Pyro, GPyTorch) [1] — take the binning/metric code, not the package, if we only need ECE variants. |
| **torch-uncertainty** | github.com/ENSTA-U2IS-AI/torch-uncertainty [3] | Apache-2.0 (README) [3] | ~528 stars; releases page shows v0.13.0 (2024-07-22) as latest visible — release cadence after that is unverified [3][4] | `TemperatureScaler`, `VectorScaler`, `MatrixScaler`, `DirichletScaler`, `IsotonicRegressionScaler`; metrics `CalibrationError`, `AdaptiveCalibrationError`, `SmoothCalibrationError`, `ClasswiseCalibrationError`, Brier, AURC/AUGRC; Lightning-based [3][5] | Adoptable, but Lightning-centric; the metric classes are the reusable part. |
| **gpleiss/temperature_scaling** | github.com/gpleiss/temperature_scaling [6] | MIT [6] | ~1.2k stars; author states the repo is unmaintained (written for PyTorch 0.3) and points to `probmetrics` [6] | `ModelWithTemperature`, LBFGS on NLL, ECE loss [6] | Reference only — our `fit_temperature` (bounded scalar NLL minimisation) is functionally the same thing per output. |
| **scikit-learn** `CalibratedClassifierCV` | scikit-learn.org [7][8] | BSD-3 [7] | 1.9.1 docs; `method="temperature"` added in 1.8 [8] | `sigmoid` (Platt), `isotonic` (non-parametric; docs advise against it with < ~1000 calibration samples because it overfits and introduces ties that alter AUC), `temperature`; `FrozenEstimator` wraps an already-fitted model [7][8] | Adoptable. Isotonic/Platt via `IsotonicRegression` / `LogisticRegression` on logits is a two-line alternative to our scalar T. |
| **MAPIE** | github.com/scikit-learn-contrib/MAPIE [9] | BSD-3-Clause [9] | ~1.6k stars; v1 API (2025) [9] | Conformal prediction sets/intervals for classification, regression, time series; risk control for multi-label and segmentation; needs scikit-learn ≥ 1.4 [9] | Adoptable. Not a calibration method — gives coverage-guaranteed sets, which is a different clinician-facing object from a probability. Park for later. |
| **betacal** | github.com/betacal/python [10] | MIT [10] | ~32 stars; small and stable [10] | Beta calibration (Kull, Silva Filho, Flach, AISTATS 2017): a 3-parameter family that contains the identity, so it cannot make a calibrated model worse the way Platt can [10][11] | Adoptable (tiny; easy to vendor). |
| **uncertainty-calibration** (verified_calibration) | github.com/p-lambda/verified_calibration [12] | MIT [12] | ~155 stars [12] | Kumar et al. 2019: debiased calibration-error estimator, scaling-binning calibrator, bootstrap CIs, top-label and per-class CE [12] | Adoptable; the debiased estimator is the cheapest fix to our ECE bias. |
| **relplot** (Smooth ECE) | github.com/apple/ml-calibration [13] | **Apple-specific licence**, not MIT/Apache/BSD: grants use/modify/redistribute but with Apple-specific terms and no patent grant [14] | ~191 stars; ICLR 2024 paper [13] | Kernel-smoothed reliability diagrams and smECE [13][15] | **Ideas only** until legal reads the licence; torch-uncertainty ships a `SmoothCalibrationError` under Apache-2.0 [5], which is the safer route. |

### 1.2 What the evidence says

**Temperature scaling (Guo et al. 2017).** Single-parameter scaling of logits, fitted by NLL on a held-out validation set, was "surprisingly effective" across datasets and beat vector and matrix scaling, which overfit; ECE was reported with 15 equal-width bins [16]. Our `fit_temperature` follows this recipe per output, with a bounded search in [0.25, 4] to stop T collapsing on tiny separable validation sets (a sensible guard the original code does not have) [6][16].

**Multi-label: per-output temperature vs vector scaling.** With independent sigmoid heads there is no softmax coupling, so "vector scaling" degenerates to one temperature (plus optionally one bias) per output, and "Platt scaling" is the two-parameter version of the same thing [16][7]. Our per-output scalar T is therefore the most conservative member of that family; adding a per-output bias (equivalent to sklearn `sigmoid`) or beta calibration (three parameters) buys flexibility at the cost of more overfitting risk on the low-positive findings (pneumothorax ≈ 5 % prevalence per D-026). Carse et al. 2022 (UNSURE/MICCAI, dermatology + histopathology) found "little or no advantage" in tuning the temperature with a calibration metric instead of NLL, and that temperature scaling on focal-loss-trained networks gave the best combination of calibration and accuracy [17]. That supports keeping NLL as the objective.

**ECE binning pitfalls.** Nixon et al. 2019 showed equal-width binned ECE is unstable — rankings of recalibration methods change with the binning — and recommended adaptive (equal-mass) bins, class-conditional evaluation and measuring all probabilities rather than only the top prediction [18]. Roelofs et al. (AISTATS 2022) quantified the *bias* of binned ECE and found equal-mass bins have lower bias than equal-width, proposing a debiased estimator and `ECE_sweep` (equal-mass, bin count grown while the calibration function stays monotone) [19]. Błasiok & Nakkiran (ICLR 2024) replaced bins with kernel smoothing (smECE) and showed binned ECE is discontinuous in the predictions [15]. Kumar et al. 2019 give a debiased plug-in estimator with bootstrap CIs [12]. **Our `expected_calibration_error` is exactly the 15-equal-width-bin, positive-class variant** these papers warn about; because our positives are rare, most bins above p ≈ 0.3 will be nearly empty on a per-finding basis, which is where equal-width bias is worst [18][19]. Metrics Reloaded (Nature Methods 2024) contains calibration recommendations in its supplementary notes, but the specific metric guidance could not be read from the open abstract — unverified beyond "calibration assessment is requested" [20].

**Calibration under shift.** Ovadia et al. (NeurIPS 2019) benchmarked methods on CIFAR-10-C, ImageNet-C, rotated/translated MNIST, text and genomics shifts. Their statement is direct: "while temperature scaling achieves low ECE for low values of shift, the ECE increases significantly as the shift increases, which indicates that calibration on the i.i.d. validation dataset does not guarantee calibration under distributional shift"; ensembles "consistently perform best across metrics" and "most of the gains of ensembling could be achieved with only 5 models" [21]. For us this means the per-source ECE tables mandated by D-010 are not optional decoration: they are the only place a site-specific miscalibration will show up, and the PSI output-distribution monitor in `drift.py` is the runtime proxy.

**CXR-specific calibration work.** Thin. CALM-CXR (MethodsX 2026) fits stage-specific temperatures on a 4,840-image paediatric pneumonia set and reports ECE ≈ 0.06–0.07, but it has no external validation and the authors say site transportability "remain[s] untested" [22]. A MICCAI 2022 paper proposes "weight scaling" for medical classifiers; its full text could not be fetched (paywall) — unverified [23]. Berger et al. 2021 incidentally show, on a CheXpert two-class task, that a baseline WRN was "reasonably calibrated" (ECE 0.04) and that ODIN's input perturbation destroys calibration (ECE 0.35 at τ = 1000) [24]. We found no CXR paper that reports per-finding ECE with CIs on an external multi-site test — which matches `competitors.md` §3: nobody publishes it.

### 1.3 Verdict on `metrics.py`

The temperature fit is sound and matches the reference implementation in spirit [6][16]. The ECE estimator is the weak point: replace or supplement equal-width bins with equal-mass bins and a debiased estimator [18][19][12], keep Brier (a proper scoring rule, unaffected by binning) as the headline calibration number, and add smECE only via an Apache-2.0 implementation [5]. Bootstrap CIs are already there; extend them to the debiased estimator.

---

## 2. Out-of-distribution detection

### 2.1 Method primer (post-hoc, no retraining)

| Score | Needs | Paper | Notes |
|---|---|---|---|
| MSP | softmax output | Hendrycks & Gimpel, ICLR 2017 [25] | Baseline. |
| MaxLogit | logits | Hendrycks et al. 2022 [26] | Beats MSP on large-scale and multi-label tasks (ImageNet-21K 2-label benchmarks) [26] — the only one of these evaluated on a *multi-label* setup. |
| Energy | logits | Liu et al., NeurIPS 2020 [27] | −T·logsumexp(z/T); −18 % FPR95 vs MSP on CIFAR-10 [27]. For sigmoid multi-label heads the natural analogue is per-output free energy, which is not what the paper evaluated — unverified transfer. |
| Mahalanobis (MDS) | penultimate features | Lee et al., NeurIPS 2018 [28] | Class-conditional means, *tied* covariance; original recipe also ensembles layers and perturbs inputs [28]. |
| Relative Mahalanobis (RMDS) | features | Ren et al. 2021 [29] | MD minus the MD to a single class-agnostic background Gaussian; "more robust to hyperparameter choice"; +15 % AUROC on genomics near-OOD [29]. |
| ViM | features + last-layer W, b | Wang et al., CVPR 2022 [30] | Residual norm off the principal subspace turned into a "virtual logit" and matched to the real logits; +4 % AUROC over best baseline on OpenImage-O [30]. |
| KNN | features (L2-normalised) | Sun et al., ICML 2022 [31] | k-th neighbour distance; no distributional assumption; −24.8 % FPR95 vs Mahalanobis-based SSD+ on ImageNet [31]. Needs the training feature bank at inference. |
| Mahalanobis++ | features | Mueller & Hein 2025 [32] | L2-normalise features before fitting the Gaussian; feature-norm variation violates the Gaussian assumption; consistent gains over 44 models [32]. |

### 2.2 Reference libraries

| Library | Repo | Licence (verified) | Maturity | Implements | Adopt? |
|---|---|---|---|---|---|
| **OpenOOD** | github.com/Jingkang50/OpenOOD [33] | MIT (LICENSE file) [34] | v1.5 (DMLR 2024), ~1.1k stars; 24 post-hoc methods incl. MSP, MLS, EBO, MDS, RMDS, ViM, KNN, ReAct, ASH, SCALE, GEN, NNGuide; live leaderboard [33][35] | Benchmark harness + postprocessors | Adoptable. Installs from git only; the `postprocessors/` files are small and self-contained — vendor the two or three we want. |
| **pytorch-ood** | github.com/kkirchheim/pytorch-ood [36] | Apache-2.0 (LICENSE file) [37] | v0.3.3 (2024-09-07), ~348 stars; 30+ detectors incl. Mahalanobis, RMD, Energy, ViM, KNN, MaxLogit, MSP, ReAct, ASH, SCALE; v0.3.2 added the OpenMIBOOD medical benchmark [36][38] | Detector classes with a `fit(loader)` / `predict(x)` API | Adoptable; the cleanest dependency of the three (torch, scipy, torchmetrics; sklearn optional for ViM/KNN) [36]. |
| **torch-uncertainty** | [3] | Apache-2.0 | see §1.1 | OOD *criteria* are output-based only: MaxLogit, Energy, MSP, Entropy, mutual information, variation ratio [5]. No Mahalanobis/ViM/KNN in the API docs at access time. | Not sufficient for feature-space OOD. |
| **OpenMIBOOD** | github.com/remic-othr/OpenMIBOOD [39] | not fetched — unverified | CVPR 2025 | Medical OOD benchmark (histopathology, endoscopy, brain MRI) over 24 OpenOOD postprocessors [39][40] | Use as evidence, not code, until licence is read. |

### 2.3 Benchmark evidence

**OpenOOD v1.5, ImageNet-1K, ResNet-50, AUROC (%)** — near-OOD / far-OOD [35]:

| MSP | MaxLogit | Energy | MDS | RMDS | ViM | KNN | ReAct | ASH |
|---|---|---|---|---|---|---|---|---|
| 76.0 / 85.2 | 76.5 / 89.6 | 75.9 / 89.5 | 55.4 / 74.3 | 77.0 / 86.4 | 72.1 / 92.7 | 71.1 / 90.2 | 77.4 / 93.7 | 78.2 / 95.7 |

CIFAR-10 (same source): MDS 84.2 / 89.7, RMDS 89.8 / 92.2, ViM 88.7 / 93.5, KNN 90.6 / 93.0, MSP 88.0 / 90.7 [35]. The authors' summary: "no single method consistently outperforms others across benchmarks" and "near-OOD remains more challenging than far-OOD" [35]. Note that plain MDS is the *worst* method on ImageNet-1K — the class-conditional tied-covariance estimate degrades at 1,000 classes and 2,048 dimensions — while RMDS, ViM and KNN all recover [35].

**OpenMIBOOD (CVPR 2025), medical near-OOD AUROC (%)** — MIDOG histopathology / PhaKIR endoscopy / OASIS-3 MRI [40]:

| MDS | RMDS | ViM | KNN | MaxLogit | Energy | MSP |
|---|---|---|---|---|---|---|
| 63.2 / 76.5 / 96.2 | 52.2 / 67.7 / 71.7 | 62.7 / 81.1 / 98.4 | 61.6 / 55.4 / 97.7 | 56.6 / 40.2 / 49.5 | 56.9 / 40.2 / 49.4 | 55.9 / 50.2 / 53.5 |

Multi-layer Mahalanobis ensembles (MDSEns) led near-OOD on all three (91.8 / 81.1 / 98.4) and hit 100 % far-OOD [40]. The authors conclude that "findings from broad-scale out-of-distribution benchmarks in natural image domains do not translate to medical applications" and that "methods that rely primarily on feature space information tend to perform better on medical images, while methods leveraging probabilities or logits show superior results on natural images" — attributing this to lower pixel-intensity variance and more cohesive feature spaces in medical images [40]. Two things matter for us: (i) the logit-based scores that look competitive on ImageNet (MaxLogit, energy) are close to chance on medical near-OOD, and (ii) RMDS — the "obvious" upgrade — was *worse* than plain MDS on all three medical benchmarks [40].

**Medical / CXR-specific evidence.**

- *Cao et al. 2020 (benchmark of medical OOD; CXR, fundus, histology).* Three OOD categories: wrong modality/organ, covariate shift, unseen disease. Methods do well on the first and "fail to recognize images close to the training distribution"; a simple binary classifier on the feature representation had the best average accuracy/AUPRC, and a Mahalanobis variant normalised per channel with means/covariance estimated on validation did well on CXR covariate shift [41].
- *Berger, Paschali, Glocker, Kamnitsas 2021 (UNSURE).* CIFAR-10 vs SVHN: Mahalanobis best (AUROC 0.984). CheXpert, mutually exclusive subsets, WRN-100×2: Setting 1 (ID cardiomegaly vs pneumothorax, OOD fracture) Mahalanobis AUROC **0.580** vs MSP 0.678, ODIN 0.841; Setting 2 (ID lung opacity vs effusion, OOD fracture/pneumonia) Mahalanobis 0.526, ODIN 0.862, deep ensemble 0.445. Their diagnosis: the ID classes are poorly separated in feature space and the OOD class overlaps them, so "fitting a Gaussian distribution to the ID embeddings is challenging"; dimensionality reduction did not help; ODIN's gain came from input perturbation, which wrecked calibration [24]. This is *unseen-disease* near-OOD — the hardest and least relevant case for our gate, whose job is view/anatomy/quality rejection — but it is the clearest warning that a Gaussian on CXR features is not automatically well-behaved.
- *Anthony & Kamnitsas 2023 (UNSURE; CheXpert).* Class-conditional Mahalanobis at multiple depths; "the optimum layer to apply Mahalanobis distance changes with the type of OOD pattern" (synthetic artefacts vs unseen pacemakers vs sex), and splitting the detector into branches at different depths improves robustness. Code MIT, 50 % of CheXpert frontal scans hand-labelled for support devices [42][43].
- *FRODO (Çallı, Murphy, Sogancioglu, van Ginneken, MIDL 2019).* Mahalanobis on intermediate activations of a frontal-CXR ResNet-50, tested on 14,821 OOD radiographs of "incorrect orientation or anatomy" (lateral, non-chest): AUC 0.99 [44]. This is precisely our gate's use case and the closest published precedent for D-019.
- *Wollek et al. 2022 ("A knee cannot have lung disease").* ChestX-ray14-trained model; OOD = IRMA, Bone Age, MURA, ImageNet. In-distribution voting AUC 0.999, Mahalanobis 0.982, MaxLogit 0.726, MaxEnergy 0.724 [45]. Code on GitLab under a non-standard "nonexclusive-distrib/1.0" licence — **ideas only** [45]. Again: feature-space Mahalanobis ≫ logit scores for far-OOD CXR.
- *Karimi, Farnia, Bae 2025 (J Imaging Inform Med).* GAN latent-space + KS test to flag lateral views from a MIMIC-CXR frontal model: 97.5 % accuracy; adults only; code on request, not public [46].
- *CheXseen (Rajpurkar group, 2021).* Models "falsely classify diseases outside of the subset (unseen diseases) as 'no disease'"; penultimate features can be re-used to detect unseen diseases with few labels [47]. Relevant to the intended-use boundary, not to the gate.
- *MOOD challenge (Zimmerer et al., IEEE TMI 2022; brain MRI + abdominal CT).* "All algorithms show a high variance for different anomalies, making it yet hard to recommend them for clinical practice" [48].
- *MedMNIST-C (Di Salvo et al. 2024).* Modality-specific corruption benchmark; dataset CC BY 4.0 except DermaMNIST-C (CC BY-NC); API on GitHub [49]. Useful as a template for a CXR corruption stress-test of the gate.
- *MedShift (Guo et al., IEEE JBHI 2023).* Unsupervised anomaly detectors to find shifted samples between an internal Emory CXR set and CheXpert/MIMIC without sharing data [50]. Code availability unverified.

### 2.4 Is D-019 a reasonable design, and what are the cheapest upgrades?

**Reasonable, with caveats.** A single class-agnostic Gaussian on penultimate DenseNet features is (a) the natural choice for a *multi-label* head, where class-conditional Gaussians are ill-defined without a partition of the data; (b) the same object FRODO used successfully for lateral/non-chest rejection [44]; (c) the "background" term of RMDS [29]. Ledoit-Wolf shrinkage is a sound fix for n < D (D = 1024 for DenseNet-121). The threshold-as-policy-knob with per-source rejection rates reported is exactly the honest framing the MOOD authors ask for [48]. The weaknesses: the evidence that this Gaussian is *well-fitted* on CXR features is mixed [24][41], and a single deep layer is the wrong depth for some artefact types [42].

**Cheapest upgrades, in order of cost:**

1. **L2-normalise features before fitting** (Mahalanobis++) — one line, no new dependency, addresses the feature-norm problem that makes the Gaussian assumption fail [32]. Not yet evaluated on medical data — unverified transfer; cheap to A/B on our validation rejection curves.
2. **Add MaxLogit (per-output max |logit|) as a second, logged score** — free, and the only score with multi-label evidence [26]; do not gate on it alone (near-chance on medical near-OOD [40], 0.73 on non-chest CXR [45]).
3. **KNN on the same normalised features** — non-parametric, robust on ImageNet [31] and OASIS-3 [40], but its PhaKIR result (55.4) shows it is not uniformly better than MDS; costs a feature bank in the bundle (N_train × 1024 float16 ≈ 200 MB per 100k images) and an inference-time search. pytorch-ood (Apache-2.0) has a drop-in [36].
4. **ViM** — best single non-ensemble method on OpenMIBOOD near-OOD and strong far-OOD on ImageNet [30][40]; needs last-layer W and b and a principal-subspace fit; ~40 lines in OpenOOD/pytorch-ood, both permissively licensed [34][37]. Its "virtual logit" is defined against a softmax head; for our sigmoid multi-label head the matching constant must be re-derived — flag as engineering, not copy-paste.
5. **Multi-depth Mahalanobis (MDSEns / MBM)** — the strongest medical evidence [40][42], and Anthony & Kamnitsas's code is MIT [43]; costs a Gaussian per tapped block and extra forward hooks. This is the upgrade to pick if the per-source rejection tables show the single-layer gate missing artefact-type shifts (e.g. burned-in text, collimation).
6. **Do not adopt RMDS as a first move**: it needs class-conditional Gaussians (ill-posed for multi-label) and was worse than MDS on all three OpenMIBOOD medical benchmarks [40].
7. **Do not adopt ODIN**: its gains come from input perturbation that destroys calibration [24], which D-020 makes part of the model version.

---

## 3. Quality gates and view classifiers for CXR

| Asset | Where | Licence (verified) | What it is | Training data | Usable as a gate? |
|---|---|---|---|---|---|
| **torchxrayvision** `xrv.baseline_models.xinario.ViewModel` | github.com/mlmed/torchxrayvision [51][52] | Apache-2.0 (LICENSE) [53] | ResNet-50, 320 px, frontal vs lateral (2 classes) [52][54] | Docstring only points to github.com/xinario/chestViewSplit (Apache-2.0), whose README does **not** disclose the training dataset [54][55] — unverified provenance | Code yes; weights are of unknown provenance, so we could not certify they exclude our external test sources. |
| torchxrayvision `riken.AgeModel` | [52] | Apache-2.0 | Age regressor, MAE 3.67 y, trained on NIH [52] | NIH ChestX-ray14 | Could flag paediatric inputs; NIH is our *training* source, so no contamination issue for a gate — but it is adult-only data, so extrapolation to children is unvalidated. |
| torchxrayvision `emory_hiti.RaceModel`, `chestx_det.PSPNet` | [52] | Apache-2.0 | Race classifier (MIMIC); 14-structure anatomical segmentation (ChestX-Det) [52] | MIMIC; ChestX-Det (NIH-derived) | PSPNet lung/heart masks would let us compute cut-off, rotation and inspiration heuristics; ChestX-Det is NIH-derived, so D-008 already forbids it as external test — fine for a gate. |
| torchxrayvision classifiers (`densenet121-res224-*`, `resnet50-res512-all`) | [51] | Apache-2.0 | 18-finding classifiers trained on NIH/CheXpert/MIMIC/PadChest/RSNA [51] | those datasets | **No** for findings (D-007). |
| **opencxr** (DIAG Nijmegen) | github.com/DIAGNijmegen/opencxr [56] | Apache-2.0 [56] | "Image sorter": frontal PA / frontal AP / lateral / non-chest, plus 90/180/270° rotation, inversion, lateral flip; lung and heart segmentation (PA only); intensity standardisation [56] | not stated in README — unverified | The most complete open gate stack we found; ~22 stars, 122 commits [56]. Provenance of training data must be asked of the authors before use. |
| **DICOM Imaging Router** (VinBigData) | github.com/vinbigdata-medical/DICOM-Imaging-Router [57] | MIT (code) [57] | 5-class body-part router: abdominal, **adult chest, paediatric chest**, spine, other; MobileNet-V1 F1 0.981 on 2,419 test images; 16,093 images [57][58] | VinDr-BodyPartXR (vindr.ai) — dataset licence page did not load; unverified | Only open model we found that separates paediatric from adult chest. Weights promised "on our project website" [57] — availability unverified. |
| UNIFESP body-part dataset (Kaggle) + 7th-place solution | kaggle.com/datasets/felipekitamura/unifesp-xray-bodypart-classification; github.com/skywall/unifesp_x_ray_body_classification [59] | Kaggle page did not render; licence **unverified** | Body-part classification competition dataset | — | Not until the licence is read. |
| **VinDr-PCXR / PediCXR** | physionet.org/content/vindr-pcxr/1.0.0 [60] | PhysioNet Restricted Health Data License 1.5.0, credentialed [60] | 9,125 paediatric CXR studies (< 10 y), 36 findings, boxes [60] | — | Data terms: research-only, credentialed (consistent with D-013). The right *test set* for a paediatric-rejection check. |
| CXR technical-quality models (inspiration/rotation/exposure/cut-off) | BJR 2024 rotation estimator [61]; Eur Radiol 2022 quality cascade [62]; J Am Coll Radiol 2023 AP inspiration/quality CNN [63] | No public code or weights found for any of the three [61][62][63] | Lung/spine segmentation → asymmetry → rotation angle (< 2° error on a phantom) [61] | — | Ideas only; the segmentation-based rotation heuristic is reproducible with opencxr or ChestX-Det masks. |
| **Presidio image redactor** (burned-in text) | github.com/microsoft/presidio [64] | MIT [64] | `DicomImageRedactorEngine`: OCR of the pixel array, metadata-augmented recogniser, box redaction [65] | — | Adoptable as a *detector* (we only need the OCR boxes, not the redaction). Needs Tesseract. |
| pydicom/dicom-cleaner | github.com/pydicom/dicom-cleaner [66] | Not shown on repo page — unverified | OCR-based burned-in pixel detection, "under development" [66] | — | Not mature enough. |

**Comment on D-007 for gate models.** D-007 exists to keep the *evaluated* model's weights free of NIH/CheXpert/MIMIC/PadChest so that external tests are honest. A view / body-part / quality classifier that only decides whether an image is *admitted* never contributes to a finding probability, so contamination of its weights cannot inflate finding AUROC. It *can* still bias the external evaluation in one way: if the gate was trained on the external test source, it may reject that source's hard images at a different rate than a clean gate would, changing the evaluated population. The honest policy is therefore: (i) gate models may use public CXR-pretrained weights, (ii) the gate model's training sources must be recorded in the model card next to the rejection rate per source (D-019 already prints the rate), and (iii) a gate whose training data cannot be established (the xinario weights [55]) should not be used in a locked evaluation. This is a proposed amendment, not a decision.

---

## 4. Explainability

### 4.1 Libraries

| Library | Repo | Licence (verified) | Maturity | Implements | Adopt? |
|---|---|---|---|---|---|
| **pytorch-grad-cam** | github.com/jacobgil/pytorch-grad-cam [67] | MIT [67][68] | PyPI `grad-cam` 1.5.7 (2026-08-21); ~13k stars; no GitHub releases, PyPI only [67][68][69] | GradCAM, HiResCAM, GradCAM++, XGradCAM, AblationCAM, ScoreCAM, EigenCAM, EigenGradCAM, LayerCAM, FullGrad, KPCA-CAM, ShapleyCAM, FinerCAM and others; ROAD (Remove-and-Debias) and confidence-drop metrics [67] | Adoptable. The `BaseCAM` + `HiResCAM` classes are ~100 lines; vendor or depend. |
| **Captum** | github.com/pytorch/captum [70] | BSD-3-Clause [70] | ~5.7k stars [70] | `LayerGradCam`, IntegratedGradients, Occlusion, DeepLift, plus infidelity/sensitivity metrics [70][71] | Adoptable; heavier than needed for CAMs, useful for the metrics. |
| **M3d-CAM** | github.com/MECLabTUDA/M3d-Cam [72] | MIT [72] | ~333 stars; 2D/3D; GaNDLF integration (no MONAI integration found) [72] | Grad-CAM, Guided Grad-CAM, Grad-CAM++, guided backprop, one-line injection [72] | Adoptable but nothing beyond pytorch-grad-cam for 2D. |
| **MONAI** `monai.visualize` | github.com/Project-MONAI/MONAI [73][74] | Apache-2.0 [73] | — | `CAM`, `GradCAM`, `GradCAMpp` (class_activation_maps.py) and `OcclusionSensitivity` [73][74] | Adoptable; a good second opinion on our hand-rolled Grad-CAM, and occlusion sensitivity is a model-agnostic sanity check. |
| **CheXlocalize** (code) | github.com/rajpurkarlab/cheXlocalize [75] | MIT (code) [75] | — | Heatmap → segmentation via Otsu threshold (or per-pathology tuned threshold); mIoU and hit-rate (pointing game) vs radiologist masks; human benchmark [75] | Adoptable code. **Dataset terms**: hosted by Stanford AIMI; 234 val / 668 test CXRs, 10 pathologies; the licence text was not displayed on the AIMI page — Stanford's standard research-use terms are reported by secondary sources as non-commercial — unverified [76]. |

### 4.2 Evidence on faithfulness and localisation

**HiResCAM (Draelos & Carin).** Replaces Grad-CAM's spatial averaging of gradients with an element-wise gradient × activation product; for CNNs that end in a single fully connected layer after the last conv layer (our DenseNet-121 + linear head is exactly this), the explanation is provably the decomposition of the class score over locations, whereas Grad-CAM "sometimes highlights locations the model did not actually use" [77]. The arXiv abstract's experiments are on PASCAL VOC with crowd-sourced evaluation [77]; the medical evidence is in the companion RAD-ChestCT paper (AxialNet), where HiResCAM plus a mask loss improved organ localisation by 33 % on 36,316 CT volumes [78]. **Caution:** the faithfulness guarantee is for the *raw* map; once we ReLU, normalise by max and threshold at 0.5 × max, we are back to a heuristic [77]. Also note that with a single linear head after global average pooling, Grad-CAM and HiResCAM differ only in whether the gradient is averaged before or after multiplying with the activations — the switch is cheap and only matters when activations vary spatially, which is where it should matter for small nodules.

**Grad-CAM++ (Chattopadhyay et al., WACV 2018).** Pixel-wise weighting of positive partial derivatives via higher-order terms; claimed better localisation and handling of multiple instances [79]. No faithfulness guarantee.

**Saporta et al., Nature Machine Intelligence 2022 (CheXlocalize).** Seven methods (Grad-CAM, Grad-CAM++, Integrated Gradients, Eigen-CAM, DeepLIFT, LRP, occlusion) × three CNNs on CheXpert, 10 pathologies, radiologist segmentations, mIoU and hit rate: Grad-CAM generally localised best, but "all seven performed significantly worse compared with the human benchmark", especially for small or complex-shaped pathologies; the paper attributes failures to low model probability, small size and complex shape [80][81]. Exact per-method numbers could not be read from the open abstract (paywalled article; medRxiv blocked) — unverified beyond the abstract.

**Arun et al., Radiology: AI 2021.** Eight saliency methods on SIIM-ACR pneumothorax and RSNA pneumonia, judged on localisation utility (AUPRC vs boxes), sensitivity to weight randomisation, repeatability and reproducibility (SSIM): every method failed at least one criterion and generally underperformed the baselines; the authors recommend detection or segmentation models when localisation is the desired output [82].

**Box-from-heatmap.** CheXlocalize converts heatmaps to masks with Otsu's threshold by default, with an optional per-pathology threshold tuned on validation [75]. Our "largest component above 0.5 × max" is the simpler cousin; the difference is that Otsu adapts to the heatmap's histogram and that per-pathology thresholds fix the known effect that low-probability predictions yield diffuse maps [80]. The NIH ChestX-ray8 release ships `BBox_List_2017.csv` with radiologist boxes for eight findings (atelectasis, cardiomegaly, effusion, infiltration, mass, nodule, pneumonia, pneumothorax) — secondary sources give ~984 boxes over 880 images; the exact count could not be confirmed from the paper abstract — unverified [83][84]. VinDr-CXR (our external test #1 under D-008/D-013) carries lesion-level boxes too, so localisation can be evaluated externally without CheXlocalize's terms.

### 4.3 Verdict on `saliency.py`

Grad-CAM on the last DenseNet block with a max-normalised, 0.5-thresholded largest component is defensible as a *research* overlay: it is the best-localising method in the only CXR-specific multi-method benchmark [80], and it is what every FDA-cleared CXR CADe we surveyed appears to ship in spirit. It is *not* defensible as a localisation claim: both benchmarks show saliency far below radiologists and failing basic sanity checks [80][82]. Cheapest improvements, in order: (1) switch the weighting to HiResCAM (same hooks, no averaging) [77]; (2) evaluate hit-rate and IoU against NIH boxes on the internal test and VinDr boxes on the external test, and report them per finding with CIs in the eval report; (3) tune the threshold per finding on validation and try Otsu as in CheXlocalize [75]; (4) run a weight-randomisation sanity check once per model version [82]. A weakly supervised localisation head is the right long-term answer if the boxes are ever promoted from "coarse region" to a clinical claim, but it changes the regulatory category and should wait.

---

## 5. Implications for Second Reader

Each item names the file, the change, the evidence and the licence status.

1. **`core/eval/metrics.py` — add an equal-mass, debiased ECE next to the current 15-equal-width-bin ECE and report both.** Equal-width bins are biased and unstable at our positive rates [18][19]; a debiased estimator with bootstrap CIs exists under MIT (`uncertainty-calibration`) [12] and equal-mass binning is ~10 lines. Keep the equal-width number for comparability with Guo-style literature [16]. Licence: MIT / own code.

2. **`core/eval/metrics.py` — make Brier the headline calibration number in the eval report; ECE is the diagnostic.** Brier is a proper scoring rule that does not depend on binning; Carse et al. found no benefit from tuning to ECE instead of NLL [17]; Berger et al.'s ECE swings (0.04 → 0.35) show how fragile ECE is [24]. Licence: n/a.

3. **`modalities/cxr/train.py` — keep per-output scalar temperature on NLL (D-020); add optional per-output Platt (a, b) or beta calibration as a *reported sensitivity analysis*, not a default.** Guo et al. show more parameters overfit [16]; beta calibration's identity-containing family is the safe extra degree of freedom [10][11]; sklearn `sigmoid` is BSD-3 [7], betacal is MIT [10].

4. **`core/eval/evaluate.py` + `core/monitoring/drift.py` — treat per-source ECE/Brier as the primary shift indicator, because temperature fitted on internal validation is not expected to hold on a new site.** Ovadia et al.: "calibration on the i.i.d. validation dataset does not guarantee calibration under distributional shift" [21]. The PSI output monitor is the runtime proxy; consider adding a per-source re-calibration path as a PCCP item (Qure precedent, `evidence_bar.md`). Licence: n/a.

5. **`modalities/cxr/gate.py` `OodModel.fit` — L2-normalise penultimate features before Ledoit-Wolf, and record both raw and normalised rejection curves for one model version.** Mahalanobis++ shows feature-norm variation breaks the Gaussian assumption [32]; Berger et al. show CXR features are not automatically Gaussian-friendly [24]. One line; no dependency. Unverified on CXR — hence the A/B.

6. **`modalities/cxr/gate.py` — log MaxLogit and a KNN distance as secondary scores, gate on Mahalanobis only (D-019 unchanged).** Feature-space scores beat logit scores on medical data [40][45]; MaxLogit is nevertheless the only score with multi-label evidence [26] and is free. KNN via pytorch-ood (Apache-2.0) [36][37] or ~20 lines of sklearn; bundle size grows by the feature bank. Do not adopt RMDS first (worse than MDS on all OpenMIBOOD benchmarks [40]); do not adopt ODIN (breaks calibration [24]).

7. **`modalities/cxr/gate.py` / `model.py` — if per-source rejection tables show artefact-type misses, move to multi-depth Mahalanobis (a Gaussian per DenseNet block, summed).** Strongest medical evidence for near-OOD [40][42]; Anthony & Kamnitsas's code is MIT [43]; FRODO used intermediate activations for exactly our lateral/non-chest use case [44]. Cost: forward hooks and k extra covariance fits.

8. **`DECISIONS.md` — propose D-0xx amending D-007 for gate models:** public CXR-pretrained weights are permitted for view / body-part / quality classifiers that never contribute to finding probabilities, provided training sources are recorded in the model card and the rejection rate per external source is reported (D-019 already prints it); gates of unknown provenance (torchxrayvision's xinario view weights [54][55]) are excluded from locked evaluations. Candidates with Apache-2.0/MIT code: opencxr image sorter (PA/AP/lateral/non-chest + rotation/flip) [56], DICOM Imaging Router (adult vs paediatric chest) [57]. Both need a training-data provenance answer from the authors before adoption — unverified.

9. **`modalities/cxr/gate.py` — add a paediatric check and a burned-in-text check as named reasons.** VinDr-PCXR is the test set (PhysioNet restricted licence, credentialed; research-only, consistent with D-013) [60]; the torchxrayvision `riken.AgeModel` (Apache-2.0, NIH-trained) is a zero-cost first age signal [52][53]; Presidio's `DicomImageRedactorEngine` OCR (MIT) can replace our heuristic text detector [64][65].

10. **`modalities/cxr/saliency.py` — switch `grad_cam` weighting to HiResCAM (element-wise gradient × activation, no spatial mean) and keep the same box logic.** Provably faithful for our conv → GAP → linear architecture [77]; identical hooks; pytorch-grad-cam's implementation is MIT if we prefer to depend on it [67][68]. Our post-processing (ReLU, max-normalise, threshold) still voids the guarantee, so present it as an improvement in the *raw* map, not as a faithfulness claim.

11. **`modalities/cxr/evaluate.py` — add localisation metrics (hit rate and IoU vs boxes) per finding with CIs, using NIH `BBox_List_2017.csv` internally and VinDr-CXR boxes externally; try Otsu and per-finding thresholds as in CheXlocalize.** Both saliency benchmarks show maps fail basic checks unless measured [80][82]; CheXlocalize code is MIT [75]; the CheXlocalize *dataset* terms are unverified and likely research-only [76], so prefer NIH/VinDr boxes, which we already hold under D-008.

12. **Documentation / model card — state explicitly that heatmaps and boxes are "coarse region of model attention", not localisation, until item 11 shows otherwise.** Arun et al. recommend detection or segmentation models when localisation is the output [82]; Saporta et al. show all saliency methods below the human benchmark [80]. Licence: n/a.

### What we could not verify (open items)

- Training data behind the xinario / torchxrayvision frontal-lateral weights [54][55] and behind opencxr's image sorter [56].
- Licence of VinDr-BodyPartXR [57][58], of the UNIFESP Kaggle dataset [59], of OpenMIBOOD's code [39], of pydicom/dicom-cleaner [66], and the CheXlocalize dataset's exact terms [76].
- Metrics Reloaded's specific calibration-metric recommendations [20]; the MICCAI 2022 "weight scaling" paper's content [23].
- Per-method numbers in Saporta et al. beyond "Grad-CAM generally best, all below human" [80][81].
- The exact NIH bounding-box count (984 boxes vs 880 images) [83][84].
- Whether torch-uncertainty has released after v0.13.0 (2024-07-22) [4].
- Transfer of Mahalanobis++ [32], ViM's virtual-logit matching [30] and energy [27] to a sigmoid multi-label head; none of the three papers evaluates that setting.

---

## References (all accessed 2026-09-21)

1. netcal README and licence badge — https://github.com/EFS-OpenSource/calibration-framework
2. netcal releases — https://github.com/EFS-OpenSource/calibration-framework/releases
3. torch-uncertainty README — https://github.com/ENSTA-U2IS-AI/torch-uncertainty
4. torch-uncertainty releases — https://github.com/ENSTA-U2IS-AI/torch-uncertainty/releases
5. torch-uncertainty API docs — https://torch-uncertainty.github.io/api.html
6. gpleiss/temperature_scaling — https://github.com/gpleiss/temperature_scaling
7. scikit-learn probability calibration user guide — https://scikit-learn.org/stable/modules/calibration.html
8. `CalibratedClassifierCV` API (1.9.1; `method="temperature"` added 1.8) — https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html
9. MAPIE — https://github.com/scikit-learn-contrib/MAPIE
10. betacal python package — https://github.com/betacal/python
11. betacal project page (Kull, Silva Filho, Flach, AISTATS 2017) — https://github.com/betacal/betacal.github.io
12. verified_calibration / `uncertainty-calibration` (Kumar, Liang, Ma, NeurIPS 2019) — https://github.com/p-lambda/verified_calibration
13. relplot (apple/ml-calibration) — https://github.com/apple/ml-calibration
14. relplot LICENSE — https://raw.githubusercontent.com/apple/ml-calibration/main/LICENSE
15. Błasiok & Nakkiran, "Smooth ECE", ICLR 2024 — https://arxiv.org/abs/2309.12236
16. Guo, Pleiss, Sun, Weinberger, "On calibration of modern neural networks", ICML 2017 — https://arxiv.org/abs/1706.04599
17. Carse, Alvarez Olmo, McKenna, "Calibration of deep medical image classifiers…", UNSURE 2022 — https://link.springer.com/chapter/10.1007/978-3-031-16749-2_9 (abstract via https://discovery.dundee.ac.uk/en/publications/calibration-of-deep-medical-image-classifiers-an-empirical-compar/)
18. Nixon et al., "Measuring calibration in deep learning", 2019 — https://arxiv.org/abs/1904.01685
19. Roelofs et al., "Mitigating bias in calibration error estimation", AISTATS 2022 — https://arxiv.org/abs/2012.08668
20. Maier-Hein et al., "Metrics reloaded", Nature Methods 2024 — https://arxiv.org/abs/2206.01653 ; https://www.nature.com/articles/s41592-023-02151-z
21. Ovadia et al., "Can you trust your model's uncertainty?", NeurIPS 2019 — https://arxiv.org/abs/1906.02530 (quotes from the PDF, §4)
22. Samarla et al., CALM-CXR, MethodsX 2026 — https://pmc.ncbi.nlm.nih.gov/articles/PMC13463977/
23. "Calibration of medical imaging classification systems with weight scaling", MICCAI 2022 — https://link.springer.com/chapter/10.1007/978-3-031-16452-1_61 (paywalled; unverified)
24. Berger, Paschali, Glocker, Kamnitsas, "Confidence-based OOD detection: a comparative study and analysis", UNSURE 2021 — https://arxiv.org/abs/2107.02568 (Tables 2–3 from the PDF)
25. Hendrycks & Gimpel, MSP baseline, ICLR 2017 — https://arxiv.org/abs/1610.02136
26. Hendrycks et al., "Scaling OOD detection to real-world settings" (MaxLogit), ICML 2022 — https://arxiv.org/abs/1911.11132
27. Liu et al., "Energy-based OOD detection", NeurIPS 2020 — https://arxiv.org/abs/2010.03759
28. Lee et al., "A simple unified framework for detecting OOD samples and adversarial attacks" (Mahalanobis), NeurIPS 2018 — https://arxiv.org/abs/1807.03888
29. Ren et al., "A simple fix to Mahalanobis distance for improving near-OOD detection", 2021 — https://arxiv.org/abs/2106.09022
30. Wang et al., "ViM: Out-of-distribution with virtual-logit matching", CVPR 2022 — https://arxiv.org/abs/2203.10807
31. Sun et al., "Out-of-distribution detection with deep nearest neighbors", ICML 2022 — https://arxiv.org/abs/2204.06507
32. Mueller & Hein, "Mahalanobis++", 2025 — https://arxiv.org/abs/2505.18032
33. OpenOOD README — https://github.com/Jingkang50/OpenOOD ; https://raw.githubusercontent.com/Jingkang50/OpenOOD/main/README.md
34. OpenOOD LICENSE (MIT) — https://raw.githubusercontent.com/Jingkang50/OpenOOD/main/LICENSE
35. Zhang et al., "OpenOOD v1.5", DMLR 2024 — https://arxiv.org/abs/2306.09301 (tables from https://arxiv.org/html/2306.09301v4); leaderboard https://zjysteven.github.io/OpenOOD/
36. pytorch-ood README — https://github.com/kkirchheim/pytorch-ood
37. pytorch-ood LICENSE (Apache-2.0) — https://raw.githubusercontent.com/kkirchheim/pytorch-ood/dev/LICENSE
38. pytorch-ood releases — https://github.com/kkirchheim/pytorch-ood/releases
39. OpenMIBOOD code — https://github.com/remic-othr/OpenMIBOOD
40. Gutbrod, Rauber, Nunes, Palm, "OpenMIBOOD", CVPR 2025 — https://arxiv.org/abs/2503.16247 (tables from https://arxiv.org/html/2503.16247)
41. Cao, Huang, Hui, Cohen, "A benchmark of medical out of distribution detection", 2020 — https://arxiv.org/abs/2007.04250
42. Anthony & Kamnitsas, "On the use of Mahalanobis distance for OOD detection with neural networks for medical imaging", UNSURE 2023 — https://arxiv.org/abs/2309.01488
43. Anthony & Kamnitsas code (MIT) — https://github.com/HarryAnthony/Mahalanobis-OOD-detection
44. Çallı, Murphy, Sogancioglu, van Ginneken, "FRODO", MIDL 2019 — https://arxiv.org/abs/1907.01253
45. Wollek et al., "A knee cannot have lung disease" (in-distribution voting), 2022 — https://arxiv.org/abs/2208.01077
46. Karimi, Farnia, Bae, "Robust detection of OOD shifts in chest X-ray imaging", J Imaging Inform Med 2025 — https://pmc.ncbi.nlm.nih.gov/articles/PMC13103168/
47. CheXseen — https://arxiv.org/abs/2103.04590
48. Zimmerer et al., "MOOD 2020", IEEE TMI 2022 — https://openaccess.city.ac.uk/id/eprint/28169/ ; https://link.springer.com/chapter/10.1007/978-3-658-41657-7_67
49. Di Salvo, Doerrich, Ledig, "MedMNIST-C", 2024 — https://arxiv.org/abs/2406.17536 ; https://github.com/francescodisalvo05/medmnistc-api
50. Guo et al., "MedShift", IEEE JBHI 2023 — https://pmc.ncbi.nlm.nih.gov/articles/PMC10513895/ ; https://arxiv.org/abs/2112.13885
51. torchxrayvision README — https://raw.githubusercontent.com/mlmed/torchxrayvision/master/README.md ; https://github.com/mlmed/torchxrayvision
52. torchxrayvision models docs (baseline_models) — https://mlmed.org/torchxrayvision/models.html
53. torchxrayvision LICENSE (Apache-2.0) — https://raw.githubusercontent.com/mlmed/torchxrayvision/master/LICENSE
54. `xrv.baseline_models.xinario.ViewModel` source — https://raw.githubusercontent.com/mlmed/torchxrayvision/master/torchxrayvision/baseline_models/xinario/__init__.py
55. xinario/chestViewSplit (Apache-2.0; training data undisclosed) — https://github.com/xinario/chestViewSplit
56. opencxr (Apache-2.0) — https://github.com/DIAGNijmegen/opencxr
57. DICOM Imaging Router code (MIT) — https://github.com/vinbigdata-medical/DICOM-Imaging-Router
58. Pham, Do, Nguyen, "DICOM Imaging Router", ICCV-W CVAMD 2021 — https://arxiv.org/abs/2108.06490 ; dataset page https://vindr.ai/datasets/bodypartxr (did not load)
59. UNIFESP body-part dataset — https://www.kaggle.com/datasets/felipekitamura/unifesp-xray-bodypart-classification (did not render); solution https://github.com/skywall/unifesp_x_ray_body_classification
60. VinDr-PCXR on PhysioNet — https://physionet.org/content/vindr-pcxr/1.0.0/
61. Sun et al., thoracic rotation estimation, BJR 2024 — https://pmc.ncbi.nlm.nih.gov/articles/PMC11417390/
62. "Automated quality assessment of chest radiographs based on deep learning and linear regression cascade", Eur Radiol 2022 — https://link.springer.com/article/10.1007/s00330-022-08771-x
63. "Assessment of inspiration and technical quality in AP thoracic radiographs using machine learning", 2023 — https://pubmed.ncbi.nlm.nih.gov/37918335/
64. Presidio (MIT) — https://github.com/microsoft/presidio
65. `DicomImageRedactorEngine` source — https://raw.githubusercontent.com/microsoft/presidio/main/presidio-image-redactor/presidio_image_redactor/dicom_image_redactor_engine.py
66. pydicom/dicom-cleaner — https://github.com/pydicom/dicom-cleaner
67. pytorch-grad-cam (MIT) — https://github.com/jacobgil/pytorch-grad-cam
68. `grad-cam` on PyPI (1.5.7, 2026-08-21, MIT) — https://pypi.org/project/grad-cam/
69. pytorch-grad-cam releases page (none) — https://github.com/jacobgil/pytorch-grad-cam/releases
70. Captum (BSD-3) — https://github.com/pytorch/captum
71. Captum layer attribution API (`LayerGradCam`) — https://captum.ai/api/layer.html
72. M3d-CAM (MIT) — https://github.com/MECLabTUDA/M3d-Cam
73. MONAI `class_activation_maps.py` (Apache-2.0 header) — https://raw.githubusercontent.com/Project-MONAI/MONAI/dev/monai/visualize/class_activation_maps.py
74. MONAI `occlusion_sensitivity.py` — https://raw.githubusercontent.com/Project-MONAI/MONAI/dev/monai/visualize/occlusion_sensitivity.py
75. CheXlocalize code (MIT) — https://github.com/rajpurkarlab/cheXlocalize
76. CheXlocalize dataset page, Stanford AIMI — https://aimi.stanford.edu/datasets/chexlocalize ; https://stanfordaimi.azurewebsites.net/datasets/23c56a0d-15de-405b-87c8-99c30138950c
77. Draelos & Carin, "Use HiResCAM instead of Grad-CAM for faithful explanations of CNNs" — https://arxiv.org/abs/2011.08891
78. Draelos & Carin, "Explainable multiple abnormality classification of chest CT volumes" (AxialNet + HiResCAM, RAD-ChestCT) — https://arxiv.org/abs/2111.12215
79. Chattopadhyay et al., "Grad-CAM++", WACV 2018 — https://arxiv.org/abs/1710.11063
80. Saporta et al., "Benchmarking saliency methods for chest X-ray interpretation", Nature Machine Intelligence 4:867–878 (2022) — https://www.nature.com/articles/s42256-022-00536-x (paywalled; abstract via https://api.semanticscholar.org/graph/v1/paper/DOI:10.1038/s42256-022-00536-x)
81. Saporta et al., medRxiv preprint — https://www.medrxiv.org/content/10.1101/2021.02.28.21252634v4.full (blocked at access time)
82. Arun et al., "Assessing the (un)trustworthiness of saliency maps for localizing abnormalities in medical imaging", Radiology: AI 2021 — https://arxiv.org/abs/2008.02766 ; https://pubs.rsna.org/doi/10.1148/ryai.2021200267 (blocked at access time)
83. Wang et al., "ChestX-ray8", CVPR 2017 — https://arxiv.org/abs/1705.02315
84. NIH `BBox_List_2017.csv` description (Kaggle mirror) — https://www.kaggle.com/datasets/nih-chest-xrays/data/home
