# PadChest — licence and access

| | |
|---|---|
| Registry id | `padchest` |
| Official source | BIMCV (Medical Imaging Databank of the Valencia Region): <https://bimcv.cipf.es/bimcv-projects/padchest/> |
| Verified on | **2026-09-21** (project page and Research Use Agreement page read) |
| Licence | "PADCHEST Dataset Research Use Agreement": <https://bimcv.cipf.es/bimcv-projects/padchest/padchest-dataset-research-use-agreement/> |
| `commercial_ok` | **false** — research only; non-academic research needs case-by-case BIMCV approval |
| Credentialing | **Human action required.** Access is requested through a Google Form linked from the project page (<https://docs.google.com/forms/d/e/1FAIpQLSe3GiDMkTt9maCNpqmvyaowc0RK6pCW42-7OzbGQkla1mY4bQ/viewform>): name, institution, email, intended use, and acceptance of the Research Use Agreement. BIMCV e-mails a personal download link (the agreement forbids sharing that link outside your organisation). No CITI course, no wet-ink DUA. |
| Size | ~1.02 TB full release (PNG 16-bit, original resolution, 160,868 images / 67,000 patients, 2009–2017, Hospital San Juan de Alicante); a 167 MB sample is on Kaggle |
| Format | PNG16 + one CSV of labels (`PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv`), 174 findings, 19 diagnoses, 104 anatomical locations; ~27 % physician-labelled, rest RNN-labelled (`MethodLabel`) |

## Licence text (verbatim excerpts from the Research Use Agreement page, 2026-09-21)

> 1. Permission is granted to view and use the PadChest Dataset without charge for research purposes only.

> 3. […] the BIMCV retains all rights, title, and interest in the PadChest Dataset.

> 4. You may make a verbatim copy of the PadChest Dataset for personal, research use as permitted in this Research Use Agreement.

> 5. YOU MAY NOT DISTRIBUTE, PUBLISH, OR REPRODUCE A COPY of any portion or all of the PadChest Dataset to others without specific prior written permission.

> 6. YOU MAY NOT SHARE THE DOWNLOAD LINK to the PadChest dataset to others outside of your organization.

> 8. The PadChest Dataset has not been reviewed or approved by the Food and Drug Administration or the European Medicines Agency, and is for Research Use Only.

Other clauses (paraphrased): any non-academic research use must be evaluated case by case and approved by BIMCV-PadChest (cl. 1); computational models that used PadChest and are publicly referenced are asked to be shared with code and weights, citing the PadChest article (cl. 2); no reverse engineering / derivative works except models per cl. 2 (cl. 7); provided "AS IS" (cl. 9); re-identification prohibited (cl. 10); breach → termination and return of copies (cl. 11); indemnification (cl. 12).

## Obligations

* Research-only, non-commercial → `commercial_ok: false`. PadChest is an **external test set** here (role `external_test`, manual-label subset only, D-0xx), so it never touches training weights.
* Never redistribute the images or the link; store under `data/raw/padchest/` (git-ignored).
* Cite: Bustos A., Pertusa A., Salinas J.-M., de la Iglesia-Vayá M., "PadChest: A large chest x-ray image dataset with multi-label annotated reports", Medical Image Analysis 66 (2020) 101797 (arXiv:1901.07441).
