# Kermany paediatric chest X-ray (Guangzhou Women and Children's Medical Center) — licence

| | |
|---|---|
| Registry id | `kermany_pediatric` |
| Official source | Mendeley Data, "Large Dataset of Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images", v3 (2018-06-01), DOI 10.17632/rscbjbr9sj.3: <https://data.mendeley.com/datasets/rscbjbr9sj/3> |
| Verified on | **2026-09-21** (Mendeley public API `data_licence` field + dataset page) |
| Licence | **CC BY 4.0** — Creative Commons Attribution 4.0 International, <http://creativecommons.org/licenses/by/4.0> |
| `commercial_ok` | true (attribution required) |
| Credentialing | **none** — anonymous direct download, no account, no click-through |
| Size | v3 single archive `ZhangLabData.zip`, **8,441,154,111 bytes (~8.4 GB)**, containing both the OCT set and the chest X-ray set (`chest_xray/{train,test}/{NORMAL,PNEUMONIA}`, 5,856 JPEGs, 1–5-year-olds, AP views). Direct URL (from the Mendeley files API): <https://data.mendeley.com/public-files/datasets/rscbjbr9sj/files/810b2ce2-11c3-4424-996e-3bef36600907/file_downloaded> |

## Licence text (verbatim, Mendeley `data_licence.description`)

> You can share, copy and modify this dataset so long as you give appropriate credit, provide a link to the CC BY license, and indicate if changes were made, but you may not do so in a way that suggests the rights holder has endorsed you or your use of the dataset. Note that further permission may be required for any content within the dataset that is identified as belonging to a third party.

The dataset page also states the images are "made available for use in research only" in its description; the binding licence attached by the depositors is CC BY 4.0.

## Obligations

* Attribute: Kermany D.S., Goldbaum M., Cai W. et al., "Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning", Cell 172(5):1122–1131 (2018), and cite the Mendeley DOI.
* Role here is `ood_negative` only (paediatric population is out of the intended-use population; used to test that the gate rejects it). Never used for training the finding heads.

## Download (unattended)

```bash
curl -L -C - -o data/raw/kermany_pediatric/ZhangLabData.zip \
  "https://data.mendeley.com/public-files/datasets/rscbjbr9sj/files/810b2ce2-11c3-4424-996e-3bef36600907/file_downloaded"
# expected size 8441154111 bytes; unzip only chest_xray/
```
