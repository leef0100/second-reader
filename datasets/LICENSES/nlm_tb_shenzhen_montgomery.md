# NLM tuberculosis chest X-ray sets (Shenzhen Hospital + Montgomery County) — licence

| | |
|---|---|
| Registry ids | `nlm_tb_shenzhen`, `nlm_tb_montgomery` |
| Official source | NLM / Lister Hill National Center for Biomedical Communications, public data server: <https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/> (moved here 2024-08-28; the old `ChinaSet_AllFiles.zip` / `MontgomerySet.zip` archives and the `lhncbc.nlm.nih.gov/LHC-downloads` links now return 403/maintenance) |
| Verified on | **2026-09-21** (directory listings + `NLM-ChinaCXRSet-ReadMe.docx` and `NLM-MontgomeryCXRSet-ReadMe.pdf` downloaded and read) |
| Licence | No formal licence; README "Citation and Use" terms (below). Public release exempted from IRB review (NIH OHSRP No. 5357). |
| `commercial_ok` | **false (treat as research-only)** — the README asks that the set not be shared outside the research group and that new requests be forwarded to NLM; no commercial grant is stated |
| Credentialing | **none** — anonymous HTTPS, no account, no click-through; files are served individually (no zip), so download = recursive fetch of the directory listing (`index.json` per folder) |
| Size | Shenzhen: 662 PNG (336 TB / 326 normal), ~3.77 GB, ~3000×3000 px, `CXR_png/CHNCXR_####_{0,1}.png` + `ClinicalReadings/*.txt` (age, sex, finding) + lung masks in `Annotations/`; Montgomery: 138 PNG (58 TB / 80 normal), ~0.61 GB, 4020×4892 or 4892×4020, `MontgomerySet/CXR_png/MCUCXR_####_{0,1}.png` + `ClinicalReadings/` + `ManualMask/` (left/right lung) + `montgomery_consensus_roi.csv` |

## Licence text (verbatim)

`NLM-ChinaCXRSet-ReadMe.docx`, "Citation and Use":

> Please do not share the dataset outside of your research group/ organization, but forward new dataset requests to us.
> Please inform us if you find errors or inconsistencies in the data.
> It is requested that publications resulting from the use of this data attribute the source (National Library of Medicine, National Institutes of Health, Bethesda, MD, USA and Shenzhen No.3 People's Hospital, Guangdong Medical College, Shenzhen, China) and cite the following publications:

`NLM-MontgomeryCXRSet-ReadMe.pdf`, "Citation and Use":

> (a) Please do not share the data set outside of your research group/organization, but forward new data set requests to us.
> (b) Please inform us if you find errors or inconsistencies in the data.
> (c) It is requested that publications resulting from the use of this data attribute the source (National Library of Medicine, National Institutes of Health, Bethesda, MD, USA) and cite the following publications:

Required citations (both sets):

* Jaeger S., Karargyris A., Candemir S. et al. "Automatic tuberculosis screening using chest radiographs." IEEE Trans Med Imaging 33(2):233–45 (2014). doi:10.1109/TMI.2013.2284099
* Candemir S., Jaeger S., Palaniappan K. et al. "Lung segmentation in chest radiographs using anatomical atlases with nonrigid registration." IEEE Trans Med Imaging 33(2):577–90 (2014). doi:10.1109/TMI.2013.2290491

## Role here

Vendor/site-shift probes for the OOD gate and lung-mask sanity checks (`ood_negative` / `gate_training`). Labels are TB vs normal only — no pneumothorax / effusion / nodule labels, so never a finding-head training or test source.

## Download (unattended)

```bash
BASE=https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets
for d in Shenzhen-Hospital-CXR-Set/CXR_png Shenzhen-Hospital-CXR-Set/ClinicalReadings \
         Montgomery-County-CXR-Set/MontgomerySet/CXR_png Montgomery-County-CXR-Set/MontgomerySet/ClinicalReadings; do
  mkdir -p data/raw/nlm_tb/$d
  curl -sS "$BASE/$d/index.json" | python3 -c 'import sys,json;[print(c["Key"]) for c in json.load(sys.stdin)["Contents"] if not c["Key"].startswith("index.")]' \
    | while read -r k; do curl -sS -C - -o "data/raw/nlm_tb/$d/$k" "$BASE/$d/$k"; done
done
```
