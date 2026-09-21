# Open-i — Indiana University chest X-ray collection (IU CXR) — licence

| | |
|---|---|
| Registry id | `openi_iu_cxr` |
| Official source | NLM Open-i: <https://openi.nlm.nih.gov/> (collection filter "Indiana U. Chest X-rays"; bulk archives historically at `https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz`, `NLMCXR_dcm.tgz`, `NLMCXR_reports.tgz`) |
| Verified on | **2026-09-21 — NOT verifiable at source today.** `openi.nlm.nih.gov` (including `/faq`) is serving only a US federal funding-lapse notice ("Because of a lapse in government funding, the information on this website may not be up to date …"), and the three `.tgz` URLs return a 13.8 kB HTML error page instead of the archives. The licence below is what the Open-i FAQ states as last published (confirmed via multiple secondary sources on 2026-09-21); **re-verify the FAQ text before any download**. |
| Licence | CC BY-NC-ND 4.0 for the Indiana University images (Open-i FAQ: images from the Indiana University hospital network are distributed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License) |
| `commercial_ok` | **false** (NonCommercial) — and NoDerivatives means we may not redistribute derived images; model training on them is a grey area, so role is limited to OOD probing / report-text experiments, never finding-head training |
| Credentialing | none — anonymous download once the site is up; no account, no click-through |
| Size | 7,470 frontal + lateral PNGs (~1.1 GB tgz) / DICOM (~ 3 GB) and 3,955 XML reports (~ 5 MB); two Indiana hospital systems, Indiana Network for Patient Care |

## Obligations

* Attribute Open-i / NLM and Indiana University; cite Demner-Fushman D. et al., "Preparing a collection of radiology examinations for distribution and retrieval", JAMIA 23(2):304–310 (2016).
* No commercial use, no redistribution of derivatives.

## Download (when the site is back)

```bash
mkdir -p data/raw/openi_iu_cxr && cd data/raw/openi_iu_cxr
for f in NLMCXR_png.tgz NLMCXR_reports.tgz; do
  curl -L -C - -o "$f" "https://openi.nlm.nih.gov/imgs/collections/$f"
  file "$f" | grep -q gzip || { echo "$f is not an archive (site still down?)"; exit 1; }
done
```
