# NIH ChestX-ray14 (ChestX-ray8 / "CXR8") — licence and attribution

| | |
|---|---|
| Registry id | `nih_cxr14` |
| Official source | NIH Clinical Center Box folder "ChestXray-NIHCC": <https://nihcc.app.box.com/v/ChestXray-NIHCC> (owner: Summers, Ronald (NIH/CC/DRD); enterprise owner: National Institutes of Health - Clinical Center; folder created 2017-09-01, last modified 2025-12-16) |
| Verified on | **2026-09-21** (Box folder browsed; `README_CHESTXRAY.pdf` and `FAQ_CHESTXRAY.pdf` downloaded and read; copies kept at `data/raw/nih_cxr14/`, git-ignored) |
| Licence class | No use restrictions; attribution requested |
| `commercial_ok` | true (no restriction stated; keep the three attribution obligations below) |
| Credentialing | none — no account, no click-through, no DUA |
| Size | 12 archives, 45,080,462,784 bytes (~45.1 GB) compressed; 112,120 PNG 1024×1024 |

## Licence text (verbatim)

From `FAQ_CHESTXRAY.pdf` (NIH Clinical Center, in the Box folder, updated 2017-12-14), Q04:

> **Q04: Are there any restrictions in using this dataset?**
> **A:** The usage of the data set is unrestricted. But you should provide the link to our original download site, acknowledge the NIH Clinical Center and provide a citation to our CVPR 2017 paper.

From `README_CHESTXRAY.pdf` (NIH Clinical Center, updated 2018-11-05):

> If you find the dataset useful for your research projects, please cite our CVPR 2017 paper:
>
> Xiaosong Wang, Yifan Peng, Le Lu, Zhiyong Lu, Mohammadhadi Bagheri, Ronald M. Summers. ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases, IEEE CVPR, pp. 3462-3471, 2017

> Acknowledgement: This work was supported by the Intramural Research Program of the NIH Clinical Center (clinicalcenter.nih.gov) and National Library of Medicine (www.nlm.nih.gov).

> Note that original radiology reports (associated with these chest x-ray studies) are not meant to be publicly shared for many reasons.

No separate licence file (e.g. CC-BY) is published; the FAQ statement above is the only terms text NIH provides.

## Our obligations (carried into every report / model card that uses this data)

1. Link to the original download site: <https://nihcc.app.box.com/v/ChestXray-NIHCC>
2. Acknowledge the **NIH Clinical Center**.
3. Cite Wang et al., CVPR 2017 (BibTeX below).

```bibtex
@InProceedings{wang2017chestxray,
  author    = {Wang, Xiaosong and Peng, Yifan and Lu, Le and Lu, Zhiyong and Bagheri, Mohammadhadi and Summers, Ronald},
  title     = {ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases},
  booktitle = {2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {3462--3471},
  year      = {2017}
}
```

## Files obtained (2026-09-21)

Image archives — URLs are verbatim from NIH's `images/batch_download_zips.py` (Box file id 371647823217, updated 2020-07-20); sizes are the server's `Content-Range` totals; MD5s are those NIH publishes in FAQ Q07 ("We thank Utku Ozbulak for providing those MD5s"):

| file | bytes | md5 |
|---|---|---|
| images_001.tar.gz | 2,008,470,987 | fe8ed0a6961412fddcbb3603c11b3698 |
| images_002.tar.gz | 3,952,623,504 | ab07a2d7cbe6f65ddd97b4ed7bde10bf |
| images_003.tar.gz | 3,929,234,850 | 2301d03bde4c246388bad3876965d574 |
| images_004.tar.gz | 3,838,903,983 | 9f1b7f5aae01b13f4bc8e2c44a4b8ef6 |
| images_005.tar.gz | 3,935,496,531 | 1861f3cd0ef7734df8104f2b0309023b |
| images_006.tar.gz | 3,986,301,172 | 456b53a8b351afd92a35bc41444c58c8 |
| images_007.tar.gz | 4,016,328,426 | 1075121ea20a137b87f290d6a4a5965e |
| images_008.tar.gz | 4,018,347,353 | b61f34cec3aa69f295fbb593cbd9d443 |
| images_009.tar.gz | 4,111,327,929 | 442a3caa61ae9b64e61c561294d1e183 |
| images_010.tar.gz | 4,181,556,296 | 09ec81c4c31e32858ad8cf965c494b74 |
| images_011.tar.gz | 4,187,084,020 | 499aefc67207a5a97692424cf5dbeed5 |
| images_012.tar.gz | 2,914,187,733 | dc9fda1757c2de0032b63347a7d2895c |

Metadata (Box refuses direct links for these — "This user is not allowed to use direct links" — so they were saved through the Box web UI; sha256 of our copies):

| file | Box file id | updated | sha256 |
|---|---|---|---|
| Data_Entry_2017_v2020.csv (8.6 MB, 112,120 rows; column `Patient Sex`, ages already cleaned) | 219760887468 | 2025-04-02 | c69a6daca3549af707ca9cacbdf9f9a7b6a9188e8c61157df653520bb72d8eb1 |
| BBox_List_2017.csv (984 boxes) | 219760940956 | 2017-09-04 | 0bbfea9d4c4e9771481b3023b1bc9f0df9dea924453b12986beb29b0c4d0c95b |
| train_val_list.txt (86,524 images) | 256056636701 | 2017-12-14 | 61fbe896321c1c1c8b75f3e4f3a08e4fef6486d95ef8a667c31d4d60dca6cb81 |
| test_list.txt (25,596 images) | 256055473534 | 2017-12-14 | 38ca5ef7f756092946f57c1a59faca882ed589a1ab1f72590b45dc06c6d5e1cc |
| README_CHESTXRAY.pdf | 220660789610 | 2018-11-05 | bcc8a007593e7d6a6bccbab98748427f554f0593a4901eb39e2018d5eeaa32f1 |
| FAQ_CHESTXRAY.pdf | 249502714403 | 2017-12-14 | 674665256e6a14c8ebaa93648f438c2b4ca21205167839559bab3ecc89b6b83a |

Note: the folder no longer carries `Data_Entry_2017.csv`; NIH replaced it with `Data_Entry_2017_v2020.csv` (same 112,120 rows; `Patient Gender` renamed `Patient Sex`; the impossible ages, e.g. 412, were corrected). The adapter accepts both.

Other folders in the same Box share (not downloaded, separately attributed if ever used): `MAPLEZ` (LLM-extracted labels, 2025), `PruneCXR` (2023), `LongTailCXR` (2022).

## Caveats recorded for the eval protocol

* Labels are NLP-mined from reports (NIH: ">90%" accuracy); "No Finding" ≠ normal (FAQ Q08).
* Reports are not released.
* NIH images are re-released inside SIIM-ACR, RSNA Pneumonia and ChestX-Det — those can never serve as external test sets (see `docs/PLAN.md`).
