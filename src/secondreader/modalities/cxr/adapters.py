"""Native dataset CSVs → unified manifest rows. Column names follow each dataset's
public release; every adapter is pure (no downloads) and unit-tested on tiny
fixture CSVs. Verify against the current release when data arrives.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd

from secondreader.modalities.cxr.data import MANIFEST_COLUMNS

_VIEW_LABEL = {"PA": 0, "AP": 0, "LL": 1, "L": 1, "RL": 1, "LATERAL": 1}


def _finish(rows: pd.DataFrame) -> pd.DataFrame:
    for c in MANIFEST_COLUMNS:
        if c not in rows.columns:
            rows[c] = np.nan
    rows["tier"] = ""
    return rows[list(MANIFEST_COLUMNS)].reset_index(drop=True)


NIH_ENTRY_CSVS: tuple[str, ...] = ("Data_Entry_2017_v2020.csv", "Data_Entry_2017.csv")


def nih_manifest(
    root: Path, entry_csv: str | None = None, image_dir: str = "images"
) -> pd.DataFrame:
    """NIH ChestX-ray14: `Finding Labels` pipe-separated; Nodule|Mass → nodule_mass.

    Accepts both NIH releases of the metadata: the current `Data_Entry_2017_v2020.csv`
    (column `Patient Sex`, ages cleaned) and the original `Data_Entry_2017.csv`
    (`Patient Gender`, a few impossible ages such as 412 → unknown). Ages may carry a
    unit suffix in some mirrors ("058Y"); only the digits are used.
    """
    root = Path(root)
    if entry_csv is None:
        found = [c for c in NIH_ENTRY_CSVS if (root / c).exists()]
        if not found:
            raise FileNotFoundError(f"none of {NIH_ENTRY_CSVS} under {root}")
        entry_csv = found[0]
    df = pd.read_csv(root / entry_csv)
    labels = df["Finding Labels"].astype(str)
    sex_col = "Patient Sex" if "Patient Sex" in df.columns else "Patient Gender"
    age = pd.to_numeric(df["Patient Age"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    out = pd.DataFrame(
        {
            "dataset": "nih_cxr14",
            "image_id": df["Image Index"].astype(str),
            "patient_id": df["Patient ID"].astype(str),
            "path": [str(root / image_dir / i) for i in df["Image Index"].astype(str)],
            "pneumothorax": labels.str.contains("Pneumothorax").astype(int),
            "pleural_effusion": labels.str.contains("Effusion").astype(int),
            "nodule_mass": (labels.str.contains("Nodule") | labels.str.contains("Mass")).astype(
                int
            ),
            "age": age.where(age < 120),
            "sex": df[sex_col].astype(str).str.upper().str[0],
            "view": df["View Position"].astype(str).str.upper(),
            "vendor": np.nan,
        }
    )
    out["view_label"] = out["view"].map(_VIEW_LABEL).fillna(-1).astype(int)
    return _finish(out)


def chexpert_manifest(root: Path, csv_name: str) -> pd.DataFrame:
    """CheXpert train.csv / valid.csv. Uncertain (-1) is preserved for masking (D-017)."""
    df = pd.read_csv(Path(root) / csv_name)
    path = df["Path"].astype(str)
    patient = path.str.extract(r"(patient\d+)")[0]

    def lab(col: str) -> pd.Series:
        v = pd.to_numeric(df.get(col), errors="coerce")
        return v.where(v.isin([0.0, 1.0, -1.0]))

    out = pd.DataFrame(
        {
            "dataset": "chexpert",
            "image_id": path,
            "patient_id": patient,
            "path": [str(Path(root).parent / p) for p in path],
            "pneumothorax": lab("Pneumothorax"),
            "pleural_effusion": lab("Pleural Effusion"),
            "nodule_mass": lab("Lung Lesion"),
            "age": pd.to_numeric(df["Age"], errors="coerce"),
            "sex": df["Sex"].astype(str).str.upper().str[0].replace({"U": np.nan}),
            "view": df["AP/PA"]
            .astype(str)
            .str.upper()
            .where(df["Frontal/Lateral"].eq("Frontal"), "LATERAL"),
            "vendor": np.nan,
        }
    )
    out["view_label"] = np.where(df["Frontal/Lateral"].eq("Frontal"), 0, 1)
    return _finish(out)


def vindr_manifest(root: Path, split: str = "train") -> pd.DataFrame:
    """VinDr-CXR. Train: majority vote of 3 radiologists; test: released consensus labels."""
    root = Path(root)
    csv = root / "annotations" / f"image_labels_{split}.csv"
    df = pd.read_csv(csv)
    cols = {
        "pneumothorax": "Pneumothorax",
        "pleural_effusion": "Pleural effusion",
        "nodule_mass": "Nodule/Mass",
    }
    if "rad_id" in df.columns:
        agg = df.groupby("image_id")[list(cols.values())].mean()
        lab = (agg >= 0.5).astype(int).reset_index()
    else:
        lab = df[["image_id", *cols.values()]].astype({c: int for c in cols.values()})
    out = pd.DataFrame(
        {
            "dataset": "vindr_cxr",
            "image_id": lab["image_id"].astype(str),
            "patient_id": lab["image_id"].astype(
                str
            ),  # one study per image; no patient id released
            "path": [str(root / split / f"{i}.dicom") for i in lab["image_id"].astype(str)],
            "pneumothorax": lab[cols["pneumothorax"]].astype(int),
            "pleural_effusion": lab[cols["pleural_effusion"]].astype(int),
            "nodule_mass": lab[cols["nodule_mass"]].astype(int),
            "view": "PA",
            "view_label": 0,
        }
    )
    return _finish(out)


def padchest_manifest(
    root: Path, csv_name: str, image_dir: str = "images", manual_only: bool = True
) -> pd.DataFrame:
    """PadChest. `Labels` is a python-list string; `MethodLabel` Physician vs RNN."""
    df = pd.read_csv(Path(root) / csv_name, low_memory=False)
    if manual_only:
        df = df[df["MethodLabel"].astype(str).str.lower().eq("physician")]
    labels = df["Labels"].astype(str).map(lambda s: [x.strip().lower() for x in _safe_list(s)])
    proj = df["Projection"].astype(str).str.upper()
    view = proj.replace(
        {"AP_HORIZONTAL": "AP", "L": "LATERAL", "COSTAL": "OTHER", "EXCLUDE": "OTHER", "UNK": ""}
    )
    year = pd.to_numeric(df["StudyDate_DICOM"].astype(str).str[:4], errors="coerce")
    birth = pd.to_numeric(df["PatientBirth"], errors="coerce")
    out = pd.DataFrame(
        {
            "dataset": "padchest",
            "image_id": df["ImageID"].astype(str),
            "patient_id": df["PatientID"].astype(str),
            "path": [
                str(Path(root) / image_dir / str(d) / i)
                for d, i in zip(df["ImageDir"], df["ImageID"], strict=True)
            ],
            "pneumothorax": labels.map(lambda ls: int("pneumothorax" in ls)),
            "pleural_effusion": labels.map(lambda ls: int("pleural effusion" in ls)),
            "nodule_mass": labels.map(
                lambda ls: int(any(x in ls for x in ("nodule", "mass", "multiple nodules")))
            ),
            "age": (year - birth).where(lambda a: (a >= 0) & (a < 120)),
            "sex": df["PatientSex_DICOM"].astype(str).str.upper().str[0].replace({"N": np.nan}),
            "view": view,
            "vendor": df["Manufacturer_DICOM"].astype(str).replace({"nan": np.nan}),
        }
    )
    out["view_label"] = (
        out["view"].map({"PA": 0, "AP": 0, "LATERAL": 1, "OTHER": 2}).fillna(-1).astype(int)
    )
    return _finish(out)


def _safe_list(s: str) -> list[str]:
    try:
        v = ast.literal_eval(s)
        return [str(x) for x in v] if isinstance(v, list | tuple) else []
    except (ValueError, SyntaxError):
        return []


# ---------------------------------------------------------------------------
# Gate-evaluation sets (datasets/OOD_EVAL_PLAN.md). Neither carries our finding
# labels, so pneumothorax / pleural_effusion / nodule_mass are blank (= unknown) and
# these manifests are never a training or finding-evaluation source. `dataset` is the
# key the gate evaluation groups on: kermany_pediatric is an out-of-population set
# (1-5-year-olds; frontal, so view_label stays 0), the NLM TB sets are adult frontal
# films from other sites/vendors and are expected to pass the gate.
# ---------------------------------------------------------------------------

_KERMANY_V3 = re.compile(r"^(NORMAL|BACTERIA|VIRUS)-(\d+)-(\d+)$", re.IGNORECASE)
_KERMANY_PNEUMONIA = re.compile(r"^person(\d+)_(bacteria|virus)_(\d+)$", re.IGNORECASE)
_KERMANY_NORMAL = re.compile(r"^(NORMAL2-)?IM-(\d+)-(\d+)(?:-\d+)?$", re.IGNORECASE)


def _kermany_patient(stem: str) -> str:
    """Best-effort grouping key; the release publishes no patient table.

    Mendeley v3 (ZhangLabData.zip) names files `<NORMAL|BACTERIA|VIRUS>-<id>-<seq>`; the
    middle number recurs across sequences *and* classes (809 of 3,081 ids on the real
    data), so it is used as the patient key. The Kaggle mirror's `person<N>_<class>_<k>`
    → `person<N>` and `[NORMAL2-]IM-<study>-<image>` → `[NORMAL2-]IM-<study>` are kept as
    fallbacks. Anything else is its own key.
    """
    m = _KERMANY_V3.match(stem)
    if m:
        return m.group(2)
    m = _KERMANY_PNEUMONIA.match(stem)
    if m:
        return f"person{m.group(1)}"
    m = _KERMANY_NORMAL.match(stem)
    if m:
        return f"{m.group(1) or ''}IM-{m.group(2)}".upper()
    return stem


def kermany_manifest(root: Path, image_dir: str = "chest_xray") -> pd.DataFrame:
    """Kermany paediatric CXR (Mendeley ZhangLabData v3).

    Layout: `chest_xray/{train,test}/{NORMAL,PNEUMONIA}/*.jpeg`.
    Population: 1-5-year-olds, AP films (Guangzhou). Role `ood_negative`: the gate should
    reject these as out-of-population (intended use is adults). `image_id` is the path
    relative to `image_dir`, so the split and NORMAL/PNEUMONIA class stay recoverable;
    per-image age/sex are not released (cohort age is recorded in the registry).
    """
    base = Path(root) / image_dir
    if not base.is_dir():
        raise FileNotFoundError(f"{base} not found (expected chest_xray/ from ZhangLabData.zip)")
    files = sorted(
        p for p in base.rglob("*") if p.is_file() and p.suffix.lower() in (".jpeg", ".jpg")
    )
    rel = [p.relative_to(base).as_posix() for p in files]
    out = pd.DataFrame(
        {
            "dataset": "kermany_pediatric",
            "image_id": rel,
            "patient_id": [_kermany_patient(p.stem) for p in files],
            "path": [str(p) for p in files],
            "age": np.nan,
            "sex": np.nan,
            "view": "AP",
            "vendor": np.nan,
            "view_label": 0,
        }
    )
    return _finish(out)


NLM_TB_SETS: dict[str, str] = {
    # registry id → sub-directory under data/raw/nlm_tb/ holding CXR_png/ + ClinicalReadings/
    "nlm_tb_shenzhen": "Shenzhen-Hospital-CXR-Set",
    "nlm_tb_montgomery": "Montgomery-County-CXR-Set/MontgomerySet",
}
_NLM_SEX = re.compile(r"\b(male|female|[MF])\b", re.IGNORECASE)
_NLM_AGE_Y = re.compile(r"(\d+)\s*(?:Y\b|yrs?\b|years?\b)", re.IGNORECASE)
_NLM_AGE_M = re.compile(r"(\d+)\s*(?:M\b|months?\b|mon\b)", re.IGNORECASE)
_NLM_AGE_D = re.compile(r"(\d+)\s*(?:D\b|days?\b)", re.IGNORECASE)


def _nlm_reading(text: str) -> tuple[str | float, float]:
    """Parse one ClinicalReadings file → (sex M/F/nan, age in years or nan).

    Shenzhen: `male 45yrs` / `female , 3 yrs` / `male 7month`; Montgomery:
    `Patient's Sex: F` / `Patient's Age: 043Y`. Sex is taken from the header lines only
    so finding text never leaks into it.
    """
    head = "\n".join(text.strip().splitlines()[:2])
    sex: str | float = np.nan
    m = _NLM_SEX.search(head)
    if m:
        sex = m.group(1)[0].upper()
    age = np.nan
    if (my := _NLM_AGE_Y.search(head)) is not None:
        age = float(my.group(1))
    elif (mm := _NLM_AGE_M.search(head)) is not None:
        age = round(float(mm.group(1)) / 12, 2)
    elif (md := _NLM_AGE_D.search(head)) is not None:
        age = round(float(md.group(1)) / 365, 3)
    if not (0 <= age < 120):
        age = np.nan
    return sex, age


def nlm_tb_manifest(root: Path, sets: tuple[str, ...] = tuple(NLM_TB_SETS)) -> pd.DataFrame:
    """NLM Shenzhen + Montgomery TB sets: one PA film per patient, `CXR_png/<id>_{0,1}.png`.

    Rows are enumerated from `ClinicalReadings/*.txt` (the metadata source; images need
    not be present to build the manifest — `path` is where the PNG lives). The trailing
    `_0`/`_1` in `image_id` is the TB label (normal / tuberculosis), which is not one of
    our findings and stays out of the label columns. Adult populations, other sites and
    vendors than the training data: the gate is expected to accept them.
    """
    root = Path(root)
    frames = []
    for ds in sets:
        sub = root / NLM_TB_SETS[ds]
        readings = sorted((sub / "ClinicalReadings").glob("*.txt"))
        if not readings:
            raise FileNotFoundError(f"no ClinicalReadings/*.txt under {sub}")
        parsed = [_nlm_reading(p.read_text(errors="replace")) for p in readings]
        stems = [p.stem for p in readings]
        frames.append(
            pd.DataFrame(
                {
                    "dataset": ds,
                    "image_id": [f"{s}.png" for s in stems],
                    "patient_id": stems,  # one image per patient in both sets
                    "path": [str(sub / "CXR_png" / f"{s}.png") for s in stems],
                    "age": [a for _, a in parsed],
                    "sex": [s for s, _ in parsed],
                    "view": "PA",
                    "vendor": np.nan,
                    "view_label": 0,
                }
            )
        )
    return _finish(pd.concat(frames, ignore_index=True))
