"""Native dataset CSVs → unified manifest rows. Column names follow each dataset's
public release; every adapter is pure (no downloads) and unit-tested on tiny
fixture CSVs. Verify against the current release when data arrives.
"""

from __future__ import annotations

import ast
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


def nih_manifest(
    root: Path, entry_csv: str = "Data_Entry_2017.csv", image_dir: str = "images"
) -> pd.DataFrame:
    """NIH ChestX-ray14: `Finding Labels` pipe-separated; Nodule|Mass → nodule_mass."""
    df = pd.read_csv(Path(root) / entry_csv)
    labels = df["Finding Labels"].astype(str)
    out = pd.DataFrame(
        {
            "dataset": "nih_cxr14",
            "image_id": df["Image Index"].astype(str),
            "patient_id": df["Patient ID"].astype(str),
            "path": [str(Path(root) / image_dir / i) for i in df["Image Index"].astype(str)],
            "pneumothorax": labels.str.contains("Pneumothorax").astype(int),
            "pleural_effusion": labels.str.contains("Effusion").astype(int),
            "nodule_mass": (labels.str.contains("Nodule") | labels.str.contains("Mass")).astype(
                int
            ),
            "age": pd.to_numeric(df["Patient Age"], errors="coerce").where(lambda a: a < 120),
            "sex": df["Patient Gender"].astype(str).str.upper().str[0],
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
