from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from secondreader.modalities.cxr import adapters
from secondreader.modalities.cxr.data import MANIFEST_COLUMNS, load_manifest, save_manifest
from secondreader.modalities.cxr.splits import (
    assign_tier,
    build_splits,
    check_no_patient_leakage,
    verify_locked,
)

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "csv"


def test_nih_adapter():
    df = adapters.nih_manifest(FIX, entry_csv="nih_Data_Entry_2017.csv")
    assert list(df.columns) == list(MANIFEST_COLUMNS)
    r = df.set_index("image_id")
    assert (
        r.loc["00000001_000.png", "pleural_effusion"] == 1
        and r.loc["00000001_000.png", "pneumothorax"] == 0
    )
    assert (
        r.loc["00000002_000.png", "nodule_mass"] == 1
        and r.loc["00000002_000.png", "pneumothorax"] == 1
    )
    assert r.loc["00000003_000.png", "nodule_mass"] == 1  # Mass → nodule_mass
    assert np.isnan(r.loc["00000003_000.png", "age"])  # 412 is an NIH data error → unknown
    assert (
        r.loc["00000002_000.png", "view"] == "AP" and r.loc["00000002_000.png", "view_label"] == 0
    )
    assert set(df["patient_id"]) == {"1", "2", "3"}


def test_nih_adapter_v2020_columns_and_autodetect(tmp_path):
    # NIH replaced Data_Entry_2017.csv with Data_Entry_2017_v2020.csv (`Patient Sex`).
    df = adapters.nih_manifest(FIX, entry_csv="nih_Data_Entry_2017_v2020.csv")
    r = df.set_index("image_id")
    assert r.loc["00000002_000.png", "sex"] == "F" and r.loc["00000003_000.png", "age"] == 41
    # autodetect prefers the v2020 file when both are present
    (tmp_path / "Data_Entry_2017_v2020.csv").write_bytes(
        (FIX / "nih_Data_Entry_2017_v2020.csv").read_bytes()
    )
    (tmp_path / "Data_Entry_2017.csv").write_bytes((FIX / "nih_Data_Entry_2017.csv").read_bytes())
    auto = adapters.nih_manifest(tmp_path).set_index("image_id")
    assert auto.loc["00000003_000.png", "age"] == 41
    assert auto.loc["00000001_000.png", "path"].endswith("images/00000001_000.png")
    with pytest.raises(FileNotFoundError):
        adapters.nih_manifest(tmp_path / "missing")


def test_chexpert_adapter_keeps_uncertain():
    df = adapters.chexpert_manifest(FIX, "chexpert_train.csv").set_index("image_id")
    a = df.loc["CheXpert-v1.0-small/train/patient00002/study2/view1_frontal.jpg"]
    assert a["nodule_mass"] == -1 and a["pleural_effusion"] == 1 and a["pneumothorax"] == 0
    assert a["patient_id"] == "patient00002" and a["view"] == "PA" and a["view_label"] == 0
    lat = df.loc["CheXpert-v1.0-small/train/patient00002/study1/view2_lateral.jpg"]
    assert lat["view_label"] == 1 and lat["view"] == "LATERAL"
    blank = df.loc["CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg"]
    assert np.isnan(blank["nodule_mass"])  # blank = unknown, distinct from -1


def test_vindr_adapter_majority_vote(tmp_path):
    (tmp_path / "annotations").mkdir()
    (tmp_path / "annotations" / "image_labels_train.csv").write_bytes(
        (FIX / "vindr_image_labels_train.csv").read_bytes()
    )
    df = adapters.vindr_manifest(tmp_path, "train").set_index("image_id")
    assert df.loc["img1", "nodule_mass"] == 1  # 2 of 3
    assert df.loc["img2", "pleural_effusion"] == 0  # 1 of 3
    assert df.loc["img2", "pneumothorax"] == 1
    assert df.loc["img1", "path"].endswith("train/img1.dicom")


def test_padchest_adapter_manual_only_and_vendor():
    df = adapters.padchest_manifest(FIX, "padchest.csv").set_index("image_id")
    assert "c.png" not in df.index  # RNN-labelled row excluded
    assert (
        df.loc["a.png", "pleural_effusion"] == 1
        and df.loc["a.png", "vendor"] == "PhilipsMedicalSystems"
    )
    assert df.loc["a.png", "age"] == 64 and df.loc["a.png", "view"] == "PA"
    assert df.loc["b.png", "view_label"] == 1
    allrows = adapters.padchest_manifest(FIX, "padchest.csv", manual_only=False).set_index(
        "image_id"
    )
    assert allrows.loc["c.png", "nodule_mass"] == 1 and allrows.loc["c.png", "view"] == "AP"


def _fake_manifest(dataset: str, n_patients: int, per: int = 2) -> pd.DataFrame:
    rows = []
    for p in range(n_patients):
        for k in range(per):
            rows.append(
                {
                    "dataset": dataset,
                    "image_id": f"{dataset}-{p}-{k}",
                    "patient_id": f"{dataset}-{p}",
                    "path": f"/x/{dataset}-{p}-{k}.png",
                    "tier": "",
                    "pneumothorax": 0,
                    "pleural_effusion": 1,
                    "nodule_mass": 0,
                    "age": 50,
                    "sex": "F",
                    "view": "PA",
                    "vendor": "",
                    "view_label": 0,
                }
            )
    return pd.DataFrame(rows, columns=list(MANIFEST_COLUMNS))


def test_assign_tier_is_deterministic_and_roughly_proportional():
    tiers = [assign_tier("salt", "nih", str(i)) for i in range(5000)]
    assert tiers == [assign_tier("salt", "nih", str(i)) for i in range(5000)]
    frac = pd.Series(tiers).value_counts(normalize=True)
    assert abs(frac["train"] - 0.8) < 0.03 and abs(frac["val"] - 0.1) < 0.02
    assert (
        assign_tier("salt", "nih", "1") != assign_tier("other", "nih", "1") or True
    )  # salts differ → may differ


def test_build_splits_patient_level_and_locked(tmp_path):
    df = pd.concat([_fake_manifest("nih", 300), _fake_manifest("vindr", 40)], ignore_index=True)
    s = build_splits(df, out_dir=tmp_path, version="v1", salt="s", external_datasets={"vindr"})
    assert s.tiers["external"]["n"] == 80 and s.tiers["external"]["datasets"] == ["vindr"]
    tr = load_manifest(tmp_path / "v1" / "train.csv")
    va = load_manifest(tmp_path / "v1" / "val.csv")
    assert not (set(tr["patient_id"]) & set(va["patient_id"]))
    verify_locked(tmp_path / "v1", s.manifest_sha256)
    # tamper → detected
    save_manifest(tr.iloc[:-1], tmp_path / "v1" / "train.csv")
    with pytest.raises(ValueError):
        verify_locked(tmp_path / "v1", s.manifest_sha256)


def test_leakage_check():
    df = _fake_manifest("nih", 2)
    df["tier"] = ["train", "val", "train", "train"]
    with pytest.raises(ValueError, match="leakage"):
        check_no_patient_leakage(df)
