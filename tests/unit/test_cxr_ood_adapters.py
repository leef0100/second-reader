"""Adapters for the gate-evaluation sets: labels blank, population/site encoded in `dataset`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from secondreader.modalities.cxr import adapters
from secondreader.modalities.cxr.config import FINDING_CODES
from secondreader.modalities.cxr.data import MANIFEST_COLUMNS, load_manifest, save_manifest

FIX = Path(__file__).resolve().parents[1] / "fixtures"


def _kermany_tree(root: Path) -> None:
    files = {
        # Mendeley v3 naming: <CLASS>-<patient id>-<seq>
        "train/NORMAL/NORMAL-1049278-0001.jpeg",
        "train/NORMAL/NORMAL-1049278-0002.jpeg",
        "train/PNEUMONIA/BACTERIA-1049278-0001.jpeg",  # same patient, later visit
        "train/PNEUMONIA/VIRUS-1220485-0001.jpeg",
        # Kaggle-mirror naming, kept as fallbacks
        "train/NORMAL/IM-0115-0001.jpeg",
        "train/NORMAL/IM-0115-0002.jpeg",
        "train/NORMAL/NORMAL2-IM-0315-0001.jpeg",
        "train/PNEUMONIA/person1_bacteria_1.jpeg",
        "train/PNEUMONIA/person1_bacteria_2.jpeg",
        "train/PNEUMONIA/person1000_virus_1681.jpeg",
        "test/NORMAL/IM-0001-0001.jpeg",
        "test/PNEUMONIA/person3_virus_15.jpeg",
        "test/PNEUMONIA/notes.txt",  # ignored: not an image
    }
    for f in files:
        p = root / "chest_xray" / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"\xff\xd8\xff")


def test_kermany_adapter(tmp_path):
    _kermany_tree(tmp_path)
    df = adapters.kermany_manifest(tmp_path)
    assert list(df.columns) == list(MANIFEST_COLUMNS) and len(df) == 12
    assert (df["dataset"] == "kermany_pediatric").all()
    # paediatric frontal films: still frontal (view_label 0); population flagged via `dataset`
    assert (df["view_label"] == 0).all() and (df["view"] == "AP").all()
    # no finding labels at all — unknown, never 0
    for c in FINDING_CODES:
        assert df[c].isna().all()
    r = df.set_index("image_id")
    # v3: the middle number groups sequences and classes of the same patient
    assert r.loc["train/NORMAL/NORMAL-1049278-0001.jpeg", "patient_id"] == "1049278"
    assert r.loc["train/NORMAL/NORMAL-1049278-0002.jpeg", "patient_id"] == "1049278"
    assert r.loc["train/PNEUMONIA/BACTERIA-1049278-0001.jpeg", "patient_id"] == "1049278"
    assert r.loc["train/PNEUMONIA/VIRUS-1220485-0001.jpeg", "patient_id"] == "1220485"
    assert r.loc["train/PNEUMONIA/person1_bacteria_1.jpeg", "patient_id"] == "person1"
    assert r.loc["train/PNEUMONIA/person1_bacteria_2.jpeg", "patient_id"] == "person1"
    assert r.loc["train/PNEUMONIA/person1000_virus_1681.jpeg", "patient_id"] == "person1000"
    assert r.loc["train/NORMAL/IM-0115-0001.jpeg", "patient_id"] == "IM-0115"
    assert r.loc["train/NORMAL/IM-0115-0002.jpeg", "patient_id"] == "IM-0115"
    assert r.loc["train/NORMAL/NORMAL2-IM-0315-0001.jpeg", "patient_id"] == "NORMAL2-IM-0315"
    assert r.loc["test/NORMAL/IM-0001-0001.jpeg", "path"].endswith(
        "chest_xray/test/NORMAL/IM-0001-0001.jpeg"
    )
    assert "test/PNEUMONIA/notes.txt" not in r.index
    with pytest.raises(FileNotFoundError):
        adapters.kermany_manifest(tmp_path / "missing")
    # survives the manifest CSV round trip used by every other dataset
    sha = save_manifest(df, tmp_path / "m.csv")
    back = load_manifest(tmp_path / "m.csv")
    assert len(sha) == 64 and len(back) == 12 and back["nodule_mass"].isna().all()


def test_nlm_tb_adapter_parses_both_reading_formats():
    df = adapters.nlm_tb_manifest(FIX / "nlm_tb")
    assert list(df.columns) == list(MANIFEST_COLUMNS) and len(df) == 5
    assert set(df["dataset"]) == {"nlm_tb_shenzhen", "nlm_tb_montgomery"}
    assert (df["view_label"] == 0).all() and (df["view"] == "PA").all()
    for c in FINDING_CODES:
        assert df[c].isna().all()  # TB status is not one of our findings
    r = df.set_index("image_id")
    assert r.loc["CHNCXR_0001_0.png", "sex"] == "M" and r.loc["CHNCXR_0001_0.png", "age"] == 45
    assert r.loc["CHNCXR_0327_1.png", "sex"] == "F" and r.loc["CHNCXR_0327_1.png", "age"] == 27
    assert r.loc["CHNCXR_0500_1.png", "age"] == 3  # "male , 3 yrs"
    assert r.loc["MCUCXR_0001_0.png", "sex"] == "F" and r.loc["MCUCXR_0001_0.png", "age"] == 43
    assert r.loc["MCUCXR_0104_1.png", "sex"] == "M" and r.loc["MCUCXR_0104_1.png", "age"] == 60
    assert r.loc["MCUCXR_0104_1.png", "dataset"] == "nlm_tb_montgomery"
    assert r.loc["CHNCXR_0327_1.png", "path"].endswith(
        "Shenzhen-Hospital-CXR-Set/CXR_png/CHNCXR_0327_1.png"
    )
    assert r.loc["MCUCXR_0001_0.png", "path"].endswith("MontgomerySet/CXR_png/MCUCXR_0001_0.png")
    assert (df["patient_id"] == df["image_id"].str.replace(".png", "", regex=False)).all()
    # one set alone
    sh = adapters.nlm_tb_manifest(FIX / "nlm_tb", sets=("nlm_tb_shenzhen",))
    assert len(sh) == 3 and (sh["dataset"] == "nlm_tb_shenzhen").all()
    with pytest.raises(FileNotFoundError):
        adapters.nlm_tb_manifest(FIX / "nowhere")


@pytest.mark.parametrize(
    "text,sex,age",
    [
        ("male 45yrs\nnormal", "M", 45.0),
        ("female , 3 yrs\nPTB", "F", 3.0),
        ("male 7month\nnormal", "M", 0.58),
        ("Patient's Sex: F\nPatient's Age: 043Y\n\nnormal", "F", 43.0),
        ("Patient's Sex: M\nPatient's Age: 011M\n\nnormal", "M", 0.92),
        ("unknown\nnormal", np.nan, np.nan),
        ("female 412yrs\nnormal", "F", np.nan),  # impossible age → unknown
    ],
)
def test_nlm_reading_parser(text, sex, age):
    got_sex, got_age = adapters._nlm_reading(text)
    assert (got_sex == sex) if isinstance(sex, str) else np.isnan(got_sex)
    assert (got_age == pytest.approx(age, abs=0.01)) if not np.isnan(age) else np.isnan(got_age)
