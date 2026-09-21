"""PHI removal tests. These are the tests that matter most in this repo."""

from __future__ import annotations

import pytest
from pydicom.tag import Tag

from secondreader.core.dicom import dataset_to_bytes, find_phi, read_dicom
from secondreader.core.dicom.pseudonym import remap_uid
from tests.fixtures.synthetic import (
    FAKE_PHI,
    SyntheticSpec,
    cxr_spec,
    dx_spec,
    make_dataset,
    secondary_capture_spec,
    ultrasound_cine_spec,
)

PHI_TAGS = list(FAKE_PHI)


@pytest.fixture
def original():
    return make_dataset(cxr_spec())


@pytest.fixture
def result(deidentifier, original):
    return deidentifier.deidentify(original)


def test_all_named_phi_attributes_are_gone_or_emptied(result):
    ds = result.dataset
    for kw in PHI_TAGS:
        if kw in (
            "ReferringPhysicianName",
            "AccessionNumber",
            "StudyID",
            "PatientName",
            "PatientID",
        ):
            continue  # Type-2 / replaced — checked below
        assert kw not in ds, f"{kw} survived de-identification"
    assert str(ds.ReferringPhysicianName) == ""
    assert str(ds.AccessionNumber) == ""
    assert str(ds.StudyID) == ""


def test_no_original_phi_value_appears_anywhere(result, deid_profile):
    values = list(FAKE_PHI.values()) + ["Jane", "Doe", "DOE"]
    problems = find_phi(result.dataset, deid_profile, known_values=values)
    assert problems == []


def test_verifier_catches_leaks(deid_profile, result):
    ds = result.dataset
    ds.add_new(Tag("PatientBirthDate"), "DA", "19700315")
    problems = find_phi(ds, deid_profile, known_values=["19700315"])
    assert any("PatientBirthDate" in p for p in problems)


def test_patient_name_and_id_are_pseudonym(result):
    ds = result.dataset
    assert str(ds.PatientID) == result.report.pseudo_key
    assert str(ds.PatientName) == result.report.pseudo_key
    assert len(result.report.pseudo_key) == 32
    assert "DOE" not in str(ds.PatientName).upper()


def test_pseudonym_is_stable_and_secret_dependent(deid_profile, secret):
    from secondreader.core.dicom import Deidentifier

    a = Deidentifier(deid_profile, secret).deidentify(make_dataset(cxr_spec()))
    b = Deidentifier(deid_profile, secret).deidentify(make_dataset(cxr_spec(seed=5)))
    c = Deidentifier(deid_profile, b"a-different-secret-value-xxxxxxxx").deidentify(
        make_dataset(cxr_spec())
    )
    assert a.report.pseudo_key == b.report.pseudo_key  # same patient → same key
    assert a.report.pseudo_key != c.report.pseudo_key  # different secret → different key


def test_uids_are_remapped_deterministically(result, original, secret):
    ds = result.dataset
    for kw in ("StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID", "FrameOfReferenceUID"):
        new, old = str(ds[kw].value), str(original[kw].value)
        assert new != old
        assert new == remap_uid(secret, old)
        assert new.startswith("2.25.") and len(new) <= 64
    assert ds.file_meta.MediaStorageSOPInstanceUID == ds.SOPInstanceUID
    assert "SourceApplicationEntityTitle" not in ds.file_meta


def test_same_patient_studies_share_key_and_date_shift(deidentifier):
    s1 = make_dataset(cxr_spec(study_date="20240102"))
    s2 = make_dataset(cxr_spec(study_date="20240301"))
    r1, r2 = deidentifier.deidentify(s1), deidentifier.deidentify(s2)
    assert r1.report.pseudo_key == r2.report.pseudo_key
    assert r1.report.date_shift_days == r2.report.date_shift_days
    assert r1.report.date_shift_days < 0
    # ordering preserved → priors still sort correctly
    assert str(r1.dataset.StudyDate) < str(r2.dataset.StudyDate)
    assert str(r1.dataset.StudyDate) != "20240102"


def test_dates_are_shifted_consistently_within_object(result):
    ds = result.dataset
    assert ds.StudyDate == ds.SeriesDate == ds.ContentDate
    assert str(ds.AcquisitionDateTime).startswith(str(ds.StudyDate))
    assert str(ds.AcquisitionDateTime).endswith("101500.000000")
    assert str(ds.LongitudinalTemporalInformationModified) == "MODIFIED"


def test_birth_date_removed_but_age_kept(result):
    ds = result.dataset
    assert "PatientBirthDate" not in ds
    assert str(ds.PatientAge) == "054Y"


def test_age_derived_from_dob_when_missing(deidentifier):
    ds = make_dataset(cxr_spec(patient_age=None, study_date="20240102"))
    r = deidentifier.deidentify(ds)
    assert str(r.dataset.PatientAge) == "053Y"  # born 1970-03-15, study 2024-01-02


def test_age_capped_at_89(deidentifier):
    ds = make_dataset(cxr_spec(patient_age="094Y"))
    r = deidentifier.deidentify(ds)
    assert str(r.dataset.PatientAge) == "089Y"


def test_private_tags_overlays_and_nested_phi_removed(result, original):
    ds = result.dataset
    assert result.report.private_tags_removed >= 1
    assert not any(Tag(t).is_private for t in ds.keys())
    assert not any(0x6000 <= Tag(t).group <= 0x60FF for t in ds.keys())
    assert "RequestAttributesSequence" not in ds
    item = ds.AnatomicRegionSequence[0]
    assert "OperatorsName" not in item
    assert str(item.CodeMeaning) == "Chest"


def test_descriptions_cleaned_by_vocabulary(result):
    ds = result.dataset
    assert str(ds.StudyDescription) == "CHEST PA"  # in vocabulary → kept
    assert str(ds.SeriesDescription) == ""  # free text → emptied


def test_clinical_and_pixel_attributes_are_preserved(result, original):
    ds = result.dataset
    for kw in (
        "Modality",
        "PatientSex",
        "BodyPartExamined",
        "ViewPosition",
        "Manufacturer",
        "ManufacturerModelName",
        "Rows",
        "Columns",
        "BitsStored",
        "PhotometricInterpretation",
        "PixelSpacing",
        "WindowCenter",
        "WindowWidth",
        "KVP",
    ):
        assert ds[kw].value == original[kw].value, kw
    assert bytes(ds.PixelData) == bytes(original.PixelData)


def test_stamped_with_method_codes(result, deid_profile):
    ds = result.dataset
    assert str(ds.PatientIdentityRemoved) == "YES"
    assert deid_profile.content_hash[:12] in str(ds.DeidentificationMethod)
    codes = {str(i.CodeValue) for i in ds.DeidentificationMethodCodeSequence}
    assert {"113100", "113105", "113107", "113108"} <= codes


def test_input_dataset_is_not_mutated(deidentifier, original):
    before = dataset_to_bytes(original)
    deidentifier.deidentify(original)
    assert dataset_to_bytes(original) == before
    assert str(original.PatientName) == FAKE_PHI["PatientName"]


def test_roundtrip_through_bytes_still_clean(result, deid_profile):
    data = dataset_to_bytes(result.dataset)
    assert b"DOE" not in data and b"MRN-0012345" not in data and b"Fictional" not in data
    ds = read_dicom(data)
    assert find_phi(ds, deid_profile, known_values=FAKE_PHI.values()) == []


def test_weak_identity_when_patient_id_missing(deidentifier):
    phi = {k: v for k, v in FAKE_PHI.items() if k != "PatientID"}
    r = deidentifier.deidentify(make_dataset(cxr_spec(), phi=phi))
    assert r.report.weak_identity is True
    assert "DOE" not in str(r.dataset.PatientName).upper()


def test_refuses_when_no_identity_at_all(deidentifier):
    phi = {
        k: v
        for k, v in FAKE_PHI.items()
        if k not in ("PatientID", "PatientName", "PatientBirthDate")
    }
    with pytest.raises(ValueError):
        deidentifier.deidentify(make_dataset(cxr_spec(), phi=phi))


# ---- burned-in text / quarantine ---------------------------------------------------------


def test_clean_cxr_is_not_quarantined(result):
    assert result.report.quarantine is False
    assert result.report.quarantine_reasons == ()


def test_burned_in_annotation_tag_quarantines(deidentifier):
    r = deidentifier.deidentify(make_dataset(cxr_spec(burned_in_tag="YES")))
    assert r.report.quarantine
    assert any("BurnedInAnnotation" in x for x in r.report.quarantine_reasons)


def test_secondary_capture_quarantines(deidentifier):
    r = deidentifier.deidentify(make_dataset(secondary_capture_spec()))
    assert r.report.quarantine
    assert any(
        "secondary-capture" in x or "screen-capture" in x for x in r.report.quarantine_reasons
    )


def test_pixel_heuristic_flags_burned_text(deidentifier):
    r = deidentifier.deidentify(make_dataset(cxr_spec(burn_text=True)))
    assert r.report.quarantine
    assert any("pixel heuristic" in x for x in r.report.quarantine_reasons)


def test_dx_and_monochrome1_handled(deidentifier, deid_profile):
    r = deidentifier.deidentify(make_dataset(dx_spec(photometric="MONOCHROME1")))
    assert find_phi(r.dataset, deid_profile, known_values=FAKE_PHI.values()) == []
    assert not r.report.quarantine


def test_multiframe_ultrasound_deidentifies_and_keeps_frames(deidentifier, deid_profile):
    """The pipeline must not assume single-frame — ultrasound cine comes later."""
    r = deidentifier.deidentify(make_dataset(ultrasound_cine_spec()))
    assert r.frames is not None and r.frames.shape[0] == 8
    assert int(r.dataset.NumberOfFrames) == 8
    assert find_phi(r.dataset, deid_profile, known_values=FAKE_PHI.values()) == []


def test_profile_hash_is_content_hash(deid_profile, repo_root):
    import hashlib

    text = (repo_root / "configs/deid/basic_profile.yaml").read_text()
    assert deid_profile.content_hash == hashlib.sha256(text.encode()).hexdigest()


@pytest.mark.parametrize(
    "spec", [SyntheticSpec(rows=64, cols=64), SyntheticSpec(bits=8, rows=300, cols=200)]
)
def test_various_geometries(deidentifier, deid_profile, spec):
    r = deidentifier.deidentify(make_dataset(spec))
    assert find_phi(r.dataset, deid_profile, known_values=FAKE_PHI.values()) == []
