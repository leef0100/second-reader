"""Independent verification that a dataset is de-identified.

Deliberately does NOT reuse the Deidentifier's own bookkeeping: it re-walks the
output and checks it against the profile and against known original values.
Used by tests and by the ingest pipeline as a final gate before storage.
"""

from __future__ import annotations

from collections.abc import Iterable

from pydicom import Dataset
from pydicom.datadict import keyword_for_tag
from pydicom.tag import Tag

from secondreader.core.dicom.profile import DeidProfile

# Attributes that must never survive, independent of profile contents.
_HARD_FORBIDDEN = (
    "PatientBirthDate",
    "PatientAddress",
    "PatientTelephoneNumbers",
    "OtherPatientIDs",
    "OtherPatientNames",
    "InstitutionName",
    "InstitutionAddress",
    "StationName",
    "DeviceSerialNumber",
    "OperatorsName",
    "PerformingPhysicianName",
    "ReferringPhysicianName",  # allowed only if empty
    "AccessionNumber",  # allowed only if empty
)

_STAMP_TAGS = frozenset(
    int(Tag(t))
    for t in (
        "PatientIdentityRemoved",
        "DeidentificationMethod",
        "DeidentificationMethodCodeSequence",
        "LongitudinalTemporalInformationModified",
    )
)


def find_phi(
    ds: Dataset,
    profile: DeidProfile,
    known_values: Iterable[str] = (),
    known_uids: Iterable[str] = (),
) -> list[str]:
    """Return a list of problems; empty means the dataset passes."""
    problems: list[str] = []
    needles = [v.strip().lower() for v in known_values if v and len(v.strip()) >= 3]
    uids = {u.strip() for u in known_uids if u}
    allowed = profile.keep_tags | set(profile.replace_tags) | _STAMP_TAGS

    def walk(d: Dataset, path: str) -> None:
        for tag in d.keys():
            t = Tag(tag)
            kw = keyword_for_tag(t) or f"({t.group:04X},{t.element:04X})"
            here = f"{path}{kw}"
            if t.is_private:
                problems.append(f"private tag present: {here}")
                continue
            if 0x5000 <= t.group <= 0x50FF or 0x6000 <= t.group <= 0x60FF:
                problems.append(f"overlay/curve present: {here}")
                continue
            if int(t) not in allowed:
                problems.append(f"attribute not in allowlist: {here}")
            el = d[t]
            if el.VR == "SQ":
                for i, item in enumerate(el.value):
                    walk(item, f"{here}[{i}].")
                continue
            if el.VR in ("OB", "OW", "OF", "OD", "OL", "OV", "UN"):
                continue
            text = _as_text(el.value)
            low = text.lower()
            for n in needles:
                if n in low:
                    problems.append(f"original value {n!r} found in {here}")
            if el.VR == "UI" and text in uids:
                problems.append(f"original UID retained in {here}")

    walk(ds, "")

    for kw in _HARD_FORBIDDEN:
        if kw in ds:
            v = ds.get(kw)
            if kw in ("ReferringPhysicianName", "AccessionNumber"):
                if str(v or "").strip():
                    problems.append(f"{kw} must be empty")
            else:
                problems.append(f"forbidden attribute present: {kw}")

    if str(ds.get("PatientIdentityRemoved", "")).upper() != "YES":
        problems.append("PatientIdentityRemoved != YES")
    age = str(ds.get("PatientAge", "") or "")
    if age[:3].isdigit() and age.endswith("Y") and int(age[:3]) > profile.max_age_years:
        problems.append(f"PatientAge {age} exceeds cap")

    fm = getattr(ds, "file_meta", None)
    if fm is not None and "SourceApplicationEntityTitle" in fm:
        problems.append("file meta retains SourceApplicationEntityTitle")
    return problems


def _as_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("latin-1", errors="ignore")
    if isinstance(value, list | tuple) or type(value).__name__ == "MultiValue":
        return "\\".join(str(v) for v in value)  # type: ignore[attr-defined]
    return str(value)
