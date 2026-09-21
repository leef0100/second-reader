"""Deterministic, keyed pseudonymization primitives.

All functions are pure given the secret, so the same patient / UID / date maps
to the same de-identified value across ingests — that is what makes prior-study
linkage possible without ever storing the real identifier.
"""

from __future__ import annotations

import hashlib
import hmac
from datetime import date, datetime, timedelta


def _mac(secret: bytes, *parts: str) -> bytes:
    return hmac.new(secret, "\x1f".join(parts).encode("utf-8"), hashlib.sha256).digest()


def pseudonym_key(secret: bytes, patient_id: str, issuer: str = "") -> str:
    """32-hex-char patient key. Never reversible without the secret."""
    return _mac(secret, "patient", issuer.strip(), patient_id.strip()).hex()[:32]


def weak_pseudonym_key(secret: bytes, name: str, birth_date: str) -> str:
    """Fallback when PatientID is absent. Flagged as weak by the caller."""
    return _mac(secret, "patient-weak", name.strip().upper(), birth_date.strip()).hex()[:32]


def remap_uid(secret: bytes, uid: str) -> str:
    """Map a UID to a valid, deterministic 2.25.<uuid-int> UID (<= 44 chars)."""
    digest = _mac(secret, "uid", uid.strip())[:16]
    return f"2.25.{int.from_bytes(digest, 'big')}"


def date_shift_days(secret: bytes, pseudo_key: str, lo: int, hi: int) -> int:
    """Per-patient shift in [-hi, -lo] days; identical for every study of the patient."""
    digest = _mac(secret, "dateshift", pseudo_key)
    span = hi - lo + 1
    return -(lo + int.from_bytes(digest[:8], "big") % span)


def shift_da(value: str, days: int) -> str:
    """Shift a DICOM DA (YYYYMMDD). Returns '' for unparsable input rather than leaking."""
    v = value.strip()
    if len(v) != 8 or not v.isdigit():
        return ""
    try:
        d = date(int(v[:4]), int(v[4:6]), int(v[6:8]))
    except ValueError:
        return ""
    return (d + timedelta(days=days)).strftime("%Y%m%d")


def shift_dt(value: str, days: int) -> str:
    """Shift the date portion of a DICOM DT; keeps time/fraction/offset if present."""
    v = value.strip()
    if len(v) < 8 or not v[:8].isdigit():
        return ""
    head = shift_da(v[:8], days)
    if not head:
        return ""
    return head + v[8:]


def age_from(birth: str, at: str) -> int | None:
    try:
        b = datetime.strptime(birth.strip(), "%Y%m%d").date()
        a = datetime.strptime(at.strip(), "%Y%m%d").date()
    except ValueError:
        return None
    years = a.year - b.year - ((a.month, a.day) < (b.month, b.day))
    return years if years >= 0 else None


def parse_age_years(value: str) -> int | None:
    """DICOM AS: nnnD | nnnW | nnnM | nnnY."""
    v = value.strip().upper()
    if len(v) != 4 or not v[:3].isdigit():
        return None
    n, unit = int(v[:3]), v[3]
    if unit == "Y":
        return n
    if unit == "M":
        return n // 12
    if unit == "W":
        return n // 52
    if unit == "D":
        return n // 365
    return None


def format_age(years: int) -> str:
    return f"{min(max(years, 0), 999):03d}Y"
