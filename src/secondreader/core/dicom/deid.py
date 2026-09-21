"""Profile-driven DICOM de-identification (allowlist strategy).

Pipeline per object:
  1. derive the pseudonym key from PatientID (+ issuer)
  2. compute the patient's deterministic date shift
  3. capture things we need *before* removal (age from DOB, original UIDs)
  4. walk the dataset (recursively through sequences): keep / replace / remove
  5. strip private tags, overlays, curves, and rebuild file meta
  6. stamp PatientIdentityRemoved + DeidentificationMethod(+CodeSequence)
  7. assess burned-in-text risk → quarantine flag

The output dataset is a new object; the input is never mutated.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydicom import Dataset, Sequence
from pydicom.datadict import keyword_for_tag
from pydicom.tag import BaseTag, Tag

from secondreader.core.dicom import burned_in, pseudonym
from secondreader.core.dicom.profile import DeidProfile
from secondreader.core.dicom.reader import new_file_meta, pixel_frames

_UID_TOKEN = "<uid>"  # noqa: S105 — template token, not a secret
_PSEUDO_TOKEN = "<pseudonym>"  # noqa: S105 — template token, not a secret
_DATE_TOKEN = "<shifted-date>"  # noqa: S105 — template token, not a secret
_DATETIME_TOKEN = "<shifted-datetime>"  # noqa: S105 — template token, not a secret
_CLEAN_TOKEN = "<cleaned>"  # noqa: S105 — template token, not a secret

# DICOM de-identification method codes (PS3.16 CID 7050)
_METHOD_CODES: tuple[tuple[str, str], ...] = (
    ("113100", "Basic Application Confidentiality Profile"),
    ("113105", "Clean Descriptors Option"),
    ("113107", "Retain Longitudinal Temporal Information Modified Dates Option"),
    ("113108", "Retain Patient Characteristics Option"),
)


@dataclass
class DeidReport:
    pseudo_key: str
    weak_identity: bool
    date_shift_days: int
    profile_hash: str
    kept: list[str] = field(default_factory=list)
    replaced: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    private_tags_removed: int = 0
    uid_map: dict[str, str] = field(default_factory=dict)  # original → de-identified
    quarantine: bool = False
    quarantine_reasons: tuple[str, ...] = ()
    burned_in_scores: dict[str, float] = field(default_factory=dict)


@dataclass
class DeidResult:
    dataset: Dataset
    report: DeidReport
    frames: NDArray[Any] | None  # decoded pixels, reused by the ingest pipeline


class Deidentifier:
    def __init__(self, profile: DeidProfile, secret: bytes) -> None:
        if len(secret) < 16:
            raise ValueError("pseudonym secret too short")
        self.profile = profile
        self._secret = secret

    # ---- public ---------------------------------------------------------------------------

    def deidentify(self, src: Dataset, *, decode_pixels: bool = True) -> DeidResult:
        ds = copy.deepcopy(src)
        pseudo, weak = self._pseudonym(ds)
        lo, hi = self.profile.date_shift_days_range
        shift = pseudonym.date_shift_days(self._secret, pseudo, lo, hi)
        report = DeidReport(
            pseudo_key=pseudo,
            weak_identity=weak,
            date_shift_days=shift,
            profile_hash=self.profile.content_hash,
        )

        frames: NDArray[Any] | None = None
        decode_error: str | None = None
        if decode_pixels and "PixelData" in ds:
            try:
                frames = pixel_frames(ds)
            except Exception as exc:  # noqa: BLE001 — any decode failure is a quarantine reason
                decode_error = f"pixel decode failed: {type(exc).__name__}"

        age = self._derive_age(ds)
        self._apply(ds, pseudo, shift, report, top_level=True)
        self._strip_groups(ds, report)
        if age is not None:
            ds.PatientAge = pseudonym.format_age(min(age, self.profile.max_age_years))
        self._stamp(ds)
        new_file_meta(ds, transfer_syntax=_transfer_syntax(src))

        assessment = burned_in.assess(ds, self.profile.burned_in, frames)
        reasons = list(assessment.reasons)
        if decode_error:
            reasons.append(decode_error)
        report.quarantine = bool(reasons)
        report.quarantine_reasons = tuple(reasons)
        report.burned_in_scores = assessment.scores
        return DeidResult(dataset=ds, report=report, frames=frames)

    # ---- internals ------------------------------------------------------------------------

    def _pseudonym(self, ds: Dataset) -> tuple[str, bool]:
        pid = str(ds.get("PatientID", "") or "").strip()
        issuer = str(ds.get("IssuerOfPatientID", "") or "").strip()
        if pid:
            return pseudonym.pseudonym_key(self._secret, pid, issuer), False
        name = str(ds.get("PatientName", "") or "")
        dob = str(ds.get("PatientBirthDate", "") or "")
        if not name and not dob:
            raise ValueError("cannot pseudonymize: no PatientID, PatientName or PatientBirthDate")
        return pseudonym.weak_pseudonym_key(self._secret, name, dob), True

    def _derive_age(self, ds: Dataset) -> int | None:
        if "PatientAge" in ds and str(ds.PatientAge).strip():
            parsed = pseudonym.parse_age_years(str(ds.PatientAge))
            if parsed is not None:
                return parsed
        dob = str(ds.get("PatientBirthDate", "") or "")
        ref = str(ds.get("StudyDate", "") or ds.get("SeriesDate", "") or "")
        if dob and ref:
            return pseudonym.age_from(dob, ref)
        return None

    def _apply(
        self, ds: Dataset, pseudo: str, shift: int, report: DeidReport, *, top_level: bool
    ) -> None:
        keep = self.profile.keep_tags
        replace = self.profile.replace_tags
        for tag in list(ds.keys()):
            t = Tag(tag)
            kw = keyword_for_tag(t) or f"({t.group:04X},{t.element:04X})"
            if t.is_private:
                continue  # handled by remove_private_tags
            if int(t) in replace:
                self._replace(ds, t, replace[int(t)], pseudo, shift, report)
                report.replaced.append(kw)
                continue
            if int(t) in keep:
                el = ds[t]
                if el.VR == "SQ":
                    for item in el.value:
                        self._apply(item, pseudo, shift, report, top_level=False)
                if top_level:
                    report.kept.append(kw)
                continue
            del ds[t]
            if top_level:
                report.removed.append(kw)
        # Type-2 replacements must exist even if the source lacked them.
        if top_level:
            for tag_int, template in replace.items():
                if tag_int not in ds and template in ("", _PSEUDO_TOKEN, "YES"):
                    self._set(ds, Tag(tag_int), template, pseudo)

    def _replace(
        self,
        ds: Dataset,
        t: BaseTag,
        template: str,
        pseudo: str,
        shift: int,
        report: DeidReport,
    ) -> None:
        el = ds[t]
        original = str(el.value) if el.value is not None else ""
        if template == _UID_TOKEN:
            new = pseudonym.remap_uid(self._secret, original) if original else ""
            if original:
                report.uid_map[original] = new
            el.value = new
        elif template == _PSEUDO_TOKEN:
            el.value = f"{pseudo}^" if el.VR == "PN" else pseudo
        elif template == _DATE_TOKEN:
            el.value = pseudonym.shift_da(original, shift)
        elif template == _DATETIME_TOKEN:
            el.value = pseudonym.shift_dt(original, shift)
        elif template == _CLEAN_TOKEN:
            norm = " ".join(original.upper().split())
            el.value = original if norm in self.profile.vocabulary else ""
        else:
            el.value = template

    def _set(self, ds: Dataset, t: BaseTag, template: str, pseudo: str) -> None:
        from pydicom.datadict import dictionary_VR

        vr = dictionary_VR(t)
        value: Any = template
        if template == _PSEUDO_TOKEN:
            value = f"{pseudo}^" if vr == "PN" else pseudo
        ds.add_new(t, vr, value)

    def _strip_groups(self, ds: Dataset, report: DeidReport) -> None:
        before = sum(1 for t in ds.keys() if Tag(t).is_private)
        ds.remove_private_tags()
        report.private_tags_removed = before
        # overlays 60xx, curves 50xx — pixel-level annotations may carry PHI
        for tag in list(ds.keys()):
            g = Tag(tag).group
            if 0x5000 <= g <= 0x50FF or 0x6000 <= g <= 0x60FF:
                del ds[tag]
                report.removed.append(f"({g:04X},{Tag(tag).element:04X})")

    def _stamp(self, ds: Dataset) -> None:
        ds.PatientIdentityRemoved = "YES"
        ds.DeidentificationMethod = self.profile.method_string
        seq = Sequence()
        for code, meaning in _METHOD_CODES:
            item = Dataset()
            item.CodeValue = code
            item.CodingSchemeDesignator = "DCM"
            item.CodeMeaning = meaning
            seq.append(item)
        ds.DeidentificationMethodCodeSequence = seq
        # LongitudinalTemporalInformationModified is the standard flag for shifted dates
        ds.LongitudinalTemporalInformationModified = "MODIFIED"


def _transfer_syntax(src: Dataset) -> str | None:
    fm = getattr(src, "file_meta", None)
    ts = getattr(fm, "TransferSyntaxUID", None) if fm is not None else None
    return str(ts) if ts else None


def frames_or_none(ds: Dataset) -> NDArray[Any] | None:
    try:
        return pixel_frames(ds)
    except Exception:  # noqa: BLE001
        return None


__all__ = ["DeidReport", "DeidResult", "Deidentifier", "frames_or_none", "np"]
