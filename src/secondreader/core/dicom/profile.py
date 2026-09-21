"""De-identification profile: a validated, content-hashed YAML document."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator
from pydicom.datadict import tag_for_keyword


class PixelHeuristicConfig(BaseModel):
    enabled: bool = True
    border_fraction: float = 0.08
    bright_quantile: float = 0.985
    bright_fraction_threshold: float = 0.02
    edge_contrast_ratio: float = 2.5


class BurnedInConfig(BaseModel):
    quarantine_if_tag_yes: bool = True
    quarantine_modalities: list[str] = Field(default_factory=list)
    quarantine_sop_classes: list[str] = Field(default_factory=list)
    pixel_heuristic: PixelHeuristicConfig = Field(default_factory=PixelHeuristicConfig)


class DeidProfile(BaseModel):
    name: str
    version: int
    keep: list[str]
    replace: dict[str, str]
    remove: list[str] = Field(default_factory=list)
    description_vocabulary: list[str] = Field(default_factory=list)
    max_age_years: int = 89
    date_shift_days_range: tuple[int, int] = (1, 365)
    burned_in: BurnedInConfig = Field(default_factory=BurnedInConfig)

    # populated by `load_profile`; sha256 of the canonical YAML text
    content_hash: str = ""

    @field_validator("keep", "remove")
    @classmethod
    def _keywords_exist(cls, v: list[str]) -> list[str]:
        unknown = [k for k in v if tag_for_keyword(k) is None]
        if unknown:
            raise ValueError(f"unknown DICOM keywords: {unknown}")
        return v

    @field_validator("replace")
    @classmethod
    def _replace_keywords_exist(cls, v: dict[str, str]) -> dict[str, str]:
        unknown = [k for k in v if tag_for_keyword(k) is None]
        if unknown:
            raise ValueError(f"unknown DICOM keywords: {unknown}")
        return v

    @property
    def keep_tags(self) -> frozenset[int]:
        return frozenset(_tag(k) for k in self.keep)

    @property
    def replace_tags(self) -> dict[int, str]:
        return {_tag(k): v for k, v in self.replace.items()}

    @property
    def remove_tags(self) -> frozenset[int]:
        return frozenset(_tag(k) for k in self.remove)

    @property
    def vocabulary(self) -> frozenset[str]:
        return frozenset(_norm(v) for v in self.description_vocabulary)

    @property
    def method_string(self) -> str:
        return f"secondreader deid {self.name} v{self.version} {self.content_hash[:12]}"


def _tag(keyword: str) -> int:
    t = tag_for_keyword(keyword)
    assert t is not None
    return int(t)


def _norm(s: str) -> str:
    return " ".join(s.upper().split())


def load_profile(path: Path | str) -> DeidProfile:
    text = Path(path).read_text(encoding="utf-8")
    data: dict[str, Any] = yaml.safe_load(text)
    profile = DeidProfile.model_validate(data)
    profile.content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return profile
