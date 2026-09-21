"""Content hashing for traceability (principle 7)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    """Hash of a canonical JSON encoding (sorted keys, no whitespace)."""
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def model_version_id(
    weights_sha256: str, config_sha256: str, split_sha256: str | None, calibration_sha256: str = ""
) -> str:
    """Content-derived id: weights + config + split + calibration (thresholds change outputs)."""
    h = hashlib.sha256(
        f"{weights_sha256}|{config_sha256}|{split_sha256 or ''}|{calibration_sha256}".encode()
    )
    return h.hexdigest()[:16]
