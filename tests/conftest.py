from __future__ import annotations

import os
from pathlib import Path

import pytest

# Test-only secrets. Set before any settings object is created.
os.environ.setdefault("SR_PSEUDONYM_SECRET", "test-pseudonym-secret-not-for-production-0000")
os.environ.setdefault("SR_BLOB_ENCRYPTION_KEY", "0f" * 32)
os.environ.setdefault("SR_AUTH_SECRET", "test-auth-secret-not-for-production")
os.environ.setdefault("SR_DATABASE_URL", "sqlite+pysqlite:///:memory:")

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO


@pytest.fixture(scope="session")
def deid_profile():
    from secondreader.core.dicom import load_profile

    return load_profile(REPO / "configs" / "deid" / "basic_profile.yaml")


@pytest.fixture(scope="session")
def secret() -> bytes:
    return b"test-pseudonym-secret-not-for-production-0000"


@pytest.fixture
def deidentifier(deid_profile, secret):
    from secondreader.core.dicom import Deidentifier

    return Deidentifier(deid_profile, secret)
