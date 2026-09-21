from __future__ import annotations

import os

import pytest

from secondreader.core.storage import EncryptedBlobStore, FilesystemBlobStore
from secondreader.core.storage.blobstore import BlobKeyError


@pytest.fixture
def fs(tmp_path):
    return FilesystemBlobStore(tmp_path / "blobs")


def test_filesystem_roundtrip(fs):
    ref = fs.put("dicom/study/inst.dcm", b"hello")
    assert fs.get("dicom/study/inst.dcm") == b"hello"
    assert ref.size == 5 and len(ref.sha256) == 64
    assert fs.exists("dicom/study/inst.dcm")
    fs.delete("dicom/study/inst.dcm")
    assert not fs.exists("dicom/study/inst.dcm")


@pytest.mark.parametrize("bad", ["../etc/passwd", "/abs", "", "a/../b", "x" * 300])
def test_rejects_bad_keys(fs, bad):
    with pytest.raises(BlobKeyError):
        fs.put(bad, b"x")


def test_encrypted_store_ciphertext_differs_and_decrypts(tmp_path):
    key = os.urandom(32)
    inner = FilesystemBlobStore(tmp_path)
    enc = EncryptedBlobStore(inner, key)
    data = b"DOE^JANE PHI-LOOKING PLAINTEXT " * 50
    enc.put("k/1", data)
    raw = inner.get("k/1")
    assert b"DOE^JANE" not in raw
    assert raw[:4] == b"SRB1"
    assert enc.get("k/1") == data


def test_encrypted_store_wrong_key_fails(tmp_path):
    inner = FilesystemBlobStore(tmp_path)
    EncryptedBlobStore(inner, b"a" * 32).put("k", b"secret")
    with pytest.raises(ValueError):
        EncryptedBlobStore(inner, b"b" * 32).get("k")


def test_encrypted_store_binds_key_name(tmp_path):
    inner = FilesystemBlobStore(tmp_path)
    enc = EncryptedBlobStore(inner, b"a" * 32)
    enc.put("k1", b"secret")
    inner.put("k2", inner.get("k1"))  # copy ciphertext under a different key
    with pytest.raises(ValueError):
        enc.get("k2")


def test_encrypted_store_detects_tamper(tmp_path):
    inner = FilesystemBlobStore(tmp_path)
    enc = EncryptedBlobStore(inner, b"a" * 32)
    enc.put("k", b"secret")
    raw = bytearray(inner.get("k"))
    raw[-1] ^= 0xFF
    inner.put("k", bytes(raw))
    with pytest.raises(ValueError):
        enc.get("k")
