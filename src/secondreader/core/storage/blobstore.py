"""Blob storage with application-level encryption at rest.

`BlobStore` is the seam; `FilesystemBlobStore` is the local implementation and
`EncryptedBlobStore` wraps any store with AES-256-GCM. Keys are sharded into
two-level directories to keep directories small.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

_KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,254}$")


@dataclass(frozen=True)
class BlobRef:
    key: str
    size: int
    sha256: str  # of the *plaintext*


class BlobStore(Protocol):
    def put(self, key: str, data: bytes) -> BlobRef: ...
    def get(self, key: str) -> bytes: ...
    def exists(self, key: str) -> bool: ...
    def delete(self, key: str) -> None: ...


class BlobKeyError(ValueError):
    pass


def validate_key(key: str) -> str:
    if not _KEY_RE.match(key) or ".." in key:
        raise BlobKeyError(f"invalid blob key: {key!r}")
    return key


class FilesystemBlobStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        validate_key(key)
        h = hashlib.sha256(key.encode()).hexdigest()
        return self.root / h[:2] / h[2:4] / key.replace("/", "__")

    def put(self, key: str, data: bytes) -> BlobRef:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, p)  # atomic on POSIX
        return BlobRef(key=key, size=len(data), sha256=hashlib.sha256(data).hexdigest())

    def get(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def delete(self, key: str) -> None:
        p = self._path(key)
        if p.exists():
            p.unlink()


_MAGIC = b"SRB1"  # format/version marker
_NONCE_LEN = 12


class EncryptedBlobStore:
    """AES-256-GCM envelope around another store. The key name is bound as AAD so
    a ciphertext copied to a different key fails to decrypt."""

    def __init__(self, inner: BlobStore, key: bytes) -> None:
        if len(key) != 32:
            raise ValueError("encryption key must be 32 bytes")
        self._inner = inner
        self._aead = AESGCM(key)

    def put(self, key: str, data: bytes) -> BlobRef:
        validate_key(key)
        nonce = os.urandom(_NONCE_LEN)
        ct = self._aead.encrypt(nonce, data, key.encode())
        self._inner.put(key, _MAGIC + nonce + ct)
        return BlobRef(key=key, size=len(data), sha256=hashlib.sha256(data).hexdigest())

    def get(self, key: str) -> bytes:
        raw = self._inner.get(key)
        if raw[: len(_MAGIC)] != _MAGIC:
            raise ValueError("blob is not in encrypted format")
        nonce = raw[len(_MAGIC) : len(_MAGIC) + _NONCE_LEN]
        ct = raw[len(_MAGIC) + _NONCE_LEN :]
        try:
            return self._aead.decrypt(nonce, ct, key.encode())
        except InvalidTag as exc:
            raise ValueError("blob authentication failed (wrong key or tampered)") from exc

    def exists(self, key: str) -> bool:
        return self._inner.exists(key)

    def delete(self, key: str) -> None:
        self._inner.delete(key)


def open_default_store(root: Path, key_hex: str) -> BlobStore:
    return EncryptedBlobStore(FilesystemBlobStore(root), bytes.fromhex(key_hex))
