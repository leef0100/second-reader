"""Runtime settings, loaded from environment (`SR_*`) or a local `.env`.

Secrets are never defaulted to usable values: missing secrets fail loudly at
startup rather than silently running with a known key.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SR_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    database_url: str = "sqlite+pysqlite:///./data/secondreader.sqlite"

    pseudonym_secret: SecretStr = Field(..., description="HMAC key for pseudonymous patient keys")
    blob_encryption_key: SecretStr = Field(..., description="64 hex chars = 32-byte AES-256 key")
    auth_secret: SecretStr = Field(..., description="JWT signing secret")

    blob_root: Path = Path("./data/blobs")
    model_registry_root: Path = Path("./models")
    active_model_id: str | None = None
    device: str = "auto"  # auto | cpu | mps | cuda

    orthanc_a_url: str = "http://localhost:8042"
    orthanc_a_user: str = "orthanc"
    orthanc_a_password: SecretStr = SecretStr("orthanc")
    orthanc_b_url: str = "http://localhost:8043"
    orthanc_b_user: str = "orthanc"
    orthanc_b_password: SecretStr = SecretStr("orthanc")

    deid_profile: Path = Path("configs/deid/basic_profile.yaml")
    access_token_ttl_minutes: int = 480
    webhook_token: SecretStr | None = None  # shared secret for the Orthanc → platform webhook

    @field_validator("blob_encryption_key")
    @classmethod
    def _key_is_32_bytes_hex(cls, v: SecretStr) -> SecretStr:
        raw = v.get_secret_value()
        try:
            key = bytes.fromhex(raw)
        except ValueError as exc:
            raise ValueError("SR_BLOB_ENCRYPTION_KEY must be hex") from exc
        if len(key) != 32:
            raise ValueError("SR_BLOB_ENCRYPTION_KEY must be 32 bytes (64 hex chars)")
        return v

    @field_validator("pseudonym_secret", "auth_secret")
    @classmethod
    def _secret_long_enough(cls, v: SecretStr) -> SecretStr:
        if len(v.get_secret_value()) < 16:
            raise ValueError("secret must be at least 16 characters")
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]  # required fields come from env
