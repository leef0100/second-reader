"""Wiring: settings → engine, blob store, de-identifier, ingest service, PACS clients."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from secondreader.core.db import make_engine, make_session_factory
from secondreader.core.dicom import Deidentifier, load_profile
from secondreader.core.ingest import IngestService
from secondreader.core.pacs import OrthancClient
from secondreader.core.plugins import PluginRegistry, load_plugins
from secondreader.core.security import LocalAuthProvider
from secondreader.core.settings import Settings, get_settings
from secondreader.core.storage import BlobStore, open_default_store


@dataclass
class Container:
    settings: Settings
    engine: Engine
    sessions: sessionmaker[Session]
    blobs: BlobStore
    deid: Deidentifier
    ingest: IngestService
    orthanc_a: OrthancClient
    orthanc_b: OrthancClient
    auth: LocalAuthProvider
    _plugins: PluginRegistry | None = None

    def plugins(self) -> PluginRegistry:
        """Plugins are loaded lazily (model weights) and cached for the process."""
        if self._plugins is None:
            self._plugins = load_plugins(
                self.settings.model_registry_root,
                self.settings.active_model_id,
                self.settings.device,
            )
        return self._plugins

    def set_plugins(self, reg: PluginRegistry) -> None:
        self._plugins = reg


def build_container(settings: Settings | None = None) -> Container:
    s = settings or get_settings()
    engine = make_engine(s.database_url)
    blobs = open_default_store(s.blob_root, s.blob_encryption_key.get_secret_value())
    deid = Deidentifier(
        load_profile(s.deid_profile), s.pseudonym_secret.get_secret_value().encode()
    )
    return Container(
        settings=s,
        engine=engine,
        sessions=make_session_factory(engine),
        blobs=blobs,
        deid=deid,
        ingest=IngestService(deid, blobs),
        orthanc_a=OrthancClient(
            s.orthanc_a_url, s.orthanc_a_user, s.orthanc_a_password.get_secret_value()
        ),
        orthanc_b=OrthancClient(
            s.orthanc_b_url, s.orthanc_b_user, s.orthanc_b_password.get_secret_value()
        ),
        auth=LocalAuthProvider(s.auth_secret.get_secret_value(), s.access_token_ttl_minutes),
    )


@lru_cache(maxsize=1)
def get_container() -> Container:
    return build_container()
