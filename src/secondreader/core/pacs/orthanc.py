"""Minimal Orthanc REST client (used for both the hospital PACS A and archive B)."""

from __future__ import annotations

from typing import Any

import httpx


class OrthancClient:
    def __init__(
        self,
        base_url: str,
        user: str | None = None,
        password: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        auth = (user, password) if user and password else None
        self._c = httpx.Client(base_url=base_url.rstrip("/"), auth=auth, timeout=timeout)

    def close(self) -> None:
        self._c.close()

    def reachable(self) -> bool:
        try:
            return self._c.get("/system").status_code == 200
        except httpx.HTTPError:
            return False

    def system(self) -> dict[str, Any]:
        r = self._c.get("/system")
        r.raise_for_status()
        return dict(r.json())

    def study_instances(self, study_id: str) -> list[str]:
        r = self._c.get(f"/studies/{study_id}/instances")
        r.raise_for_status()
        return [str(i["ID"]) for i in r.json()]

    def instance_file(self, instance_id: str) -> bytes:
        r = self._c.get(f"/instances/{instance_id}/file")
        r.raise_for_status()
        return r.content

    def upload(self, dicom: bytes) -> dict[str, Any]:
        r = self._c.post("/instances", content=dicom, headers={"Content-Type": "application/dicom"})
        r.raise_for_status()
        return dict(r.json())

    def find_study_by_uid(self, study_uid: str) -> str | None:
        r = self._c.post(
            "/tools/find", json={"Level": "Study", "Query": {"StudyInstanceUID": study_uid}}
        )
        r.raise_for_status()
        ids = r.json()
        return str(ids[0]) if ids else None

    def study(self, study_id: str) -> dict[str, Any]:
        r = self._c.get(f"/studies/{study_id}")
        r.raise_for_status()
        return dict(r.json())

    def delete_study(self, study_id: str) -> None:
        self._c.delete(f"/studies/{study_id}").raise_for_status()

    def list_studies(self) -> list[str]:
        r = self._c.get("/studies")
        r.raise_for_status()
        return [str(s) for s in r.json()]

    def instance_preview_png(self, instance_id: str, frame: int = 0) -> bytes:
        r = self._c.get(f"/instances/{instance_id}/frames/{frame}/preview")
        r.raise_for_status()
        return r.content
