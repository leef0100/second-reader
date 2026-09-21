"""Audit trail: who did what to which object, when. Append-only by convention."""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from secondreader.core.db.models import AuditEvent


def record(
    session: Session,
    action: str,
    *,
    user_id: str | None = None,
    username: str | None = None,
    object_type: str | None = None,
    object_id: str | None = None,
    details: dict[str, Any] | None = None,
    client_ip: str | None = None,
) -> AuditEvent:
    ev = AuditEvent(
        user_id=user_id,
        username=username,
        action=action,
        object_type=object_type,
        object_id=object_id,
        details=details or {},
        client_ip=client_ip,
    )
    session.add(ev)
    session.flush()
    return ev


class AuditLogger:
    """Bound convenience wrapper used by request handlers."""

    def __init__(self, session: Session, user_id: str | None, username: str | None, ip: str | None):
        self._s, self._uid, self._uname, self._ip = session, user_id, username, ip

    def __call__(
        self,
        action: str,
        object_type: str | None = None,
        object_id: str | None = None,
        **details: Any,
    ) -> AuditEvent:
        return record(
            self._s,
            action,
            user_id=self._uid,
            username=self._uname,
            object_type=object_type,
            object_id=object_id,
            details=details,
            client_ip=self._ip,
        )
