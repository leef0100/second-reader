"""Shared dependencies: container, DB session, auth principal, audit."""

from __future__ import annotations

from collections.abc import Iterator

from fastapi import Depends, Header, HTTPException, Request
from sqlalchemy.orm import Session

from secondreader.core.audit import AuditLogger
from secondreader.core.security import Principal
from secondreader.core.serving.deps import Container, get_container


def container() -> Container:
    return get_container()


def db(c: Container = Depends(container)) -> Iterator[Session]:
    s = c.sessions()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def principal(
    authorization: str | None = Header(default=None), c: Container = Depends(container)
) -> Principal:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    p = c.auth.verify_token(authorization.split(" ", 1)[1].strip())
    if p is None:
        raise HTTPException(status_code=401, detail="invalid or expired token")
    return p


def audit(
    request: Request, s: Session = Depends(db), p: Principal = Depends(principal)
) -> AuditLogger:
    ip = request.client.host if request.client else None
    return AuditLogger(s, p.user_id, p.username, ip)
