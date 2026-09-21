from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from secondreader.core.audit import record
from secondreader.core.constants import RESEARCH_USE_LABEL
from secondreader.core.db.models import Role, User
from secondreader.core.security import Principal, create_user
from secondreader.core.serving.deps import Container
from secondreader.core.serving.routes.common import container, db, principal

router = APIRouter(prefix="/v1/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"  # noqa: S105 — OAuth2 token type, not a secret
    username: str
    role: str
    research_use: str = RESEARCH_USE_LABEL


def bootstrap_admin(c: Container, s: Session) -> None:
    """Create 'admin' from SR_BOOTSTRAP_ADMIN_PASSWORD when no admin-role user exists yet."""
    pw = c.settings.bootstrap_admin_password
    if pw is None or s.scalars(select(User).where(User.role == Role.admin).limit(1)).first():
        return
    create_user(s, "admin", pw.get_secret_value(), "admin")
    record(s, "bootstrap_admin_created", username="system")


@router.post("/login", response_model=TokenResponse)
def login(
    body: LoginRequest,
    request: Request,
    c: Container = Depends(container),
    s: Session = Depends(db),
) -> TokenResponse:
    bootstrap_admin(c, s)
    p = c.auth.authenticate(s, body.username, body.password)
    ip = request.client.host if request.client else None
    if p is None:
        record(s, "login_failed", username=body.username, client_ip=ip)
        raise HTTPException(status_code=401, detail="invalid credentials")
    record(s, "login", user_id=p.user_id, username=p.username, client_ip=ip)
    return TokenResponse(access_token=c.auth.issue_token(p), username=p.username, role=p.role.value)


class Me(BaseModel):
    username: str
    role: str
    research_use: str = RESEARCH_USE_LABEL


@router.get("/me", response_model=Me)
def me(p: Principal = Depends(principal)) -> Me:
    return Me(username=p.username, role=p.role.value)
