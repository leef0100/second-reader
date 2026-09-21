"""Local username/password auth with roles (D-016). `AuthProvider` is the seam for OIDC."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from sqlalchemy import select
from sqlalchemy.orm import Session

from secondreader.core.db.models import Role, User

_ph = PasswordHasher()


@dataclass(frozen=True)
class Principal:
    user_id: str
    username: str
    role: Role


class AuthProvider(Protocol):
    def authenticate(self, session: Session, username: str, password: str) -> Principal | None: ...
    def issue_token(self, principal: Principal) -> str: ...
    def verify_token(self, token: str) -> Principal | None: ...


def create_user(session: Session, username: str, password: str, role: str = "clinician") -> User:
    if len(password) < 10:
        raise ValueError("password must be at least 10 characters")
    u = User(username=username, password_hash=_ph.hash(password), role=Role(role))
    session.add(u)
    session.flush()
    return u


class LocalAuthProvider:
    def __init__(self, secret: str, ttl_minutes: int = 480) -> None:
        self._secret = secret
        self._ttl = timedelta(minutes=ttl_minutes)

    def authenticate(self, session: Session, username: str, password: str) -> Principal | None:
        u = session.scalars(
            select(User).where(User.username == username, User.active.is_(True))
        ).first()
        if u is None:
            _ph.hash(password)  # constant-ish time
            return None
        try:
            _ph.verify(u.password_hash, password)
        except VerifyMismatchError:
            return None
        return Principal(user_id=u.id, username=u.username, role=u.role)

    def issue_token(self, p: Principal) -> str:
        now = datetime.now(UTC)
        payload = {
            "sub": p.user_id,
            "username": p.username,
            "role": p.role.value,
            "iat": now,
            "exp": now + self._ttl,
        }
        return jwt.encode(payload, self._secret, algorithm="HS256")

    def verify_token(self, token: str) -> Principal | None:
        try:
            data = jwt.decode(token, self._secret, algorithms=["HS256"])
        except jwt.PyJWTError:
            return None
        return Principal(
            user_id=str(data["sub"]), username=str(data["username"]), role=Role(data["role"])
        )
