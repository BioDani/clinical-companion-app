"""Verify HS256 JWTs issued by the rbac service. No identity-store hop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .config import jwt_algorithm, jwt_secret

REQUIRED_CLAIMS = ("sub", "role", "permissions", "exp", "iat")
_bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class Principal:
    user_id: str
    role: str
    permissions: frozenset[str]


def _unauthorized() -> HTTPException:
    return HTTPException(
        status_code=401,
        detail="unauthorized",
        headers={"WWW-Authenticate": "Bearer"},
    )


def require_principal(
    creds: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> Principal:
    """Require a valid rbac access token on agent routes."""
    if creds is None or creds.scheme.lower() != "bearer" or not creds.credentials:
        raise _unauthorized()
    try:
        payload: dict[str, Any] = jwt.decode(
            creds.credentials,
            jwt_secret(),
            algorithms=[jwt_algorithm()],
        )
    except jwt.PyJWTError as exc:
        raise _unauthorized() from exc
    if any(claim not in payload for claim in REQUIRED_CLAIMS):
        raise _unauthorized()
    perms = payload.get("permissions") or []
    if not isinstance(perms, list):
        perms = []
    return Principal(
        user_id=str(payload["sub"]),
        role=str(payload["role"]),
        permissions=frozenset(str(p) for p in perms),
    )
