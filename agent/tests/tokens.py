"""Mint HS256 access tokens that match the agent verifier."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import jwt

TEST_SECRET = "unit-test-jwt-secret-32-bytes-long"


def make_token(**overrides: object) -> str:
    now = datetime.now(timezone.utc)
    payload: dict[str, object] = {
        "sub": "user-1",
        "role": "admin",
        "permissions": ["chat:write"],
        "iat": now,
        "exp": now + timedelta(minutes=30),
    }
    payload.update(overrides)
    return jwt.encode(
        payload,
        os.environ["JWT_SECRET"],
        algorithm=os.environ.get("JWT_ALGORITHM") or "HS256",
    )
