"""Bearer JWT checks for rbac access tokens."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import jwt
import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from app.auth import REQUIRED_CLAIMS, require_principal
from tests.tokens import TEST_SECRET, make_token


def _creds(token: str, scheme: str = "Bearer") -> HTTPAuthorizationCredentials:
    return HTTPAuthorizationCredentials(scheme=scheme, credentials=token)


def test_required_claims_match_rbac_access_token():
    assert REQUIRED_CLAIMS == ("sub", "role", "permissions", "exp", "iat")


def test_valid_token_becomes_principal():
    principal = require_principal(_creds(make_token()))
    assert principal.user_id == "user-1"
    assert principal.role == "admin"
    assert principal.permissions == frozenset({"chat:write"})


def test_permissions_are_stringified():
    token = make_token(permissions=["read", 7])
    principal = require_principal(_creds(token))
    assert principal.permissions == frozenset({"read", "7"})


def test_non_list_permissions_become_empty():
    token = make_token(permissions="chat:write")
    principal = require_principal(_creds(token))
    assert principal.permissions == frozenset()


@pytest.mark.parametrize(
    "creds",
    [
        None,
        HTTPAuthorizationCredentials(scheme="Basic", credentials="abc"),
        HTTPAuthorizationCredentials(scheme="Bearer", credentials=""),
    ],
)
def test_missing_or_non_bearer_credentials_are_unauthorized(creds):
    with pytest.raises(HTTPException) as exc:
        require_principal(creds)
    assert exc.value.status_code == 401
    assert exc.value.detail == "unauthorized"
    assert exc.value.headers == {"WWW-Authenticate": "Bearer"}


def test_bad_signature_is_unauthorized():
    token = jwt.encode(
        {
            "sub": "user-1",
            "role": "admin",
            "permissions": [],
            "iat": datetime.now(timezone.utc),
            "exp": datetime.now(timezone.utc) + timedelta(minutes=5),
        },
        "other-secret-that-is-long-enough!!",
        algorithm="HS256",
    )
    with pytest.raises(HTTPException) as exc:
        require_principal(_creds(token))
    assert exc.value.status_code == 401


def test_expired_token_is_unauthorized():
    token = make_token(exp=datetime.now(timezone.utc) - timedelta(minutes=5))
    with pytest.raises(HTTPException) as exc:
        require_principal(_creds(token))
    assert exc.value.status_code == 401


@pytest.mark.parametrize("claim", REQUIRED_CLAIMS)
def test_missing_claim_is_unauthorized(claim: str):
    payload = {
        "sub": "user-1",
        "role": "admin",
        "permissions": [],
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(minutes=5),
    }
    del payload[claim]
    token = jwt.encode(payload, TEST_SECRET, algorithm="HS256")
    with pytest.raises(HTTPException) as exc:
        require_principal(_creds(token))
    assert exc.value.status_code == 401


def test_garbage_token_is_unauthorized():
    with pytest.raises(HTTPException) as exc:
        require_principal(_creds("not-a-jwt"))
    assert exc.value.status_code == 401
