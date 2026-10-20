"""JWT gate for the LangGraph API. No call to the identity service."""

from __future__ import annotations

from typing import Any

import jwt

try:
    from app.auth import REQUIRED_CLAIMS
    from app.config import jwt_algorithm, jwt_secret
except ImportError:  # loaded as part of the app package
    from ..auth import REQUIRED_CLAIMS
    from ..config import jwt_algorithm, jwt_secret

CHANNEL = "assistant-ui"


class AuthError(Exception):
    """Missing or invalid bearer token."""


def identity_from_authorization(authorization: str | None) -> str:
    """Return JWT sub or raise AuthError. The thread id is never this value."""
    if not authorization:
        raise AuthError("unauthorized")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise AuthError("unauthorized")
    try:
        payload: dict[str, Any] = jwt.decode(
            token.strip(),
            jwt_secret(),
            algorithms=[jwt_algorithm()],
        )
    except jwt.PyJWTError as exc:
        raise AuthError("unauthorized") from exc
    if any(claim not in payload for claim in REQUIRED_CLAIMS):
        raise AuthError("unauthorized")
    return str(payload["sub"])


def thread_metadata(identity: str) -> dict[str, str]:
    return {"owner": identity, "channel": CHANNEL}


def stamp_create(identity: str, value: dict[str, Any]) -> dict[str, str]:
    """Own the new thread. Reject a thread id that is the account id."""
    thread_id = value.get("thread_id")
    if thread_id is not None and str(thread_id) == identity:
        raise AuthError("thread id is the account id")
    metadata = value.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
        value["metadata"] = metadata
    metadata["owner"] = identity
    metadata["channel"] = CHANNEL
    return thread_metadata(identity)


def load_status(identity: str, metadata: dict[str, Any] | None) -> int:
    """Search and load miss with 404 for another owner or another channel."""
    return 200 if visible(identity, metadata) else 404


def visible(identity: str, metadata: dict[str, Any] | None) -> bool:
    """A list or a load returns a chat only for this owner and channel."""
    data = metadata or {}
    return data.get("owner") == identity and data.get("channel") == CHANNEL


def _install(auth: Any) -> None:
    @auth.authenticate
    async def authenticate(authorization: str | None = None) -> dict[str, Any]:
        try:
            identity = identity_from_authorization(authorization)
        except AuthError as exc:
            raise PermissionError("unauthorized") from exc
        return {"identity": identity, "is_authenticated": True}

    @auth.on.threads.create
    async def on_create(ctx: Any, value: dict[str, Any]) -> dict[str, str]:
        try:
            return stamp_create(ctx.user.identity, value)
        except AuthError as exc:
            raise PermissionError(str(exc)) from exc

    @auth.on.threads.search
    async def on_search(ctx: Any, value: dict[str, Any]) -> dict[str, str]:
        del value
        return thread_metadata(ctx.user.identity)

    @auth.on.threads.read
    async def on_read(ctx: Any, value: dict[str, Any]) -> dict[str, str]:
        del value
        return thread_metadata(ctx.user.identity)

    @auth.on.threads.update
    async def on_update(ctx: Any, value: dict[str, Any]) -> dict[str, str]:
        del value
        return thread_metadata(ctx.user.identity)

    @auth.on.threads.delete
    async def on_delete(ctx: Any, value: dict[str, Any]) -> dict[str, str]:
        del value
        return thread_metadata(ctx.user.identity)


try:
    from langgraph_sdk import Auth
except ImportError:  # pragma: no cover - image always has the SDK
    auth = None
else:
    auth = Auth()
    _install(auth)
