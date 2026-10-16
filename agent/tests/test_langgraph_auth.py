"""Thread ownership for the LangGraph API."""

from app.internal.langgraph_auth import (
    AuthError,
    identity_from_authorization,
    load_status,
    stamp_create,
    thread_metadata,
)
from tests.tokens import make_token


def test_bad_token_is_unauthorized():
    try:
        identity_from_authorization(None)
    except AuthError:
        pass
    else:
        raise AssertionError("missing token")
    try:
        identity_from_authorization("Bearer not-a-jwt")
    except AuthError:
        pass
    else:
        raise AssertionError("bad token")


def test_create_stamps_owner_and_channel():
    identity = identity_from_authorization(f"Bearer {make_token(sub='user-9')}")
    assert identity == "user-9"
    value: dict = {}
    filt = stamp_create(identity, value)
    assert filt == thread_metadata("user-9")
    assert value["metadata"]["owner"] == "user-9"
    assert value["metadata"]["channel"] == "assistant-ui"
    assert load_status("user-9", value["metadata"]) == 200


def test_thread_id_is_not_the_account_id():
    try:
        stamp_create("user-9", {"thread_id": "user-9"})
    except AuthError:
        pass
    else:
        raise AssertionError("thread id")


def test_other_owner_or_channel_is_404():
    own = {"owner": "user-9", "channel": "assistant-ui"}
    assert load_status("user-2", own) == 404
    assert load_status("user-9", {"owner": "user-9", "channel": "open-webui"}) == 404
    assert load_status("user-9", None) == 404
