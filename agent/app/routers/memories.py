"""Account memories for the signed-in home. The generic store API stays disabled."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Response

from app.internal.intake import public_status
from app.internal.langgraph_auth import AuthError, identity_from_authorization
from app.internal.user_memory import MemoryStore, delete_memory, list_memories

app = FastAPI()


def get_store() -> MemoryStore:
    from app.internal.persistence import open_memory_store

    return open_memory_store()


def _user(authorization: str | None) -> str:
    try:
        return identity_from_authorization(authorization)
    except AuthError as exc:
        raise HTTPException(status_code=401, detail="unauthorized") from exc


@app.get("/intake")
def read_intake(
    authorization: Annotated[str | None, Header()] = None,
    store: MemoryStore = Depends(get_store),
) -> dict[str, str | None]:
    """status is pending, finished, or skipped. next_field is null when not pending."""
    return public_status(store, _user(authorization))


@app.get("/memories")
def read_memories(
    authorization: Annotated[str | None, Header()] = None,
    store: MemoryStore = Depends(get_store),
) -> dict[str, list[dict[str, str | None]]]:
    return {"items": list_memories(store, _user(authorization))}


@app.delete("/memories/{item_id}", status_code=204)
def remove_memory(
    item_id: str,
    authorization: Annotated[str | None, Header()] = None,
    store: MemoryStore = Depends(get_store),
) -> Response:
    if not delete_memory(store, _user(authorization), item_id):
        raise HTTPException(status_code=404, detail="not found")
    return Response(status_code=204)
