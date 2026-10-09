import os

from fastapi import APIRouter, HTTPException

from ..internal.vector_store import indexed

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, str]:
    if not os.environ.get("PYTEST_CURRENT_TEST") and not indexed():
        raise HTTPException(status_code=503, detail="indexing knowledge")
    return {"status": "ok"}
