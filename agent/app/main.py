"""Clinical Companion FastAPI application."""

import logging
import os
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import jwt_secret, langfuse_enabled
from .internal.vector_store import ensure_indexed
from .routers import health, openai

logger = logging.getLogger(__name__)


def _index_ebook() -> None:
    try:
        ensure_indexed()
    except Exception:
        logger.exception("Ebook index was not built")


def _flush_langfuse() -> None:
    if not langfuse_enabled():
        return
    try:
        from langfuse import get_client

        get_client().flush()
    except Exception:
        logger.exception("Langfuse flush failed")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    jwt_secret()
    # Do not wait for ingest, so /health can pass while Weaviate is still opening.
    # Unit tests enter this lifespan and must not open Weaviate or Hugging Face.
    if not os.environ.get("PYTEST_CURRENT_TEST"):
        threading.Thread(target=_index_ebook, name="ebook-index", daemon=True).start()
    yield
    _flush_langfuse()


app = FastAPI(title="Clinical Companion agent", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(openai.router)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Clinical Companion"}
