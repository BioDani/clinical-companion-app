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


def _flush_langfuse() -> None:
    if not langfuse_enabled():
        return
    try:
        from langfuse import get_client

        get_client().flush()
    except Exception:
        logger.exception("Langfuse flush failed")


def _index_in_background() -> None:
    try:
        ensure_indexed()
    except Exception:
        logger.exception("Knowledge index was not built")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    jwt_secret()
    # Bind the port before ingest. A blocked startup makes OrbStack dial the
    # container and report 502 connection refused. /health stays 503 until the
    # index matches the PDFs. Tests must not open Weaviate or Hugging Face.
    if not os.environ.get("PYTEST_CURRENT_TEST"):
        threading.Thread(
            target=_index_in_background,
            name="knowledge-index",
            daemon=True,
        ).start()
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
