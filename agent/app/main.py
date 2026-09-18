"""Clinical Companion FastAPI application."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import jwt_secret
from .routers import health, openai


@asynccontextmanager
async def lifespan(_app: FastAPI):
    jwt_secret()
    yield


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
