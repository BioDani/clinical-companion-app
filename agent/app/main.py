"""Clinical Companion FastAPI application."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import health, openai

app = FastAPI(title="Clinical Companion agent", version="0.1.0")
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
