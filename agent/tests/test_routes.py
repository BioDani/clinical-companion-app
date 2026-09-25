"""HTTP behavior of public and JWT-protected routes. The graph is mocked."""

from __future__ import annotations

import asyncio
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import AGENT_MODEL_ID
from app.main import app, lifespan


def test_root_and_health_are_public(client: TestClient):
    root = client.get("/")
    health = client.get("/health")
    assert root.status_code == 200
    assert root.json() == {"message": "Clinical Companion"}
    assert health.status_code == 200
    assert health.json() == {"status": "ok"}


def test_models_requires_bearer_token(client: TestClient):
    response = client.get("/v1/models")
    assert response.status_code == 401
    assert response.json()["detail"] == "unauthorized"
    assert response.headers["www-authenticate"].lower().startswith("bearer")


def test_models_lists_companion(client: TestClient, auth_header: dict[str, str]):
    response = client.get("/v1/models", headers=auth_header)
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert body["data"][0]["id"] == AGENT_MODEL_ID
    assert body["data"][0]["owned_by"] == "clinical-companion"


def test_chat_rejects_empty_messages(client: TestClient, auth_header: dict[str, str]):
    response = client.post(
        "/v1/chat/completions",
        headers=auth_header,
        json={"messages": []},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "messages is required"


def test_chat_rejects_missing_messages(client: TestClient, auth_header: dict[str, str]):
    response = client.post("/v1/chat/completions", headers=auth_header, json={})
    assert response.status_code == 422


def test_chat_completion_uses_session_and_reply(
    client: TestClient,
    auth_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
):
    seen: dict = {}

    def run(messages, thread_id: str) -> str:
        seen["roles"] = [message.role for message in messages]
        seen["thread_id"] = thread_id
        return "oats and fruit"

    monkeypatch.setattr("app.routers.openai.run_companion", run)
    response = client.post(
        "/v1/chat/completions",
        headers={**auth_header, "X-Session-Id": "sess-9"},
        json={
            "model": "custom-model",
            "user": "should-not-be-the-thread",
            "messages": [{"role": "user", "content": "breakfast?"}],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert seen["thread_id"] == "sess-9"
    assert seen["roles"] == ["user"]
    assert body["object"] == "chat.completion"
    assert body["model"] == "custom-model"
    assert body["id"].startswith("chatcmpl-")
    assert body["choices"][0]["message"] == {
        "role": "assistant",
        "content": "oats and fruit",
    }
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"] == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }


def test_chat_without_session_mints_a_thread(
    client: TestClient,
    auth_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
):
    seen: dict = {}

    def run(messages, thread_id: str) -> str:
        seen["thread_id"] = thread_id
        return "ok"

    monkeypatch.setattr("app.routers.openai.run_companion", run)
    response = client.post(
        "/v1/chat/completions",
        headers=auth_header,
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    assert response.json()["model"] == AGENT_MODEL_ID
    assert seen["thread_id"]
    assert seen["thread_id"] != "user-1"


def test_chat_stream_emits_sse(
    client: TestClient,
    auth_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr("app.routers.openai.run_companion", lambda messages, thread_id: "streamed")
    response = client.post(
        "/v1/chat/completions",
        headers=auth_header,
        json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    data_lines = [line for line in response.text.splitlines() if line.startswith("data: ")]
    assert data_lines[-1] == "data: [DONE]"
    first = json.loads(data_lines[0].removeprefix("data: "))
    last = json.loads(data_lines[1].removeprefix("data: "))
    assert first["object"] == "chat.completion.chunk"
    assert first["choices"][0]["delta"]["content"] == "streamed"
    assert last["choices"][0]["finish_reason"] == "stop"


def test_chat_requires_a_valid_token(client: TestClient):
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 401


def test_lifespan_accepts_configured_secret():
    async def _run() -> bool:
        async with lifespan(app):
            return True

    assert asyncio.run(_run()) is True


def test_lifespan_rejects_missing_secret(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("JWT_SECRET", raising=False)

    async def _run() -> None:
        async with lifespan(FastAPI()):
            pass

    with pytest.raises(ValueError, match="JWT_SECRET"):
        asyncio.run(_run())
