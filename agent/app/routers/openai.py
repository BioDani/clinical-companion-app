"""OpenAI-compatible routes for Open WebUI and curl."""

from __future__ import annotations

import json
import time
import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..config import AGENT_MODEL_ID
from ..dependencies import get_session_id, run_companion

router = APIRouter(prefix="/v1", tags=["openai"])


class ChatMessageIn(BaseModel):
    role: str
    content: Any = ""


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessageIn]
    stream: bool = False
    user: str | None = None


def _completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


@router.get("/models")
def list_models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": AGENT_MODEL_ID,
                "object": "model",
                "created": 0,
                "owned_by": "clinical-companion",
            }
        ],
    }


@router.post("/chat/completions")
def chat_completions(
    body: ChatCompletionRequest,
    session_id: Annotated[str | None, Depends(get_session_id)],
):
    if not body.messages:
        raise HTTPException(status_code=400, detail="messages is required")

    # Open WebUI sends the full history each turn; a fresh thread avoids
    # duplicating messages in MemorySaver. Pass X-Session-Id to persist.
    thread_id = session_id or body.user or str(uuid.uuid4())
    text = run_companion(body.messages, thread_id)
    created = int(time.time())
    completion_id = _completion_id()
    model = body.model or AGENT_MODEL_ID

    if body.stream:
        first = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": text},
                    "finish_reason": None,
                }
            ],
        }
        last = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }

        def events():
            yield f"data: {json.dumps(first)}\n\n"
            yield f"data: {json.dumps(last)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
