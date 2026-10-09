"""OpenAI-compatible routes for Open WebUI and curl."""

from __future__ import annotations

import json
import time
import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..auth import Principal, require_principal
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


LOADING_STATUS = "Reviewing your question…"

_STREAM_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
}


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _status_event(*, done: bool, hidden: bool) -> str:
    return _sse(
        {
            "event": {
                "type": "status",
                "data": {
                    "description": LOADING_STATUS,
                    "done": done,
                    "hidden": hidden,
                },
            }
        }
    )


@router.get("/models")
def list_models(
    _principal: Annotated[Principal, Depends(require_principal)],
) -> dict[str, Any]:
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
    principal: Annotated[Principal, Depends(require_principal)],
):
    if not body.messages:
        raise HTTPException(status_code=400, detail="messages is required")

    # Open WebUI sends the full history each turn. Reusing body.user /
    # principal.user_id as the thread id appends that replay onto the
    # checkpoint. A fresh thread keeps MemorySaver from concatenating;
    # pass X-Session-Id to persist.
    thread_id = session_id or str(uuid.uuid4())
    created = int(time.time())
    completion_id = _completion_id()
    model = body.model or AGENT_MODEL_ID

    if body.stream:

        def events():
            yield _status_event(done=False, hidden=False)
            try:
                text = run_companion(body.messages, thread_id, principal.user_id)
            except HTTPException as exc:
                detail = exc.detail if isinstance(exc.detail, str) else "Agent failed"
                yield _sse({"error": {"message": detail, "code": exc.status_code}})
                yield "data: [DONE]\n\n"
                return
            yield _status_event(done=True, hidden=True)
            yield _sse(
                {
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
            )
            yield _sse(
                {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
            )
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers=_STREAM_HEADERS,
        )

    text = run_companion(body.messages, thread_id, principal.user_id)
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
