"""Shared FastAPI dependencies."""

from __future__ import annotations

import logging
import re
from typing import Annotated, Any

from fastapi import Header, HTTPException
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .config import langfuse_enabled
from .internal.graph import companion

logger = logging.getLogger(__name__)


def get_session_id(x_session_id: Annotated[str | None, Header()] = None) -> str | None:
    """Optional conversation id from Open WebUI / curl (`X-Session-Id`)."""
    return x_session_id


def message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text") or ""))
        return "".join(parts)
    return "" if content is None else str(content)


def to_lc_messages(messages: list):
    out = []
    for msg in messages:
        text = message_text(msg.content)
        if msg.role == "system":
            out.append(SystemMessage(content=text))
        elif msg.role == "assistant":
            out.append(AIMessage(content=text))
        else:
            out.append(HumanMessage(content=text))
    return out


def last_assistant(messages: list) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) or getattr(msg, "type", "") == "ai":
            return message_text(msg.content)
    return ""


RETRY_REPLY = "I couldn't complete that just now. Please try again."

_HIDDEN_LABELS = {
    "SMALLTALK",
    "RAG",
    "TOOLS",
    "DANGEROUS",
    "END",
    "SUFFICIENT",
    "INSUFFICIENT",
    "MISS",
    "BMI",
    "KNOWLEDGE",
    "SAFE",
}

_ERROR_LEAK = re.compile(r"^\s*_?[A-Za-z]+,\s*error\b", re.IGNORECASE)


def _is_hidden_reply(text: str) -> bool:
    """Route labels and model error dumps are not answers."""
    token = text.strip()
    if token in _HIDDEN_LABELS:
        return True
    return _ERROR_LEAK.match(token) is not None


def turn_replies(messages: list) -> str:
    """Assistant text after the latest human turn.

    Route labels and replies that start like ``_Greeting, error`` are omitted.
    When that leaves nothing to show, a short retry sentence is returned.
    Several real replies are joined by a blank line.
    """
    visible: list[str] = []
    hid_internal = False
    for msg in messages:
        kind = getattr(msg, "type", "")
        if isinstance(msg, HumanMessage) or kind == "human":
            visible = []
            hid_internal = False
            continue
        if isinstance(msg, AIMessage) or kind == "ai":
            text = message_text(msg.content).strip()
            if not text:
                continue
            if _is_hidden_reply(text):
                hid_internal = True
                continue
            visible.append(text)
    if visible:
        return "\n\n".join(visible)
    if hid_internal:
        return RETRY_REPLY
    return ""


def invoke_config(thread_id: str, user_id: str | None = None) -> dict[str, Any]:
    """Graph config for one turn. Langfuse metadata is added only when tracing is on."""
    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    if not langfuse_enabled():
        return config
    from langfuse.langchain import CallbackHandler

    metadata: dict[str, Any] = {
        "langfuse_session_id": thread_id,
        "langfuse_tags": ["clinical-companion"],
    }
    if user_id:
        metadata["langfuse_user_id"] = user_id
    config["callbacks"] = [CallbackHandler()]
    config["metadata"] = metadata
    return config


def run_companion(messages: list, thread_id: str, user_id: str | None = None) -> str:
    try:
        result = companion.invoke(
            {"messages": to_lc_messages(messages)},
            invoke_config(thread_id, user_id),
        )
    except ValueError as exc:
        logger.exception("LLM configuration error")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Agent graph failed")
        raise HTTPException(status_code=502, detail=f"Agent failed: {exc}") from exc
    return turn_replies(result.get("messages") or [])
