"""Shared FastAPI dependencies."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import Header, HTTPException
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .internal.graph import companion


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


def run_companion(messages: list, thread_id: str) -> str:
    try:
        result = companion.invoke(
            {"messages": to_lc_messages(messages), "retrieved": ""},
            {"configurable": {"thread_id": thread_id}},
        )
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Agent failed: {exc}") from exc
    return last_assistant(result.get("messages") or [])
