"""Shared FastAPI dependencies."""

from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import Header, HTTPException
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .internal.graph import _LABELS, companion

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


def _is_route_label(text: str) -> bool:
    token = text.strip()
    if not token:
        return False
    if token in _LABELS:
        return True
    return " " not in token and "\n" not in token


def turn_replies(messages: list) -> str:
    """Assistant text after the latest human turn.

    A leading route label is omitted when a real reply follows. A lone label,
    such as END, stays the reply. Several replies are joined by a blank line.
    """
    after: list[str] = []
    for msg in messages:
        kind = getattr(msg, "type", "")
        if isinstance(msg, HumanMessage) or kind == "human":
            after = []
            continue
        if isinstance(msg, AIMessage) or kind == "ai":
            after.append(message_text(msg.content))
    if len(after) >= 2 and _is_route_label(after[0]):
        after = after[1:]
    return "\n\n".join(text for text in after if text)


def run_companion(messages: list, thread_id: str) -> str:
    try:
        result = companion.invoke(
            {
                "messages": to_lc_messages(messages),
                "retrieved": "",
                "issues": [],
                "route": "",
                "search_query": "",
                "searches": 0,
                "coverage": "",
            },
            {"configurable": {"thread_id": thread_id}},
        )
    except ValueError as exc:
        logger.exception("LLM configuration error")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Agent graph failed")
        raise HTTPException(status_code=502, detail=f"Agent failed: {exc}") from exc
    return turn_replies(result.get("messages") or [])
