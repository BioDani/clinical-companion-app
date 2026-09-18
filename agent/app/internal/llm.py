"""LLM calls via smolagents InferenceClientModel (Hugging Face Inference)."""

from __future__ import annotations

from functools import lru_cache

from langchain_core.messages import BaseMessage
from smolagents import ChatMessage, InferenceClientModel

from ..config import hf_model, hf_token

_ROLE = {
    "human": "user",
    "ai": "assistant",
    "system": "system",
    "tool": "tool",
}


@lru_cache(maxsize=1)
def get_model() -> InferenceClientModel:
    return InferenceClientModel(
        model_id=hf_model(),
        token=hf_token(),
        max_tokens=512,
        temperature=0,
    )


def _content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text") or ""))
            elif hasattr(part, "text"):
                parts.append(str(getattr(part, "text") or ""))
        return "".join(parts)
    return "" if content is None else str(content)


def _chat_content(text: str) -> list[dict[str, str]]:
    # smolagents concatenates consecutive same-role messages and requires
    # list content for that merge (Open WebUI often sends two user turns).
    return [{"type": "text", "text": text}]


def complete(messages: list[BaseMessage]) -> str:
    payload = [
        ChatMessage(
            role=_ROLE.get(msg.type, "user"),
            content=_chat_content(_content_text(msg.content)),
        )
        for msg in messages
    ]
    result = get_model().generate(payload)
    return _content_text(getattr(result, "content", result)).strip()
