"""Keep introduction answers and personal notes out of Langfuse traces."""

from __future__ import annotations

from typing import Any

from .intake import CLOSE, INTAKE_START, NOTICE

NOTES_PREFIX = "Personal notes this person asked to remember."
REDACTED = "[redacted]"
_KEEP = {
    "langfuse_session_id",
    "langfuse_user_id",
    "session_id",
    "user_id",
    "langfuse_tags",
}
_sensitive: set[str] = set()


def note_sensitive(text: str) -> None:
    cleaned = text.strip()
    if cleaned:
        _sensitive.add(cleaned)


def _private(text: str) -> bool:
    if text in _sensitive or text == INTAKE_START:
        return True
    if text.startswith(NOTES_PREFIX) or text.startswith(NOTICE) or text == CLOSE:
        return True
    return any(secret and secret in text for secret in _sensitive if len(secret) > 2)


def redact(data: Any) -> Any:
    """Replace intake and personal-note text. Leave the session id, sub, and tag."""
    if isinstance(data, str):
        return REDACTED if _private(data) else data
    if isinstance(data, dict):
        return {key: value if key in _KEEP else redact(value) for key, value in data.items()}
    if isinstance(data, list):
        return [redact(item) for item in data]
    return data


def install_langfuse_mask() -> None:
    """Attach the mask before the Langfuse callback reads a run."""
    from langfuse import Langfuse

    Langfuse(mask=redact)
