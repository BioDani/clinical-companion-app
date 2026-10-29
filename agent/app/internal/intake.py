"""Optional first-chat questions for one signed-in account.

Progress lives beside the memories list. Deleting a memory does not restart it.
"""

from __future__ import annotations

import re
from typing import Any

from .bmi import BmiTool
from .user_memory import MemoryStore, store_field

INTAKE_START = "INTAKE_START"
NOTICE = (
    "Hi, I’m wellnest, your wellness companion 👋\n"
    "To give you more relevant answers, you can choose to share a bit about yourself, "
    "like health conditions or daily habits. This is completely optional, and anything "
    "you share will be used only to tailor my answers to you.\n"
    "Your information is kept safe and handled according to our Privacy Policy. "
    "It will never be used to train AI models."
)
CLOSE = "Thank you. I will use what you chose to share on later habit questions."

FIELDS: tuple[tuple[str, str, str], ...] = (
    ("age", "Age", "What is your age?"),
    ("weight", "Weight", "What is your weight?"),
    ("height", "Height", "What is your height?"),
    ("sleeping_hours", "Sleeping hours", "How many hours do you sleep at night?"),
    ("meals_per_day", "Meals per day", "How many meals do you eat per day?"),
    ("diet_type", "Diet type", "What kind of diet do you follow?"),
    (
        "condition",
        "Condition",
        "Is there a condition that should change suggestions, "
        "such as knee problems, elevated blood pressure, or obesity?",
    ),
)

_ORDER = [field_id for field_id, _label, _question in FIELDS]
_BY_ID = {field_id: (label, question) for field_id, label, question in FIELDS}
_SKIP_ONE = {"skip", "skip this", "rather not say", "prefer not to say"}
_SKIP_REST = {"skip the rest", "skip all"}
_GREETINGS = {
    "hi",
    "hello",
    "hey",
    "good morning",
    "good afternoon",
    "good evening",
    "thanks",
    "thank you",
}
_KG = re.compile(r"(\d+(?:[.,]\d+)?\s*kg)\b", re.IGNORECASE)
_CM = re.compile(r"(\d+(?:[.,]\d+)?\s*cm)\b", re.IGNORECASE)
_METERS = re.compile(r"(\d+(?:[.,]\d+)?\s*m)\b", re.IGNORECASE)
PROGRESS_KEY = "progress"


def intake_namespace(user_id: str) -> tuple[str, str]:
    return (user_id, "intake")


def question_for(field_id: str) -> str:
    return _BY_ID[field_id][1]


def opening() -> str:
    return f"{NOTICE}\n\n{question_for('age')}"


def _blank() -> dict[str, Any]:
    return {"status": "pending", "next_field": "age", "answered": [], "skipped": []}


def _search(store: MemoryStore, namespace: tuple[str, ...]) -> list[Any]:
    try:
        return list(store.search(namespace, limit=100))
    except TypeError:
        return list(store.search(namespace))


def load_progress(store: MemoryStore, user_id: str) -> dict[str, Any]:
    """A missing record means the introduction is pending at age."""
    for item in _search(store, intake_namespace(user_id)):
        if getattr(item, "key", None) != PROGRESS_KEY:
            continue
        value = getattr(item, "value", None)
        if not isinstance(value, dict):
            return _blank()
        progress = _blank()
        status = value.get("status")
        if status in {"pending", "finished", "skipped"}:
            progress["status"] = status
        answered = value.get("answered")
        skipped = value.get("skipped")
        if isinstance(answered, list):
            progress["answered"] = [field for field in answered if field in _BY_ID]
        if isinstance(skipped, list):
            progress["skipped"] = [field for field in skipped if field in _BY_ID]
        if progress["status"] == "pending":
            progress["next_field"] = _next(progress) or ""
            if not progress["next_field"]:
                progress["status"] = "finished"
        else:
            progress["next_field"] = ""
        return progress
    return _blank()


def save_progress(store: MemoryStore, user_id: str, progress: dict[str, Any]) -> None:
    store.put(
        intake_namespace(user_id),
        PROGRESS_KEY,
        {
            "status": progress["status"],
            "next_field": progress["next_field"],
            "answered": list(progress["answered"]),
            "skipped": list(progress["skipped"]),
        },
    )


def _next(progress: dict[str, Any]) -> str | None:
    done = set(progress["answered"]) | set(progress["skipped"])
    for field_id in _ORDER:
        if field_id not in done:
            return field_id
    return None


def _finish(progress: dict[str, Any], status: str) -> None:
    progress["status"] = status
    progress["next_field"] = ""


def _advance(progress: dict[str, Any]) -> None:
    nxt = _next(progress)
    if nxt is None:
        _finish(progress, "finished")
        return
    progress["status"] = "pending"
    progress["next_field"] = nxt


def _phrase(text: str) -> str:
    return text.strip().lower().rstrip(".!")


def paired_measurements(text: str) -> tuple[str, str] | None:
    """Weight and height spans when the message has both. No body-mass reading."""
    if BmiTool().from_text(text) is None:
        return None
    weight = _KG.search(text)
    height = _CM.search(text) or _METERS.search(text)
    if weight is None or height is None:
        return None
    return weight.group(1).strip(), height.group(1).strip()


def _remember_field(store: MemoryStore, user_id: str, field_id: str, value: str) -> str:
    label = _BY_ID[field_id][0]
    text = f"{label}: {value.strip()}"
    store_field(store, user_id, field_id, text)
    return text


def _mark(progress: dict[str, Any], bucket: str, field_id: str) -> None:
    if field_id not in progress[bucket]:
        progress[bucket].append(field_id)


def public_status(store: MemoryStore, user_id: str) -> dict[str, str | None]:
    """Shape for GET /intake. The notice is only before the first question."""
    progress = load_progress(store, user_id)
    status = str(progress["status"])
    nxt = progress["next_field"] or None
    if status != "pending":
        nxt = None
    notice = NOTICE if status == "pending" and nxt == "age" else None
    question = question_for(nxt) if isinstance(nxt, str) and nxt in _BY_ID else None
    return {"status": status, "next_field": nxt, "notice": notice, "question": question}


def handle_intake(store: MemoryStore, user_id: str, human: str) -> tuple[bool, str | None, list[str]]:
    """Return handled, reply, and texts that must stay out of a trace.

    handled is false when the introduction is already done, or when the message
    is a diagnosis, dose, or medication request that the content filter owns.
    """
    from .guardrails import _blocked_reply

    progress = load_progress(store, user_id)
    if progress["status"] != "pending":
        return False, None, []
    text = human.strip()
    if text != INTAKE_START:
        blocked = _blocked_reply(text)
        if blocked and not blocked.startswith("BMI"):
            return False, None, []
    field_id = str(progress["next_field"] or _next(progress) or "")
    if field_id not in _BY_ID:
        _finish(progress, "finished")
        save_progress(store, user_id, progress)
        return True, CLOSE, []

    if text == INTAKE_START:
        reply = opening() if field_id == "age" and not progress["answered"] and not progress["skipped"] else question_for(field_id)
        return True, reply, [INTAKE_START, reply]

    sensitive: list[str] = []
    phrase = _phrase(text)
    if phrase in _SKIP_REST:
        for remaining in _ORDER:
            if remaining not in progress["answered"] and remaining not in progress["skipped"]:
                _mark(progress, "skipped", remaining)
        _finish(progress, "skipped")
        save_progress(store, user_id, progress)
        return True, CLOSE, [text, CLOSE]

    if phrase in _SKIP_ONE:
        _mark(progress, "skipped", field_id)
        _advance(progress)
        save_progress(store, user_id, progress)
        reply = CLOSE if progress["status"] != "pending" else question_for(progress["next_field"])
        return True, reply, [text, reply]

    if not text or _phrase(text) in _GREETINGS or text.endswith("?"):
        reply = question_for(field_id)
        return True, reply, [reply]

    pair = paired_measurements(text) if field_id in {"weight", "height"} else None
    if pair is not None:
        weight, height = pair
        sensitive.append(_remember_field(store, user_id, "weight", weight))
        sensitive.append(_remember_field(store, user_id, "height", height))
        _mark(progress, "answered", "weight")
        _mark(progress, "answered", "height")
        sensitive.append(text)
    else:
        sensitive.append(_remember_field(store, user_id, field_id, text))
        _mark(progress, "answered", field_id)
        sensitive.append(text)
    _advance(progress)
    save_progress(store, user_id, progress)
    reply = CLOSE if progress["status"] != "pending" else question_for(progress["next_field"])
    sensitive.append(reply)
    return True, reply, sensitive

