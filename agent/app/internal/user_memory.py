"""Personal notes keyed by the JWT subject. Not citations and not training data."""

from __future__ import annotations

import asyncio
import hashlib
from typing import Any, Protocol

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import SystemMessage
from langgraph.runtime import Runtime

REMEMBER = "Please remember that "
NAMESPACE_NAME = "memories"


class MemoryStore(Protocol):
    def put(self, namespace: tuple[str, ...], key: str, value: dict[str, Any]) -> None: ...

    def search(self, namespace: tuple[str, ...], limit: int = 100) -> list[Any]: ...

    def delete(self, namespace: tuple[str, ...], key: str) -> None: ...


def memory_text(human: str) -> str | None:
    """Remainder of a remember-sentence, or None when there is nothing to store."""
    if not human.startswith(REMEMBER):
        return None
    text = human[len(REMEMBER) :].strip()
    return text or None


def memory_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def namespace_for(user_id: str) -> tuple[str, str]:
    return (user_id, NAMESPACE_NAME)


def _item_text(item: Any) -> str:
    value = getattr(item, "value", item)
    if isinstance(value, dict):
        return str(value.get("text") or "").strip()
    return ""


def _search(store: MemoryStore, namespace: tuple[str, ...]) -> list[Any]:
    try:
        return list(store.search(namespace, limit=100))  # type: ignore[call-arg]
    except TypeError:
        return list(store.search(namespace))


def existing_texts(store: MemoryStore, user_id: str) -> list[str]:
    found: list[str] = []
    for item in _search(store, namespace_for(user_id)):
        text = _item_text(item)
        if text and text not in found:
            found.append(text)
    return found


def _item_field(item: Any) -> str | None:
    value = getattr(item, "value", item)
    if not isinstance(value, dict):
        return None
    field = value.get("field")
    if not isinstance(field, str) or not field.strip():
        return None
    return field


def store_field(store: MemoryStore, user_id: str, field_id: str, text: str) -> str | None:
    """Store one introduction answer. The same field key is not written again."""
    if not user_id or not field_id or not text.strip():
        return None
    namespace = namespace_for(user_id)
    for item in _search(store, namespace):
        if getattr(item, "key", None) == field_id:
            return _item_text(item) or None
    store.put(namespace, field_id, {"text": text, "field": field_id})
    return text


def list_memories(store: MemoryStore, user_id: str) -> list[dict[str, str | None]]:
    items: list[dict[str, str | None]] = []
    for item in _search(store, namespace_for(user_id)):
        text = _item_text(item)
        key = getattr(item, "key", None)
        if not text or not key:
            continue
        items.append({"id": str(key), "text": text, "field": _item_field(item)})
    return items


def delete_memory(store: MemoryStore, user_id: str, key: str) -> bool:
    """Delete one memory for this account. Intake progress is left alone."""
    if not user_id or not key:
        return False
    namespace = namespace_for(user_id)
    if not any(getattr(item, "key", None) == key for item in _search(store, namespace)):
        return False
    store.delete(namespace, key)
    return True


def remember(store: MemoryStore, user_id: str, human: str) -> str | None:
    """Store a new note. The same text is not stored twice."""
    text = memory_text(human)
    if text is None or not user_id:
        return None
    if text in existing_texts(store, user_id):
        return text
    store.put(namespace_for(user_id), memory_key(text), {"text": text})
    return text


def notes_message(notes: list[str]) -> SystemMessage | None:
    if not notes:
        return None
    lines = "\n".join(f"- {note}" for note in notes)
    return SystemMessage(
        content=(
            "Personal notes this person asked to remember. "
            "They are not sources and not citations. "
            "Do not treat them as evidence. "
            "A topic with no kept sources is still not covered. "
            "When a note names a condition, set aside a suggestion that conflicts with it "
            "when the kept evidence supports a habit that does not conflict. "
            "Do not diagnose, name a medication, or state a dose.\n"
            f"{lines}"
        )
    )


def _configurable(runtime: Runtime | None) -> dict[str, Any] | None:
    """The run config. Current LangGraph runtimes do not carry it themselves."""
    config = getattr(runtime, "config", None)
    if isinstance(config, dict):
        configurable = config.get("configurable")
        if isinstance(configurable, dict):
            return configurable
    try:
        from langgraph.config import get_config

        config = get_config()
    except Exception:
        return None
    if not isinstance(config, dict):
        return None
    configurable = config.get("configurable")
    return configurable if isinstance(configurable, dict) else None


def user_id_from_runtime(runtime: Runtime | None) -> str | None:
    configurable = _configurable(runtime)
    if configurable is None:
        return None
    user = configurable.get("langgraph_auth_user") or {}
    if isinstance(user, dict):
        identity = user.get("identity")
    else:
        identity = getattr(user, "identity", None)
    if identity is None or identity == "":
        identity = configurable.get("langgraph_auth_user_id")
    if identity is None or identity == "":
        return None
    return str(identity)


def _platform_store(store: Any) -> bool:
    """The API injects its own store. Account memories stay in the sqlite store."""
    module = type(store).__module__
    return module.startswith("langgraph.store.memory") or module.startswith("langgraph_runtime")


def store_from_runtime(runtime: Runtime | None) -> MemoryStore | None:
    store = getattr(runtime, "store", None) if runtime is not None else None
    if store is not None and not _platform_store(store):
        return store
    if user_id_from_runtime(runtime) is None:
        return None
    from .persistence import open_memory_store

    return open_memory_store()


class UserMemoryMiddleware(AgentMiddleware):
    """Read and write (sub, memories). No-op without a store or an auth user."""

    def _apply(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        store = store_from_runtime(runtime)
        user_id = user_id_from_runtime(runtime)
        if store is None or not user_id:
            return None
        from .guardrails import latest_human

        remember(store, user_id, latest_human(state))
        message = notes_message(existing_texts(store, user_id))
        if message is None:
            return None
        for prior in state.get("messages") or []:
            if getattr(prior, "content", "") == message.content:
                return None
        return {"messages": [message]}

    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        return self._apply(state, runtime)

    async def abefore_agent(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        return await asyncio.to_thread(self._apply, state, runtime)
