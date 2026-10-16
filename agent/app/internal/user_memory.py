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

    def search(self, namespace: tuple[str, ...]) -> list[Any]: ...


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


def existing_texts(store: MemoryStore, user_id: str) -> list[str]:
    found: list[str] = []
    for item in store.search(namespace_for(user_id)):
        text = _item_text(item)
        if text and text not in found:
            found.append(text)
    return found


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
            "A topic with no kept sources is still not covered.\n"
            f"{lines}"
        )
    )


def user_id_from_runtime(runtime: Runtime | None) -> str | None:
    if runtime is None:
        return None
    config = getattr(runtime, "config", None)
    configurable: Any = None
    if isinstance(config, dict):
        configurable = config.get("configurable")
    if not isinstance(configurable, dict):
        return None
    user = configurable.get("langgraph_auth_user") or {}
    if isinstance(user, dict):
        identity = user.get("identity")
    else:
        identity = getattr(user, "identity", None)
    if identity is None or identity == "":
        return None
    return str(identity)


def store_from_runtime(runtime: Runtime | None) -> MemoryStore | None:
    if runtime is None:
        return None
    store = getattr(runtime, "store", None)
    if store is None:
        return None
    return store


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
