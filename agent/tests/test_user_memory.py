"""Personal notes stay on one subject."""

from langchain_core.messages import HumanMessage

from app.internal.user_memory import (
    UserMemoryMiddleware,
    existing_texts,
    memory_text,
    remember,
)


class _Store:
    def __init__(self) -> None:
        self.rows: dict[tuple[str, ...], dict[str, dict]] = {}

    def put(self, namespace, key, value) -> None:
        self.rows.setdefault(tuple(namespace), {})[key] = value

    def search(self, namespace):
        class Item:
            def __init__(self, value) -> None:
                self.value = value

        return [Item(value) for value in self.rows.get(tuple(namespace), {}).values()]


def test_remember_sentence_is_stored_once():
    assert memory_text("Please remember that ") is None
    assert memory_text("Please remember that    ") is None
    store = _Store()
    assert remember(store, "user-1", "Please remember that I avoid lactose.") == (
        "I avoid lactose."
    )
    remember(store, "user-1", "Please remember that I avoid lactose.")
    assert existing_texts(store, "user-1") == ["I avoid lactose."]
    assert existing_texts(store, "user-2") == []


def test_middleware_does_nothing_without_a_store():
    class Runtime:
        store = None
        config = {"configurable": {"langgraph_auth_user": {"identity": "user-1"}}}

    state = {"messages": [HumanMessage(content="Please remember that I walk daily.")]}
    assert UserMemoryMiddleware().before_agent(state, Runtime()) is None
    assert memory_text("Please remember that I walk daily.") == "I walk daily."
