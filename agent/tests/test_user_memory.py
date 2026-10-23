"""Personal notes stay on one subject."""

from langchain_core.messages import HumanMessage

from app.internal.user_memory import (
    UserMemoryMiddleware,
    existing_texts,
    memory_text,
    remember,
    user_id_from_runtime,
)


class _Store:
    def __init__(self) -> None:
        self.rows: dict[tuple[str, ...], dict[str, dict]] = {}

    def put(self, namespace, key, value) -> None:
        self.rows.setdefault(tuple(namespace), {})[key] = value

    def search(self, namespace, limit: int = 100):
        del limit

        class Item:
            def __init__(self, key, value) -> None:
                self.key = key
                self.value = value

        rows = self.rows.get(tuple(namespace), {})
        return [Item(key, value) for key, value in rows.items()]

    def delete(self, namespace, key) -> None:
        self.rows.get(tuple(namespace), {}).pop(key, None)


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


def test_user_id_comes_from_the_run_config_when_the_runtime_has_none(monkeypatch):
    monkeypatch.setattr(
        "langgraph.config.get_config",
        lambda: {"configurable": {"langgraph_auth_user_id": "account-9"}},
    )

    class Runtime:
        store = None

    assert user_id_from_runtime(Runtime()) == "account-9"


def test_middleware_does_nothing_without_a_user():
    store = _Store()

    class Runtime:
        pass

    runtime = Runtime()
    runtime.store = store
    state = {"messages": [HumanMessage(content="Please remember that I walk daily.")]}
    assert UserMemoryMiddleware().before_agent(state, runtime) is None
    assert existing_texts(store, "user-1") == []


def test_platform_store_is_not_where_notes_are_written(monkeypatch):
    account = _Store()
    monkeypatch.setattr("app.internal.persistence.open_memory_store", lambda: account)

    class PlatformStore:
        def __init__(self) -> None:
            self.rows: list[tuple] = []

        def put(self, namespace, key, value) -> None:
            self.rows.append((namespace, key, value))

        def search(self, namespace, limit: int = 100):
            del namespace, limit
            return []

    PlatformStore.__module__ = "langgraph_runtime_inmem.store"
    platform = PlatformStore()

    class Runtime:
        store = platform
        config = {"configurable": {"langgraph_auth_user": {"identity": "user-1"}}}

    state = {"messages": [HumanMessage(content="Please remember that I walk daily.")]}
    assert UserMemoryMiddleware().before_agent(state, Runtime()) is not None
    assert platform.rows == []
    assert existing_texts(account, "user-1") == ["I walk daily."]
