"""Introduction answers stay on one account and out of search text."""

from langchain_core.messages import HumanMessage, SystemMessage

from app.internal.graph import (
    IntakeMiddleware,
    _middleware,
    lookup_lines,
)
from app.internal.guardrails import ContentFilterMiddleware
from app.internal.intake import CLOSE, handle_intake, load_progress, save_progress
from app.internal.trace_mask import NOTES_PREFIX, redact, note_sensitive
from app.internal.user_memory import delete_memory, existing_texts, list_memories
from app.routers.memories import app, get_store
from fastapi.testclient import TestClient
from tests.tokens import make_token


class Store:
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


class Runtime:
    def __init__(self, store: Store, user: str) -> None:
        self.store = store
        self.config = {"configurable": {"langgraph_auth_user": {"identity": user}}}


def test_sleeping_hours_are_stored_once_without_the_account_id():
    store = Store()
    save_progress(
        store,
        "user-1",
        {
            "status": "pending",
            "next_field": "sleeping_hours",
            "answered": ["age", "weight", "height"],
            "skipped": [],
        },
    )
    handle_intake(store, "user-1", "7 hours a night")
    progress = load_progress(store, "user-1")
    progress["status"] = "pending"
    progress["next_field"] = "sleeping_hours"
    progress["answered"] = [field for field in progress["answered"] if field != "sleeping_hours"]
    save_progress(store, "user-1", progress)
    handle_intake(store, "user-1", "7 hours a night")
    texts = existing_texts(store, "user-1")
    assert texts == ["Sleeping hours: 7 hours a night"]
    assert "user-1" not in texts[0]
    assert existing_texts(store, "user-2") == []


def test_skip_stores_nothing_and_skip_the_rest_ends_the_set():
    store = Store()
    save_progress(
        store,
        "user-1",
        {"status": "pending", "next_field": "weight", "answered": ["age"], "skipped": []},
    )
    handle_intake(store, "user-1", "skip")
    assert existing_texts(store, "user-1") == []
    assert load_progress(store, "user-1")["next_field"] == "height"
    handle_intake(store, "user-1", "skip the rest")
    assert load_progress(store, "user-1")["status"] == "skipped"
    assert existing_texts(store, "user-1") == []
    handled, _reply, _sensitive = handle_intake(store, "user-1", "42")
    assert handled is False


def test_height_and_weight_during_intake_are_not_a_body_mass_reading():
    store = Store()
    save_progress(
        store,
        "user-1",
        {"status": "pending", "next_field": "weight", "answered": ["age"], "skipped": []},
    )
    _handled, reply, _sensitive = handle_intake(store, "user-1", "78 kg and 170 cm")
    assert reply is not None and "BMI" not in reply
    texts = existing_texts(store, "user-1")
    assert "Weight: 78 kg" in texts
    assert "Height: 170 cm" in texts
    save_progress(
        store,
        "user-1",
        {"status": "finished", "next_field": "", "answered": ["weight", "height"], "skipped": []},
    )
    assert (
        IntakeMiddleware().before_agent(
            {"messages": [HumanMessage(content="78 kg and 170 cm")]},
            Runtime(store, "user-1"),
        )
        is None
    )
    blocked = ContentFilterMiddleware().before_agent(
        {"messages": [HumanMessage(content="78 kg and 170 cm")]},
        Runtime(store, "user-1"),
    )
    assert blocked is not None
    assert "BMI" in blocked["messages"][0].content


def test_deleting_a_memory_does_not_restart_the_introduction():
    store = Store()
    save_progress(
        store,
        "user-1",
        {"status": "finished", "next_field": "", "answered": ["age"], "skipped": []},
    )
    store.put(("user-1", "memories"), "age", {"text": "Age: 42", "field": "age"})
    assert delete_memory(store, "user-1", "age") is True
    assert list_memories(store, "user-1") == []
    assert load_progress(store, "user-1")["status"] == "finished"
    assert delete_memory(store, "user-1", "age") is False


def test_lookup_skips_intake_and_personal_notes():
    lines = lookup_lines(
        [
            HumanMessage(content="42", additional_kwargs={"intake": True}),
            SystemMessage(content=f"{NOTES_PREFIX}\n- Age: 42"),
            HumanMessage(content="What is an evidence-based breakfast pattern?"),
        ]
    )
    assert lines == ["User: What is an evidence-based breakfast pattern?"]


def test_open_webui_path_does_not_mount_intake():
    plain = _middleware(object(), with_memory=False)
    signed_in = _middleware(object(), with_memory=True)
    assert all(type(item).__name__ != "IntakeMiddleware" for item in plain)
    assert type(signed_in[0]).__name__ == "IntakeMiddleware"
    assert type(signed_in[1]).__name__ == "ContentFilterMiddleware"


def test_trace_mask_redacts_intake_and_keeps_the_account_stamp():
    note_sensitive("42")
    note_sensitive("Sleeping hours: 7 hours a night")
    masked = redact(
        {
            "langfuse_user_id": "user-1",
            "langfuse_session_id": "thread-1",
            "langfuse_tags": ["clinical-companion"],
            "content": "42",
            "notes": f"{NOTES_PREFIX}\n- Age: 42",
        }
    )
    assert masked["langfuse_user_id"] == "user-1"
    assert masked["langfuse_session_id"] == "thread-1"
    assert masked["langfuse_tags"] == ["clinical-companion"]
    assert masked["content"] == "[redacted]"
    assert masked["notes"] == "[redacted]"


def test_greeting_is_not_stored():
    store = Store()
    _handled, reply, _sensitive = handle_intake(store, "user-1", "hello")
    assert reply is not None and "What is your age?" in reply
    assert existing_texts(store, "user-1") == []


def test_intake_routes_list_and_delete_for_the_token_subject():
    store = Store()
    app.dependency_overrides[get_store] = lambda: store
    client = TestClient(app)
    missing = client.get("/intake")
    assert missing.status_code == 401
    header = {"Authorization": f"Bearer {make_token(sub='user-1')}"}
    body = client.get("/intake", headers=header).json()
    assert body["status"] == "pending"
    assert body["next_field"] == "age"
    assert body["notice"] is not None
    assert "wellnest" in body["notice"]
    assert "optional" in body["notice"]
    assert "never be used to train" in body["notice"]
    handle_intake(store, "user-1", "42")
    listed = client.get("/memories", headers=header).json()["items"]
    assert listed[0]["text"] == "Age: 42"
    assert listed[0]["field"] == "age"
    other = client.get("/memories", headers={"Authorization": f"Bearer {make_token(sub='user-2')}"})
    assert other.json()["items"] == []
    gone = client.delete(f"/memories/{listed[0]['id']}", headers=header)
    assert gone.status_code == 204
    assert client.get("/memories", headers=header).json()["items"] == []
    missing_item = client.delete("/memories/age", headers=header)
    assert missing_item.status_code == 404
    assert load_progress(store, "user-1")["status"] == "pending"
    app.dependency_overrides.clear()


def test_middleware_ends_the_turn_with_the_notice():
    store = Store()
    result = IntakeMiddleware().before_agent(
        {"messages": [HumanMessage(content="INTAKE_START")]},
        Runtime(store, "user-1"),
    )
    assert result is not None
    assert result["jump_to"] == "end"
    text = result["messages"][0].content
    assert "wellnest" in text
    assert "optional" in text
    assert "never be used to train" in text
    assert "What is your age?" in text
    assert "BMI" not in text
    assert CLOSE not in text
