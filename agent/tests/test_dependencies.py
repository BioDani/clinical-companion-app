"""Message conversion and graph invocation used by the chat route."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.dependencies import (
    get_session_id,
    last_assistant,
    message_text,
    run_companion,
    to_lc_messages,
)


def test_session_id_is_the_header_value():
    assert get_session_id("sess-1") == "sess-1"
    assert get_session_id(None) is None


def test_message_text_passes_strings_through():
    assert message_text("hello") == "hello"


def test_message_text_joins_text_parts_only():
    content = [
        "A",
        {"type": "text", "text": "B"},
        {"type": "image_url", "text": "ignored"},
        {"type": "text", "text": None},
        5,
    ]
    assert message_text(content) == "AB"


def test_message_text_none_and_other():
    assert message_text(None) == ""
    assert message_text(3) == "3"


def test_to_lc_messages_maps_roles_and_flattens_content():
    messages = to_lc_messages(
        [
            SimpleNamespace(role="system", content="rules"),
            SimpleNamespace(
                role="assistant",
                content=["see ", {"type": "text", "text": "you"}],
            ),
            SimpleNamespace(role="user", content=None),
            SimpleNamespace(role="tool", content="fallback"),
        ]
    )
    assert isinstance(messages[0], SystemMessage)
    assert messages[0].content == "rules"
    assert isinstance(messages[1], AIMessage)
    assert messages[1].content == "see you"
    assert isinstance(messages[2], HumanMessage)
    assert messages[2].content == ""
    assert isinstance(messages[3], HumanMessage)
    assert messages[3].content == "fallback"


def test_last_assistant_skips_user_turns():
    messages = [
        HumanMessage(content="question"),
        AIMessage(content=[{"type": "text", "text": "answer"}]),
        HumanMessage(content="again"),
    ]
    assert last_assistant(messages) == "answer"


def test_last_assistant_accepts_duck_typed_ai_message():
    class FakeAI:
        type = "ai"
        content = "from-node"

    assert last_assistant([HumanMessage(content="q"), FakeAI()]) == "from-node"
    assert last_assistant([HumanMessage(content="q")]) == ""


def test_run_companion_returns_last_assistant_text(monkeypatch: pytest.MonkeyPatch):
    captured: dict = {}

    def invoke(state, config):
        captured["state"] = state
        captured["config"] = config
        return {"messages": [HumanMessage(content="q"), AIMessage(content="done")]}

    monkeypatch.setattr("app.dependencies.companion.invoke", invoke)
    text = run_companion([SimpleNamespace(role="user", content="hi")], "thread-1")

    assert text == "done"
    assert captured["config"]["configurable"]["thread_id"] == "thread-1"
    assert captured["state"]["retrieved"] == ""
    assert captured["state"]["issues"] == []
    assert captured["state"]["route"] == ""
    assert captured["state"]["search_query"] == ""
    assert captured["state"]["searches"] == 0
    assert captured["state"]["coverage"] == ""
    assert isinstance(captured["state"]["messages"][0], HumanMessage)
    assert captured["state"]["messages"][0].content == "hi"


def test_run_companion_empty_result_is_blank(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "app.dependencies.companion.invoke",
        lambda state, config: {},
    )
    assert run_companion([SimpleNamespace(role="user", content="hi")], "t") == ""


def test_run_companion_maps_value_error_to_503(monkeypatch: pytest.MonkeyPatch):
    def invoke(state, config):
        raise ValueError("Set HF_TOKEN")

    monkeypatch.setattr("app.dependencies.companion.invoke", invoke)
    with pytest.raises(HTTPException) as exc:
        run_companion([SimpleNamespace(role="user", content="hi")], "t")
    assert exc.value.status_code == 503
    assert exc.value.detail == "Set HF_TOKEN"


def test_run_companion_maps_other_errors_to_502(monkeypatch: pytest.MonkeyPatch):
    def invoke(state, config):
        raise RuntimeError("upstream down")

    monkeypatch.setattr("app.dependencies.companion.invoke", invoke)
    with pytest.raises(HTTPException) as exc:
        run_companion([SimpleNamespace(role="user", content="hi")], "t")
    assert exc.value.status_code == 502
    assert exc.value.detail == "Agent failed: upstream down"
