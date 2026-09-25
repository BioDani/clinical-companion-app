"""smolagents payload shaping. Inference itself stays mocked."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.internal.llm import _chat_content, _content_text, complete, get_model

pytestmark = pytest.mark.optional


@pytest.fixture(autouse=True)
def _clear_model_cache():
    get_model.cache_clear()
    yield
    get_model.cache_clear()


def test_content_text_flattens_parts():
    class Part:
        text = "tail"

    assert _content_text("plain") == "plain"
    assert (
        _content_text(
            [
                "A",
                {"type": "text", "text": "B"},
                {"type": "image", "text": "no"},
                Part(),
            ]
        )
        == "ABtail"
    )
    assert _content_text(None) == ""
    assert _content_text(4) == "4"


def test_chat_content_is_a_text_block():
    assert _chat_content("hello") == [{"type": "text", "text": "hello"}]


def test_get_model_requires_hf_token(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HF_TOKEN", "")
    with pytest.raises(ValueError, match="HF_TOKEN"):
        get_model()


def test_complete_maps_roles_and_strips_reply(monkeypatch: pytest.MonkeyPatch):
    captured: dict = {}

    class FakeModel:
        def generate(self, payload):
            captured["payload"] = payload
            return SimpleNamespace(content="  reply  ")

    monkeypatch.setattr("app.internal.llm.get_model", lambda: FakeModel())
    text = complete(
        [
            SystemMessage(content="rules"),
            HumanMessage(content=[{"type": "text", "text": "question"}]),
            AIMessage(content="earlier"),
        ]
    )

    assert text == "reply"
    roles = [message.role for message in captured["payload"]]
    contents = [message.content for message in captured["payload"]]
    assert roles == ["system", "user", "assistant"]
    assert contents[1] == [{"type": "text", "text": "question"}]


def test_complete_maps_tool_and_unknown_roles(monkeypatch: pytest.MonkeyPatch):
    captured: dict = {}

    class FakeModel:
        def generate(self, payload):
            captured["payload"] = payload
            return SimpleNamespace(content="ok")

    class Tool:
        type = "tool"
        content = "call"

    class Odd:
        type = "function"
        content = "nope"

    monkeypatch.setattr("app.internal.llm.get_model", lambda: FakeModel())
    assert complete([Tool(), Odd()]) == "ok"
    assert [message.role for message in captured["payload"]] == ["tool", "user"]


def test_complete_uses_raw_result_when_content_missing(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "app.internal.llm.get_model",
        lambda: SimpleNamespace(generate=lambda payload: "  bare  "),
    )
    assert complete([HumanMessage(content="hi")]) == "bare"
