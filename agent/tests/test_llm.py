"""Hugging Face chat model. The router call stays mocked."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.internal.llm import HF_OPENAI_BASE, HuggingFaceChat

pytestmark = pytest.mark.optional


def test_constructing_the_chat_model_does_not_read_the_token(
    monkeypatch: pytest.MonkeyPatch,
):
    def fail() -> str:
        raise AssertionError("token should not be read")

    monkeypatch.setattr("app.internal.llm.hf_token", fail)
    HuggingFaceChat()


def test_generate_requires_hf_token(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HF_TOKEN", "")
    with pytest.raises(ValueError, match="HF_TOKEN"):
        HuggingFaceChat()._generate([HumanMessage(content="hi")])


def test_generate_uses_the_huggingface_router(monkeypatch: pytest.MonkeyPatch):
    captured: dict = {}

    class FakeClient:
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            captured["messages"] = messages
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content="ok"))]
            )

    def fake_openai(**kwargs):
        captured["kwargs"] = kwargs
        return FakeClient()

    monkeypatch.setenv("HF_TOKEN", "hf_test")
    monkeypatch.setenv("HF_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
    monkeypatch.setattr("app.internal.llm.ChatOpenAI", fake_openai)
    result = HuggingFaceChat()._generate([HumanMessage(content="hi")])

    assert captured["kwargs"]["base_url"] == HF_OPENAI_BASE
    assert captured["kwargs"]["api_key"] == "hf_test"
    assert captured["kwargs"]["model"] == "Qwen/Qwen3-4B-Instruct-2507"
    assert captured["kwargs"]["temperature"] == 0
    assert result.generations[0].message.content == "ok"
    assert captured["messages"][0].content == "hi"


def test_bind_tools_delegates_to_the_router_client(monkeypatch: pytest.MonkeyPatch):
    captured: dict = {}

    class FakeClient:
        def bind_tools(self, tools, **kwargs):
            captured["tools"] = tools
            captured["kwargs"] = kwargs
            return "bound"

    monkeypatch.setenv("HF_TOKEN", "hf_test")
    monkeypatch.setattr("app.internal.llm.ChatOpenAI", lambda **kwargs: FakeClient())

    def search(query: str) -> str:
        return query

    assert HuggingFaceChat().bind_tools([search], tool_choice="auto") == "bound"
    assert captured["tools"] == [search]
    assert captured["kwargs"]["tool_choice"] == "auto"
