"""Companion graph: guardrail, book, web search, BMI, and smalltalk."""

from __future__ import annotations

import uuid

import pytest
from langchain_core.messages import HumanMessage

from app.internal.graph import (
    ANSWER_PROMPT,
    BOOK_TITLE,
    GENERAL_PROMPT,
    HABIT_BOUNDARY,
    REFUSAL,
    SMALL_TALK_PROMPT,
    build_graph,
    companion,
)

pytestmark = pytest.mark.optional


def _state(text: str) -> dict:
    return {
        "messages": [HumanMessage(content=text)],
        "retrieved": "",
        "issues": [],
        "route": "",
        "search_query": "",
        "searches": 0,
        "coverage": "",
    }


def _config() -> dict:
    return {"configurable": {"thread_id": f"unit-{uuid.uuid4().hex}"}}


def _assistant(result: dict) -> list[str]:
    return [
        message.content
        for message in result["messages"]
        if getattr(message, "type", "") == "ai"
    ]


def _complete(replies):
    def fake(messages, max_tokens=512):
        return next(replies)

    return fake


def test_compiled_graph_routes_through_guard_and_answer():
    drawn = build_graph().get_graph()
    nodes = set(drawn.nodes)
    edges = {(edge.source, edge.target) for edge in drawn.edges}
    assert {
        "guard",
        "classify",
        "refuse",
        "habits",
        "smalltalk",
        "bmi",
        "rag_search",
        "rag_grade",
        "rag_answer",
        "web_search",
        "web_answer",
        "general",
    } <= nodes
    assert ("__start__", "guard") in edges
    assert ("guard", "refuse") in edges
    assert ("guard", "habits") in edges
    assert ("guard", "classify") in edges
    assert ("classify", "smalltalk") in edges
    assert ("classify", "bmi") in edges
    assert ("classify", "rag_search") in edges
    assert ("rag_search", "rag_grade") in edges
    assert ("rag_grade", "rag_answer") in edges
    assert ("rag_grade", "web_search") in edges
    assert ("web_search", "web_answer") in edges
    assert ("web_search", "general") in edges
    for node in ("refuse", "habits", "smalltalk", "bmi", "rag_answer", "web_answer", "general"):
        assert (node, "__end__") in edges


def test_diagnosis_is_refused_without_calling_the_model(monkeypatch: pytest.MonkeyPatch):
    def fail(messages, max_tokens=512):
        raise AssertionError("model should not be called")

    monkeypatch.setattr("app.internal.graph.complete", fail)
    result = companion.invoke(_state("Do I have diabetes?"), _config())

    assert result["messages"][-1].content == REFUSAL
    assert result["issues"] == ["Do I have diabetes?"]
    assert result["retrieved"] == ""


def test_what_to_take_stops_before_search_and_names_no_drug(
    monkeypatch: pytest.MonkeyPatch,
):
    def fail_complete(messages, max_tokens=512):
        raise AssertionError("model should not be called")

    def fail_search(self, query):
        raise AssertionError("book should not be searched")

    def fail_web(query):
        raise AssertionError("web should not be searched")

    monkeypatch.setattr("app.internal.graph.complete", fail_complete)
    monkeypatch.setattr("app.internal.book.BookIndex.search", fail_search)
    monkeypatch.setattr("app.internal.web_search.search_web", fail_web)
    result = companion.invoke(_state("what should I take for Alzheimer?"), _config())

    answer = result["messages"][-1].content
    assert answer == HABIT_BOUNDARY
    assert "donepezil" not in answer.lower()
    assert "memantine" not in answer.lower()
    assert result["issues"] == ["what should I take for Alzheimer?"]
    assert result["retrieved"] == ""


def test_diet_question_answers_from_book_passages(monkeypatch: pytest.MonkeyPatch):
    calls: list[dict] = []
    queries: list[str] = []
    replies = iter(["breakfast fiber", "SUFFICIENT", "a quoted answer"])

    def fake_complete(messages, max_tokens=512):
        calls.append({"messages": messages, "max_tokens": max_tokens})
        return next(replies)

    monkeypatch.setattr(
        "app.internal.book.BookIndex.search",
        lambda self, query: queries.append(query) or "page 3: fiber at breakfast",
    )
    monkeypatch.setattr("app.internal.graph.complete", fake_complete)
    result = companion.invoke(_state("What is a good diet?"), _config())

    answer = result["messages"][-1].content
    assert queries == ["breakfast fiber"]
    assert result["retrieved"] == "page 3: fiber at breakfast"
    assert result["searches"] == 1
    assert calls[-1]["max_tokens"] == 1024
    assert calls[-1]["messages"][0].content == ANSWER_PROMPT
    assert "Elaborate" in ANSWER_PROMPT
    assert "BMI" not in ANSWER_PROMPT
    assert calls[-1]["messages"][-1].content == "Retrieved context:\npage 3: fiber at breakfast"
    assert answer.startswith("a quoted answer")
    assert f"{BOOK_TITLE}, page 3" in answer
    assert result["issues"] == []


def test_insufficient_book_grade_searches_the_web_once(monkeypatch: pytest.MonkeyPatch):
    queries: list[str] = []
    web_queries: list[str] = []
    replies = iter(["breakfast fiber", "INSUFFICIENT", "from the web"])

    monkeypatch.setattr(
        "app.internal.book.BookIndex.search",
        lambda self, query: queries.append(query) or "page 2: unrelated",
    )
    monkeypatch.setattr(
        "app.internal.web_search.search_web",
        lambda query: web_queries.append(query)
        or [{"title": "Fiber", "url": "https://example.com/fiber", "content": "eat fiber"}],
    )
    monkeypatch.setattr("app.internal.graph.complete", _complete(replies))
    result = companion.invoke(_state("What is a good diet?"), _config())

    answer = result["messages"][-1].content
    assert queries == ["breakfast fiber"]
    assert web_queries == ["What is a good diet?"]
    assert "from the web" in answer
    assert "https://example.com/fiber" in answer
    assert result["searches"] == 1


def test_next_question_starts_a_fresh_search(monkeypatch: pytest.MonkeyPatch):
    replies = iter(
        [
            "desayuno",
            "SUFFICIENT",
            "first answer",
            "ejercicio",
            "SUFFICIENT",
            "second answer",
        ]
    )
    monkeypatch.setattr(
        "app.internal.book.BookIndex.search",
        lambda self, query: f"page 1: {query}",
    )
    monkeypatch.setattr("app.internal.graph.complete", _complete(replies))
    config = _config()
    first = companion.invoke(_state("What is a good diet?"), config)
    second = companion.invoke(_state("How should I exercise?"), config)

    assert first["retrieved"] == "page 1: desayuno"
    assert second["retrieved"] == "page 1: ejercicio"
    assert second["searches"] == 1
    assert "page 1" in second["messages"][-1].content


def test_height_and_weight_returns_bmi_without_the_book(
    monkeypatch: pytest.MonkeyPatch,
):
    def fail_search(self, query):
        raise AssertionError("book should not be searched")

    def fail_complete(messages, max_tokens=512):
        raise AssertionError("model should not be called")

    monkeypatch.setattr("app.internal.book.BookIndex.search", fail_search)
    monkeypatch.setattr("app.internal.graph.complete", fail_complete)
    result = companion.invoke(
        _state("I weigh 82 kg and I am 1.78 m. What habits help?"),
        _config(),
    )

    assert _assistant(result) == ["BMI 25.9 (overweight)"]
    assert result["retrieved"] == ""


def test_bmi_without_measurements_asks_for_them(monkeypatch: pytest.MonkeyPatch):
    def fail_search(self, query):
        raise AssertionError("book should not be searched")

    monkeypatch.setattr("app.internal.book.BookIndex.search", fail_search)
    monkeypatch.setattr(
        "app.internal.graph.complete",
        lambda messages, max_tokens=512: "Please send weight in kg and height in cm.",
    )
    result = companion.invoke(_state("What is my BMI?"), _config())

    assert _assistant(result) == ["Please send weight in kg and height in cm."]
    assert result["retrieved"] == ""


def test_hello_is_one_smalltalk_reply(monkeypatch: pytest.MonkeyPatch):
    calls: list = []

    def fake_complete(messages, max_tokens=512):
        calls.append(messages)
        return "Hello from companion."

    monkeypatch.setattr("app.internal.graph.complete", fake_complete)
    result = companion.invoke(_state("hello"), _config())

    assert len(calls) == 1
    assert calls[0][0].content == SMALL_TALK_PROMPT
    assert result["messages"][-1].content == "Hello from companion."
    assert result["retrieved"] == ""


def test_question_missing_from_book_and_web_uses_general_knowledge(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[dict] = []
    replies = iter(["paris", "Paris is the capital of France."])

    def fake_complete(messages, max_tokens=512):
        calls.append({"messages": messages, "max_tokens": max_tokens})
        return next(replies)

    monkeypatch.setattr("app.internal.book.BookIndex.search", lambda self, query: "")
    monkeypatch.setattr("app.internal.web_search.search_web", lambda query: [])
    monkeypatch.setattr("app.internal.graph.complete", fake_complete)
    result = companion.invoke(_state("What is the capital of France?"), _config())

    assert result["messages"][-1].content == "Paris is the capital of France."
    assert result["issues"] == []
    assert calls[-1]["messages"][0].content == GENERAL_PROMPT
    assert calls[-1]["max_tokens"] == 1024
