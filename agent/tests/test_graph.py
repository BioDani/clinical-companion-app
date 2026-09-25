"""Router graph: guardrail, book-backed answers, and BMI."""

from __future__ import annotations

import uuid

import pytest
from langchain_core.messages import HumanMessage

from app.internal.graph import ANSWER_PROMPT, REFUSAL, build_graph, companion

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


def test_compiled_graph_routes_through_guard_and_answer():
    drawn = build_graph().get_graph()
    nodes = set(drawn.nodes)
    edges = {(edge.source, edge.target) for edge in drawn.edges}
    assert {"guard", "plan", "search", "grade", "answer", "Dangerous"} <= nodes
    assert ("__start__", "guard") in edges
    assert ("plan", "search") in edges
    assert ("search", "grade") in edges
    assert ("grade", "answer") in edges
    assert ("grade", "plan") in edges
    assert ("answer", "__end__") in edges
    assert ("Dangerous", "__end__") in edges


def test_diagnosis_is_refused_without_calling_the_model(monkeypatch: pytest.MonkeyPatch):
    def fail(messages):
        raise AssertionError("model should not be called")

    monkeypatch.setattr("app.internal.graph.complete", fail)
    result = companion.invoke(_state("Do I have diabetes?"), _config())

    assert result["messages"][-1].content == REFUSAL
    assert result["issues"] == ["Do I have diabetes?"]
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

    assert queries == ["breakfast fiber"]
    assert result["retrieved"] == "page 3: fiber at breakfast"
    assert calls[-1]["max_tokens"] == 1024
    assert calls[-1]["messages"][0].content == ANSWER_PROMPT
    assert calls[-1]["messages"][-1].content == "Retrieved context:\npage 3: fiber at breakfast"
    assert result["messages"][-1].content == "a quoted answer"
    assert result["issues"] == []


def test_revise_searches_the_book_a_second_time(monkeypatch: pytest.MonkeyPatch):
    queries: list[str] = []
    replies = iter(
        ["breakfast fiber", "REVISE", "morning meal habits", "SUFFICIENT", "elaborated"]
    )

    monkeypatch.setattr(
        "app.internal.book.BookIndex.search",
        lambda self, query: queries.append(query) or f"page {len(queries)}: {query}",
    )
    monkeypatch.setattr(
        "app.internal.graph.complete",
        lambda messages, max_tokens=512: next(replies),
    )
    result = companion.invoke(_state("What is a good diet?"), _config())

    assert queries == ["breakfast fiber", "morning meal habits"]
    assert "page 1: breakfast fiber" in result["retrieved"]
    assert "page 2: morning meal habits" in result["retrieved"]
    assert result["searches"] == 2
    assert result["messages"][-1].content == "elaborated"


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
    monkeypatch.setattr(
        "app.internal.graph.complete",
        lambda messages, max_tokens=512: next(replies),
    )
    config = _config()
    first = companion.invoke(_state("What is a good diet?"), config)
    second = companion.invoke(_state("How should I exercise?"), config)

    assert first["retrieved"] == "page 1: desayuno"
    assert second["retrieved"] == "page 1: ejercicio"
    assert second["searches"] == 1


def test_height_and_weight_include_bmi_with_the_book(monkeypatch: pytest.MonkeyPatch):
    replies = iter(["movement habits", "SUFFICIENT", "grounded"])
    monkeypatch.setattr(
        "app.internal.book.BookIndex.search",
        lambda self, query: "page 4: daily movement",
    )
    monkeypatch.setattr(
        "app.internal.graph.complete",
        lambda messages, max_tokens=512: next(replies),
    )
    result = companion.invoke(
        _state("I weigh 82 kg and I am 1.78 m. What habits help?"),
        _config(),
    )

    assert result["retrieved"].startswith("BMI 25.9 (overweight)")
    assert "page 4: daily movement" in result["retrieved"]
    assert result["messages"][-1].content == "grounded"


def test_smalltalk_and_end_follow_the_model_label(monkeypatch: pytest.MonkeyPatch):
    replies = iter(["SMALLTALK", "Hello from companion."])
    monkeypatch.setattr("app.internal.graph.complete", lambda messages: next(replies))
    smalltalk = companion.invoke(_state("hello"), _config())
    assert smalltalk["messages"][-1].content == "Hello from companion."
    assert smalltalk["retrieved"] == ""

    monkeypatch.setattr("app.internal.graph.complete", lambda messages: "END")
    ended = companion.invoke(_state("What is the capital of France?"), _config())
    assert ended["messages"][-1].content == "END"
    assert ended["retrieved"] == ""


def test_unknown_model_label_is_treated_as_dangerous(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("app.internal.graph.complete", lambda messages: "maybe")
    result = companion.invoke(_state("What is the capital of France?"), _config())
    assert result["messages"][-1].content == REFUSAL
    assert result["issues"] == ["What is the capital of France?"]
