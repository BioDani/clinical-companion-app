"""Weaviate passage formatting and embedding shapes. No live services."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from app.internal.embeddings import sentence_vectors
from app.internal.vector_store import Passage, grade_coverage, search_hits, search_passages


class _Query:
    def __init__(self, objects: list):
        self.objects = objects
        self.calls: list[dict] = []

    def near_vector(self, *, near_vector, limit, return_metadata=None):
        self.calls.append(
            {
                "near_vector": near_vector,
                "limit": limit,
                "distance": getattr(return_metadata, "distance", None),
            }
        )
        return SimpleNamespace(objects=self.objects)


class _Client:
    def __init__(self, query: _Query):
        self.query = query
        self.collections = self

    def use(self, name: str):
        assert name == "Pilares"
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _install(monkeypatch: pytest.MonkeyPatch, objects: list):
    query = _Query(objects)
    seen: list[list[str]] = []

    def embed(texts: list[str]) -> list[list[float]]:
        seen.append(list(texts))
        return [[0.1, 0.2]]

    monkeypatch.setattr("app.internal.vector_store.ensure_indexed", lambda: None)
    monkeypatch.setattr("app.internal.vector_store.embed_texts", embed)
    monkeypatch.setattr("app.internal.vector_store._connect", lambda: _Client(query))
    return query, seen


def test_search_passages_formats_pages(monkeypatch: pytest.MonkeyPatch):
    query, seen = _install(
        monkeypatch,
        [
            SimpleNamespace(properties={"page": 3, "text": "fiber at breakfast"}),
            SimpleNamespace(properties={"page": 0, "text": "unknown page"}),
            SimpleNamespace(properties={"page": 4, "text": "  "}),
        ],
    )

    assert search_passages("desayuno") == "page 3: fiber at breakfast\n\npage ?: unknown page"
    assert seen == [["desayuno"]]
    assert query.calls == [{"near_vector": [0.1, 0.2], "limit": 4, "distance": True}]


def test_search_passages_empty_result_is_blank(monkeypatch: pytest.MonkeyPatch):
    _install(monkeypatch, [])

    assert search_passages("desayuno") == ""


def test_blank_query_does_not_embed(monkeypatch: pytest.MonkeyPatch):
    def fail(texts):
        raise AssertionError("should not embed")

    monkeypatch.setattr("app.internal.vector_store.ensure_indexed", lambda: None)
    monkeypatch.setattr("app.internal.vector_store.embed_texts", fail)

    assert search_passages("  ") == ""


def test_search_hits_keeps_distance(monkeypatch: pytest.MonkeyPatch):
    _install(
        monkeypatch,
        [
            SimpleNamespace(
                properties={"page": 3, "text": "fiber at breakfast"},
                metadata=SimpleNamespace(distance=0.2),
            )
        ],
    )

    assert search_hits("desayuno") == [Passage(text="page 3: fiber at breakfast", distance=0.2)]


def test_grade_coverage_uses_overlap_and_distance():
    explained = Passage(text="page 1: what a good diet includes", distance=0.2)
    far = Passage(text="page 1: what a good diet includes", distance=0.8)
    thin = Passage(text="page 8: breakfast routines", distance=0.2)
    unrelated = Passage(text="page 9: unrelated sleep chapter", distance=0.9)

    assert grade_coverage("What is a good diet?", [explained]) == "explained"
    assert grade_coverage("What is a good diet?", [far]) == "barely"
    assert grade_coverage("What is a good breakfast?", [thin]) == "barely"
    assert grade_coverage("What is a good diet?", [unrelated]) == "miss"
    assert grade_coverage("What is a good diet?", []) == "miss"


def test_sentence_vectors_mean_pool_token_matrices():
    pooled = sentence_vectors(np.array([[1.0, 3.0], [3.0, 5.0]]), 1)
    assert pooled == [[2.0, 4.0]]

    batch = sentence_vectors(
        [
            np.array([[0.0, 2.0], [2.0, 4.0]]),
            np.array([1.0, 1.0]),
        ],
        2,
    )
    assert batch == [[1.0, 3.0], [1.0, 1.0]]

    wrapped = sentence_vectors(np.array([[[1.0, 3.0], [3.0, 5.0]]]), 1)
    assert wrapped == [[2.0, 4.0]]
