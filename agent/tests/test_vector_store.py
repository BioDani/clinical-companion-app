"""Weaviate passage formatting and embedding shapes. No live services."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app.internal.book import corpus_fingerprint
from app.internal.embeddings import sentence_vectors
from app.internal.graph import format_book_reference
from app.internal.vector_store import (
    Passage,
    ensure_indexed,
    grade_coverage,
    search_hits,
    search_passages,
)


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

    assert search_hits("desayuno") == [
        Passage(text="page 3: fiber at breakfast", distance=0.2, source="", page=3)
    ]


def test_search_hits_names_the_source_file(monkeypatch: pytest.MonkeyPatch):
    _install(
        monkeypatch,
        [
            SimpleNamespace(
                properties={
                    "page": 12,
                    "text": "fiber at breakfast",
                    "source": "DGA.pdf",
                },
                metadata=SimpleNamespace(distance=0.2),
            )
        ],
    )

    assert search_hits("desayuno") == [
        Passage(
            text="DGA.pdf, page 12: fiber at breakfast",
            distance=0.2,
            source="DGA.pdf",
            page=12,
        )
    ]


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


def test_grade_coverage_ignores_words_in_the_filename():
    titled = Passage(
        text="Diet health impact.pdf, page 2: unrelated sleep chapter",
        distance=0.2,
        source="Diet health impact.pdf",
        page=2,
    )

    assert grade_coverage("What is diet health impact?", [titled]) == "barely"


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


class _Doc:
    def __init__(self, page_content: str, metadata: dict):
        self.page_content = page_content
        self.metadata = dict(metadata)


class _Loader:
    def __init__(self, path: str):
        self.path = Path(path)

    def load(self) -> list[_Doc]:
        name = self.path.name
        if name.startswith("bad"):
            raise RuntimeError(f"unreadable {name}")
        if name.startswith("empty"):
            return [_Doc("  \n ", {"page": 0})]
        return [_Doc(f"{self.path.stem} explains a good diet", {"page": 0})]


class _InsertResult:
    has_errors = False
    errors = None


class _Data:
    def __init__(self, objects: list):
        self.objects = objects

    def insert_many(self, batch: list):
        self.objects.extend(batch)
        return _InsertResult()


class _Config:
    def __init__(self):
        self.description = ""

    def get(self):
        return self

    def update(self, description: str | None = None, **kwargs):
        if description is not None:
            self.description = description


class _Aggregate:
    def __init__(self, objects: list):
        self.objects = objects

    def over_all(self, total_count: bool = True):
        return SimpleNamespace(total_count=len(self.objects))


class _Collection:
    def __init__(self):
        self.objects: list = []
        self.config = _Config()
        self.data = _Data(self.objects)
        self.aggregate = _Aggregate(self.objects)


class _Collections:
    def __init__(self):
        self.collection = _Collection()
        self.exists_flag = False
        self.deleted = 0

    def exists(self, name: str) -> bool:
        assert name == "Pilares"
        return self.exists_flag

    def delete(self, name: str) -> None:
        assert name == "Pilares"
        self.deleted += 1
        self.exists_flag = False
        self.collection = _Collection()

    def create(self, **kwargs) -> None:
        assert kwargs["name"] == "Pilares"
        self.exists_flag = True

    def use(self, name: str) -> _Collection:
        assert name == "Pilares"
        return self.collection


class _ReadyClient:
    def __init__(self, collections: _Collections):
        self.collections = collections
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _write_pdfs(folder: Path, names: list[str]) -> None:
    for name in names:
        (folder / name).write_bytes(name.encode())


def _install_index(monkeypatch: pytest.MonkeyPatch, folder: Path):
    import app.internal.vector_store as vector_store

    collections = _Collections()
    client = _ReadyClient(collections)
    embedded: list[list[str]] = []

    def embed(texts: list[str]) -> list[list[float]]:
        embedded.append(list(texts))
        return [[0.1, 0.2] for _ in texts]

    vector_store._indexed = False
    monkeypatch.setattr(vector_store, "_connect_ready", lambda: client)
    monkeypatch.setattr(vector_store, "embed_texts", embed)
    monkeypatch.setattr("app.internal.book.PyPDFLoader", _Loader)
    return collections, embedded


def _sources(collection: _Collection) -> list[str]:
    return [item.properties["source"] for item in collection.objects]


def test_corpus_fingerprint_tracks_file_bytes(tmp_path: Path):
    _write_pdfs(tmp_path, ["b.pdf", "a.pdf"])

    first = corpus_fingerprint(tmp_path)
    assert first == corpus_fingerprint(tmp_path)

    (tmp_path / "a.pdf").write_bytes(b"changed")
    assert corpus_fingerprint(tmp_path) != first


def test_index_includes_every_pdf_then_skips_and_rebuilds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import app.internal.vector_store as vector_store

    _write_pdfs(tmp_path, ["b.pdf", "a.pdf"])
    collections, embedded = _install_index(monkeypatch, tmp_path)

    ensure_indexed(tmp_path)

    assert _sources(collections.collection) == ["a.pdf", "b.pdf"]
    assert [item.properties["page"] for item in collections.collection.objects] == [1, 1]
    assert collections.collection.config.description == corpus_fingerprint(tmp_path)
    assert collections.deleted == 0
    first_embed_calls = len(embedded)
    assert first_embed_calls == 2

    vector_store._indexed = False
    ensure_indexed(tmp_path)

    assert len(embedded) == first_embed_calls
    assert collections.deleted == 0
    assert _sources(collections.collection) == ["a.pdf", "b.pdf"]

    (tmp_path / "a.pdf").write_bytes(b"revised a")
    vector_store._indexed = False
    ensure_indexed(tmp_path)

    assert collections.deleted == 1
    assert _sources(collections.collection) == ["a.pdf", "b.pdf"]
    assert collections.collection.config.description == corpus_fingerprint(tmp_path)
    assert len(embedded) == first_embed_calls * 2


def test_index_skips_a_pdf_with_no_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    _write_pdfs(tmp_path, ["empty.pdf", "a.pdf"])
    collections, _embedded = _install_index(monkeypatch, tmp_path)

    with caplog.at_level("WARNING"):
        ensure_indexed(tmp_path)

    assert _sources(collections.collection) == ["a.pdf"]
    assert "No extractable text in empty.pdf" in caplog.text


def test_unreadable_pdf_drops_a_partial_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import app.internal.vector_store as vector_store

    _write_pdfs(tmp_path, ["a.pdf", "bad.pdf"])
    collections, _embedded = _install_index(monkeypatch, tmp_path)

    with pytest.raises(RuntimeError, match="unreadable"):
        ensure_indexed(tmp_path)

    assert collections.deleted == 1
    assert collections.exists_flag is False
    assert vector_store._indexed is False


def test_reference_groups_pages_by_source_file():
    hits = [
        Passage(
            text="DGA.pdf, page 12: vegetables",
            distance=0.1,
            source="DGA.pdf",
            page=12,
        ),
        Passage(
            text="DGA.pdf, page 14: fruit",
            distance=0.2,
            source="DGA.pdf",
            page=14,
        ),
        Passage(
            text="WHO_TRS_916.pdf, page 4: diet",
            distance=0.2,
            source="WHO_TRS_916.pdf",
            page=4,
        ),
    ]

    assert format_book_reference("\n\n".join(hit.text for hit in hits), hits) == (
        "### Reference\n\n"
        "- DGA.pdf, page 12, page 14\n"
        "- WHO_TRS_916.pdf, page 4"
    )
