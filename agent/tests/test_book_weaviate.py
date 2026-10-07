"""BookIndex retrieval via Weaviate: search, property extraction, citations."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.optional


def _mock_client(matches):
    """Build a MagicMock Weaviate client whose query returns *matches*."""
    client = MagicMock()
    builder = client.query.get.return_value
    hybrid_result = builder.with_hybrid.return_value
    with_limit_result = hybrid_result.with_limit.return_value
    with_limit_result.do.return_value = {
        "data": {"Get": {"AgenticDocuments": matches}}
    }
    return client


def _make_match(page=3, para=1, content="fiber at breakfast"):
    return {
        "title": "Pilares",
        "author": "Dr. Carlos Jaramillo",
        "page_number": page,
        "paragraph": para,
        "source_doc": "ebook.pdf",
        "content": content,
    }


def test_search_returns_passages_and_sources(monkeypatch):
    matches = [_make_match(page=3, para=5, content="Fiber at breakfast helps.")]
    client = _mock_client(matches)
    monkeypatch.setattr(
        "app.internal.embeddings.embed_query",
        lambda q: [0.1] * 1024,
    )

    from app.internal.book import BookIndex

    idx = BookIndex(client=client)
    passages, sources = idx.search("desayuno")

    assert "page 3, paragraph 5" in passages
    assert "Fiber at breakfast helps." in passages
    assert len(sources) == 1
    assert sources[0]["author"] == "Dr. Carlos Jaramillo"
    assert sources[0]["title"] == "Pilares"
    assert sources[0]["page_number"] == 3
    assert sources[0]["paragraph"] == 5


def test_search_empty_results_when_no_matches(monkeypatch):
    client = _mock_client([])
    monkeypatch.setattr(
        "app.internal.embeddings.embed_query",
        lambda q: [0.0] * 1024,
    )

    from app.internal.book import BookIndex

    passages, sources = BookIndex(client=client).search("nonexistent")
    assert passages == ""
    assert sources == []


def test_search_falls_back_when_weaviate_unavailable(monkeypatch):
    from app.internal.book import BookIndex

    def fail():
        raise ConnectionError("weaviate is down")

    client = MagicMock()
    # Simulate the entire query chain raising
    builder = client.query.get.return_value
    builder.with_hybrid.return_value.with_limit.return_value.do.side_effect = (
        ConnectionError("weaviate is down")
    )
    monkeypatch.setattr(
        "app.internal.embeddings.embed_query",
        lambda q: [0.1] * 1024,
    )

    passages, sources = BookIndex(client=client).search("anything")
    assert passages == ""
    assert sources == []


def test_search_skips_empty_content(monkeypatch):
    matches = [
        {"title": "Pilares", "author": "Dr. Carlos Jaramillo",
         "page_number": 3, "paragraph": 1, "content": "   "},
        _make_match(content="real content"),
    ]
    client = _mock_client(matches)
    monkeypatch.setattr(
        "app.internal.embeddings.embed_query",
        lambda q: [0.1] * 1024,
    )

    from app.internal.book import BookIndex

    passages, sources = BookIndex(client=client).search("test")
    assert "real content" in passages
    assert len(sources) == 1


def test_extract_properties_handles_flat_dict():
    from app.internal.book import BookIndex

    item = {"page_number": 5, "content": "text"}
    props = BookIndex._extract_properties(item)
    assert props["page_number"] == 5


def test_extract_properties_handles_nested_metadata():
    from app.internal.book import BookIndex

    item = {"metadata": {"properties": {"page_number": 7, "content": "x"}}}
    props = BookIndex._extract_properties(item)
    assert props["page_number"] == 7


def test_extract_properties_non_dict_returns_empty():
    from app.internal.book import BookIndex

    assert BookIndex._extract_properties({"properties": "not-a-dict"}) == {}
