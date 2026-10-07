"""PDF ingestion pipeline: chunking, embedding, and Weaviate storage."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.optional


class _FakeDoc:
    def __init__(self, page_content: str, page: int):
        self.page_content = page_content
        self.metadata = {"page": page}


class _FakeSplitter:
    """Minimal splitter that mimics split_text for tests."""

    def split_text(self, text: str) -> list[str]:
        parts = text.replace("\n\n", "\n").split("\n")
        result = [p.strip() for p in parts if p.strip()]
        return result if result else [text]


def _configured_client():
    """MagicMock Weaviate client with batch context-manager support."""
    client = MagicMock()
    client.schema.get.return_value = {"classes": []}
    client.batch.__enter__.return_value = client.batch
    client.batch.__exit__.return_value = False
    return client


def test_clean_delegates_to_book_clean():
    from app.internal.ingest import PdfIngestor

    ingestor = PdfIngestor.__new__(PdfIngestor)  # bypass __init__
    assert ingestor._clean("hello\nworld") == "hello world"
    assert ingestor._clean("word1-\nword2") == "word1word2"


def test_ingests_pdf_chunks_and_stores_vectors(monkeypatch):
    docs = [
        _FakeDoc(page_content="Page 1 para one.\n\nPage 1 para two.", page=0),
        _FakeDoc(page_content="Page 2 content here.", page=1),
    ]

    client = _configured_client()

    monkeypatch.setattr(
        "app.internal.embeddings.embed_texts",
        lambda texts: [[0.1, 0.2, 0.3] for _ in texts],
    )

    from app.internal.ingest import PdfIngestor

    ingestor = PdfIngestor(client=client, collection_name="TestDocs")
    ingestor._load_pdf = lambda path: docs
    ingestor._make_splitter = lambda: _FakeSplitter()

    count = ingestor.ingest_pdf("/fake/ebook.pdf")

    assert count >= 2
    assert client.batch.add_data_object.call_count >= 2
    first_call = client.batch.add_data_object.call_args_list[0]
    assert first_call.kwargs.get("vector") is not None
    assert first_call.kwargs.get("class_name") == "TestDocs"


def test_ingested_chunks_carry_rich_metadata(monkeypatch):
    docs = [_FakeDoc(page_content="Intro text for page one.", page=0)]

    client = _configured_client()
    monkeypatch.setattr(
        "app.internal.embeddings.embed_texts",
        lambda texts: [[0.0] * 768 for _ in texts],
    )

    from app.internal.ingest import PdfIngestor

    ingestor = PdfIngestor(client=client, collection_name="TestDocs")
    ingestor._load_pdf = lambda path: docs
    ingestor._make_splitter = lambda: _FakeSplitter()

    ingestor.ingest_pdf("/fake/ebook.pdf", title="My Book", author="Dr. Tester")

    call = client.batch.add_data_object.call_args_list[0]
    props = call.kwargs["data_object"]
    assert props["title"] == "My Book"
    assert props["author"] == "Dr. Tester"
    assert props["page_number"] == 1
    assert props["paragraph"] == 0
    assert props["source_doc"] == "ebook.pdf"
    assert "content" in props


def test_ingest_directory_finds_all_pdfs(tmp_path, monkeypatch):
    (tmp_path / "doc1.pdf").write_text("dummy")
    (tmp_path / "doc2.pdf").write_text("dummy")

    client = _configured_client()
    monkeypatch.setattr(
        "app.internal.embeddings.embed_texts",
        lambda texts: [[0.0] * 768 for _ in texts],
    )

    from app.internal.ingest import PdfIngestor

    ingestor = PdfIngestor(client=client, collection_name="TestDocs")
    calls: list[Path] = []
    ingestor._load_pdf = lambda path: (
        calls.append(path),
        [_FakeDoc(page_content="content for " + path.name, page=0)],
    )[1]
    ingestor._make_splitter = lambda: _FakeSplitter()

    count = ingestor.ingest_directory(str(tmp_path))

    assert count == 2
    assert len(calls) == 2
    assert calls[0].name == "doc1.pdf"
    assert calls[1].name == "doc2.pdf"


def test_reset_deletes_and_recreates():
    client = MagicMock()
    client.schema.get.side_effect = [
        {"classes": [{"class": "TestDocs"}]},
        {"classes": []},
    ]

    from app.internal.ingest import PdfIngestor

    ingestor = PdfIngestor(client=client, collection_name="TestDocs")
    ingestor.reset()

    client.schema.delete_class.assert_called_once_with("TestDocs")
    client.schema.create_class.assert_called_once()


def test_reset_handles_missing_collection():
    client = MagicMock()
    client.schema.get.return_value = {"classes": []}

    from app.internal.ingest import PdfIngestor

    ingestor = PdfIngestor(client=client, collection_name="TestDocs")
    client.schema.delete_class.side_effect = ValueError("not found")
    ingestor.reset()

    assert client.schema.delete_class.called
    assert client.schema.create_class.called


def test_ingest_pdf_empty_pdf_returns_zero(monkeypatch):
    client = _configured_client()
    monkeypatch.setattr(
        "app.internal.embeddings.embed_texts",
        lambda texts: [],
    )

    from app.internal.ingest import PdfIngestor

    ingestor = PdfIngestor(client=client, collection_name="TestDocs")
    ingestor._load_pdf = lambda path: []
    ingestor._make_splitter = lambda: _FakeSplitter()

    count = ingestor.ingest_pdf("/fake/empty.pdf")
    assert count == 0
    assert client.batch.add_data_object.call_count == 0
