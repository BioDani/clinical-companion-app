"""Weaviate schema definition: structure, creation, and deletion."""

from __future__ import annotations

from unittest.mock import MagicMock

from app.internal.weaviate_schema import (
    BOOK_AUTHOR,
    BOOK_TITLE,
    collection_schema,
    create_collection,
    delete_collection,
    metadata_properties,
    return_properties,
    text_properties,
)


def test_collection_schema_has_vectorizer_none():
    schema = collection_schema("TestDocs")
    assert schema["class"] == "TestDocs"
    assert schema["vectorizer"] == "none"
    assert "cosine" in schema["vectorIndexConfig"]["distance"]


def test_collection_schema_includes_all_metadata_properties():
    schema = collection_schema("TestDocs")
    names = {p["name"] for p in schema["properties"]}
    assert names >= {
        "title",
        "author",
        "page_number",
        "paragraph",
        "source_doc",
        "chunk_index",
        "heading",
        "content",
    }
    type_map = {p["name"]: p["dataType"][0] for p in schema["properties"]}
    assert type_map["page_number"] == "int"
    assert type_map["paragraph"] == "int"
    assert type_map["chunk_index"] == "int"
    assert type_map["title"] == "text"
    assert type_map["content"] == "text"


def test_property_helper_functions():
    assert text_properties() == ["content"]
    meta = metadata_properties()
    assert "title" in meta
    assert "author" in meta
    assert "page_number" in meta
    assert "paragraph" in meta
    assert "content" not in meta
    all_props = return_properties()
    assert "content" in all_props
    assert all(p in all_props for p in meta)


def test_create_collection_creates_when_missing():
    client = MagicMock()
    client.schema.get.return_value = {"classes": []}

    create_collection(client, "MyDocs")
    client.schema.create_class.assert_called_once()
    schema = client.schema.create_class.call_args[0][0]
    assert schema["class"] == "MyDocs"


def test_create_collection_reuses_when_existing():
    client = MagicMock()
    client.schema.get.return_value = {"classes": [{"class": "ExistingDocs"}]}

    create_collection(client, "ExistingDocs")
    client.schema.create_class.assert_not_called()


def test_create_collection_handles_schema_error():
    client = MagicMock()
    client.schema.get.side_effect = RuntimeError("conn refused")

    create_collection(client, "MyDocs")
    client.schema.create_class.assert_called_once()


def test_delete_collection_calls_schema_delete():
    client = MagicMock()
    delete_collection(client, "ToDrop")
    client.schema.delete_class.assert_called_once_with("ToDrop")


def test_delete_collection_swallows_not_found():
    client = MagicMock()
    client.schema.delete_class.side_effect = ValueError("not found")
    delete_collection(client, "ToDrop")  # should not raise


def test_book_metadata_constants():
    assert BOOK_AUTHOR == "Dr. Carlos Jaramillo"
    assert BOOK_TITLE == "Pilares"
