"""Weaviate collection schema with rich metadata for cited retrieval.

Properties stored per chunk:
  - title:       Document or section title
  - author:      Document author
  - page_number: 1-based page number from the PDF
  - paragraph:   Paragraph index within the page
  - source_doc:  Source PDF filename
  - chunk_index: Global chunk index for ordering
  - heading:     Optional section heading context
  - content:     The text chunk (indexed for BM25 keyword search)
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

BOOK_TITLE = "Pilares"
BOOK_AUTHOR = "Dr. Carlos Jaramillo"

PROPERTIES = [
    {
        "name": "title",
        "data_type": "text",
        "description": "Document or section title",
    },
    {
        "name": "author",
        "data_type": "text",
        "description": "Document author",
    },
    {
        "name": "page_number",
        "data_type": "int",
        "description": "1-based page number from the PDF",
    },
    {
        "name": "paragraph",
        "data_type": "int",
        "description": "Paragraph index within the page",
    },
    {
        "name": "source_doc",
        "data_type": "text",
        "description": "Source PDF filename",
    },
    {
        "name": "chunk_index",
        "data_type": "int",
        "description": "Global chunk index for ordering",
    },
    {
        "name": "heading",
        "data_type": "text",
        "description": "Optional section heading context",
    },
    {
        "name": "content",
        "data_type": "text",
        "description": "The text chunk for BM25 keyword search",
    },
]

VECTOR_SIZE = 1024


def _convert_property(prop: dict) -> dict:
    """Convert a property dict to the Weaviate REST schema format."""
    return {
        "name": prop["name"],
        "dataType": [prop["data_type"]],
        "description": prop.get("description", ""),
        "indexSearchable": True,
        "indexFilterable": True,
    }


def collection_schema(collection_name: str, vector_size: int = VECTOR_SIZE) -> dict:
    """Return the Weaviate REST schema for the document collection."""
    return {
        "class": collection_name,
        "description": (
            "Clinical Companion knowledge base with rich metadata "
            "for precise, cited retrieval from uploaded PDF documents."
        ),
        "vectorizer": "none",
        "vectorIndexConfig": {
            "distance": "cosine",
            "dimensions": vector_size,
        },
        "properties": [_convert_property(p) for p in PROPERTIES],
    }


def text_properties() -> list[str]:
    """Property names that hold free-text content for BM25 search."""
    return ["content"]


def metadata_properties() -> list[str]:
    """Property names carrying citation-relevant metadata."""
    return [
        "title",
        "author",
        "page_number",
        "paragraph",
        "source_doc",
        "chunk_index",
        "heading",
    ]


def return_properties() -> list[str]:
    """All properties to request in a retrieval query."""
    return text_properties() + metadata_properties()


def create_collection(client, collection_name: str, vector_size: int = VECTOR_SIZE):
    """Create the collection on a Weaviate client if it does not already exist."""
    existing = _list_collections(client)
    if collection_name in existing:
        logger.info("Collection '%s' already exists; reusing it.", collection_name)
        return
    schema = collection_schema(collection_name, vector_size)
    client.schema.create_class(schema)
    logger.info("Created collection '%s' with %d properties.", collection_name, len(PROPERTIES))


def delete_collection(client, collection_name: str):
    """Delete the collection (used for re-ingestion)."""
    try:
        client.schema.delete_class(collection_name)
        logger.info("Deleted collection '%s'.", collection_name)
    except Exception:
        logger.debug("Collection '%s' did not exist; nothing to delete.", collection_name)


def _list_collections(client) -> list[str]:
    """Return the list of collection (class) names on the server."""
    try:
        schema = client.schema.get()
    except Exception:
        return []
    classes = schema.get("classes") if isinstance(schema, dict) else None
    if not classes:
        return []
    return [c["class"] for c in classes if isinstance(c, dict) and "class" in c]