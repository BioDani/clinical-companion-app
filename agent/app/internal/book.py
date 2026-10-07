"""Lazy retrieval over clinical companion documents via Weaviate vector store.

The :class:`BookIndex` performs a **hybrid search** (semantic vector similarity
+ BM25 keyword) against a Weaviate collection that was populated by
:mod:`app.internal.ingest`.  Results carry rich metadata — author, title,
page number, paragraph — enabling precise, cited retrieval.

The text-cleaning helpers (``_clean_text``, ``_EBOOK``) are retained and
re-used by the ingestion pipeline so that stored passages are normalised.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_BOOK_AUTHOR = "Dr. Carlos Jaramillo"
_BOOK_TITLE = "Pilares"

_EBOOK = (
    Path(__file__).resolve().parents[2]
    / "knowledge"
    / "ebook_pilares_DrCarlosJaramillo_V2.pdf"
)

_TOKEN = re.compile(r"\w+", re.UNICODE)
_LINE_HYPHEN = re.compile(r"(\w)\s*-\s*\n\s*(\w)")
_LINE_BREAK = re.compile(r"\s*\n\s*")


def _clean_text(text: str) -> str:
    """Normalise PDF text: join hyphenated line breaks and collapse whitespace."""
    text = _LINE_HYPHEN.sub(r"\1\2", text)
    return _LINE_BREAK.sub(" ", text).strip()


def _tokens(text: str) -> set[str]:
    """Lowercased token set for simple keyword fallback (kept for diagnostics)."""
    return {word for word in _TOKEN.findall(text.lower()) if len(word) > 2}


class BookIndex:
    """Retrieve passages from Weaviate with rich citation metadata.

    On ``search`` the query is embedded via Hugging Face (same stack as
    smolagents), then a hybrid search combines vector similarity and BM25
    keyword matching.  Up to *k* passages are returned.

    Returns
    -------
    tuple[str, list[dict]]
        ``(passages, sources)`` where *passages* is a newline-joined string
        of ``"page N, paragraph P: <text>"`` snippets ready for the LLM
        prompt, and *sources* is a list of metadata dicts used for citation.
    """

    def __init__(self, client=None, collection_name=None):
        from ..config import weaviate_collection

        self._client = client  # None = use the lazy default client
        self.collection_name = collection_name or weaviate_collection()

    @property
    def _client_or_default(self):
        if self._client is not None:
            return self._client
        from .weaviate_client import get_weaviate_client

        return get_weaviate_client()

    def search(self, query: str, k: int = 4) -> tuple[str, list[dict]]:
        """Hybrid search the Weaviate collection for *query*.

        Returns ``(passages, sources)``.  Both are empty strings/lists
        when no matches are found or when Weaviate is unavailable — the
        graph's grading step treats an empty result as a miss without
        invoking the model, gracefully falling back to web search.
        """
        from .embeddings import embed_query
        from .weaviate_schema import return_properties

        try:
            client = self._client_or_default
            query_vector = embed_query(query)
            results = (
                client.query.get(
                    self.collection_name,
                    return_properties(),
                )
                .with_hybrid(
                    query=query,
                    vector=query_vector,
                    alpha=0.7,
                )
                .with_limit(k)
                .do()
            )
            matches = (
                results.get("data", {})
                .get("Get", {})
                .get(self.collection_name, [])
            )
        except Exception as exc:
            logger.warning("Weaviate search failed; falling back: %s", exc)
            return "", []

        if not matches:
            return "", []

        passages: list[str] = []
        sources: list[dict] = []
        for item in matches:
            props = self._extract_properties(item)
            content = (props.get("content") or "").strip()
            if not content:
                continue
            page = props.get("page_number") or 0
            para = props.get("paragraph") or 0
            passages.append(
                f"page {page}, paragraph {para}: {content}"
            )
            sources.append(
                {
                    "author": props.get("author", _BOOK_AUTHOR),
                    "title": props.get("title", _BOOK_TITLE),
                    "page_number": page,
                    "paragraph": para,
                    "source_doc": props.get("source_doc", ""),
                    "content": content,
                }
            )

        return "\n\n".join(passages), sources

    @staticmethod
    def _extract_properties(item: dict) -> dict:
        """Flatten a Weaviate result object into a properties dict."""
        if "properties" in item:
            props = item["properties"]
        elif "metadata" in item and "properties" in item["metadata"]:
            props = item["metadata"]["properties"]
        else:
            props = item
        return props if isinstance(props, dict) else {}


def format_book_reference(sources: list[dict]) -> str:
    """Build a citation block from structured retrieval metadata."""
    if not sources:
        return ""
    lines: list[str] = []
    seen: set[tuple] = set()
    for src in sources:
        key = (src.get("page_number"), src.get("paragraph"))
        if key in seen:
            continue
        seen.add(key)
        author = src.get("author", _BOOK_AUTHOR)
        title = src.get("title", _BOOK_TITLE)
        page = src.get("page_number", "?")
        para = src.get("paragraph")
        if para is not None:
            lines.append(
                f"- {author}, _{title}_, page {page}, paragraph {para}"
            )
        else:
            lines.append(f"- {author}, _{title}_, page {page}")
    return "### Sources\n\n" + "\n".join(lines)