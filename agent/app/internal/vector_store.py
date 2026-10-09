"""Weaviate index of every PDF in the knowledge folder."""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import weaviate
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.data import DataObject
from weaviate.classes.query import MetadataQuery

from ..config import weaviate_host
from .book import _tokens, corpus_fingerprint, knowledge_pdfs, load_pdf_chunks
from .embeddings import embed_texts

logger = logging.getLogger(__name__)

COLLECTION_NAME = "Pilares"
RETRIEVAL_LIMIT = 4
EXPLAINED_OVERLAP = 3
FAR_DISTANCE = 0.55
_HTTP_PORT = 8080
_GRPC_PORT = 50051
_INSERT_BATCH = 16
_CONNECT_ATTEMPTS = 30
_CONNECT_DELAY_SECONDS = 2.0

_index_lock = threading.Lock()
_indexed = False


def indexed() -> bool:
    """True after ensure_indexed has matched or built the corpus."""
    return _indexed
_PASSAGE_PREFIX = re.compile(r"^(?:.+?, )?page [^:]+:\s*")


@dataclass(frozen=True)
class Passage:
    """One page-tagged knowledge passage and its vector distance."""

    text: str
    distance: float | None
    source: str = ""
    page: int = 0


def search_hits(query: str, limit: int = RETRIEVAL_LIMIT) -> list[Passage]:
    """Embed the query and return up to four passages with distance."""
    ensure_indexed()
    text = (query or "").strip()
    if not text:
        return []
    vector = embed_texts([text])[0]
    with _connect() as client:
        collection = client.collections.use(COLLECTION_NAME)
        results = collection.query.near_vector(
            near_vector=vector,
            limit=limit,
            return_metadata=MetadataQuery(distance=True),
        )
    hits: list[Passage] = []
    for obj in results.objects:
        properties = getattr(obj, "properties", None) or {}
        formatted = _format_hit(properties)
        if not formatted:
            continue
        hits.append(
            Passage(
                text=formatted,
                distance=_distance(obj),
                source=_source(properties),
                page=_page(properties),
            )
        )
    return hits


def search_passages(query: str, limit: int = RETRIEVAL_LIMIT) -> str:
    """Embed the query and return up to four page-tagged passages."""
    return "\n\n".join(hit.text for hit in search_hits(query, limit))


def grade_coverage(query: str, hits: list[Passage]) -> str:
    """Grade knowledge coverage as explained, barely, or miss.

    Explained means a near passage shares several content words with the
    question. A far neighbor, or a passage that only shares a word or two,
    is barely. No passages, or a far neighbor with no shared words, is a miss.
    """
    if not hits:
        return "miss"
    best = max(_overlap(query, _passage_body(hit.text)) for hit in hits)
    distances = [hit.distance for hit in hits if hit.distance is not None]
    nearest = min(distances) if distances else None
    if nearest is not None and nearest > FAR_DISTANCE:
        return "miss" if best == 0 else "barely"
    if best >= EXPLAINED_OVERLAP:
        return "explained"
    return "barely"


def _passage_body(text: str) -> str:
    """Drop the source and page label so a filename cannot count as coverage."""
    return _PASSAGE_PREFIX.sub("", text or "", count=1)


def _overlap(query: str, passage: str) -> int:
    return len(_tokens(query) & _tokens(passage))


def _distance(obj: object) -> float | None:
    metadata = getattr(obj, "metadata", None)
    if metadata is None:
        return None
    distance = getattr(metadata, "distance", None)
    if distance is None:
        return None
    return float(distance)


def ensure_indexed(root: Path | None = None) -> None:
    """Create Pilares and insert every knowledge PDF when the corpus changed."""
    global _indexed
    with _index_lock:
        if _indexed:
            return
        fingerprint = corpus_fingerprint(root)
        client = _connect_ready()
        try:
            if _index_is_current(client, fingerprint):
                logger.info(
                    "Pilares collection matches the knowledge corpus; skipping ingest"
                )
                _indexed = True
                return
            if client.collections.exists(COLLECTION_NAME):
                logger.info("Rebuilding Pilares collection for a new knowledge corpus")
                client.collections.delete(COLLECTION_NAME)
            _ensure_collection(client)
            collection = client.collections.use(COLLECTION_NAME)
            try:
                _insert_corpus(collection, root)
                collection.config.update(description=fingerprint)
            except Exception:
                if client.collections.exists(COLLECTION_NAME):
                    client.collections.delete(COLLECTION_NAME)
                raise
            _indexed = True
        finally:
            client.close()


def _index_is_current(client, fingerprint: str) -> bool:
    if not client.collections.exists(COLLECTION_NAME):
        return False
    collection = client.collections.use(COLLECTION_NAME)
    if _collection_description(collection) != fingerprint:
        return False
    return _object_count(collection) > 0


def _collection_description(collection) -> str:
    config = collection.config.get()
    return str(getattr(config, "description", "") or "").strip()


def _connect():
    return weaviate.connect_to_local(
        host=weaviate_host(),
        port=_HTTP_PORT,
        grpc_port=_GRPC_PORT,
    )


def _connect_ready():
    host = weaviate_host()
    last_error: Exception | None = None
    for attempt in range(1, _CONNECT_ATTEMPTS + 1):
        client = None
        try:
            client = _connect()
            if client.is_ready():
                return client
            last_error = ConnectionError(f"Weaviate at {host} is not ready")
        except Exception as exc:
            last_error = exc
        if client is not None:
            client.close()
        logger.warning(
            "Weaviate not ready at %s (attempt %s/%s): %s",
            host,
            attempt,
            _CONNECT_ATTEMPTS,
            last_error,
        )
        if attempt < _CONNECT_ATTEMPTS:
            time.sleep(_CONNECT_DELAY_SECONDS)
    raise ConnectionError(f"Could not connect to Weaviate at {host}") from last_error


def _ensure_collection(client) -> None:
    if client.collections.exists(COLLECTION_NAME):
        return
    client.collections.create(
        name=COLLECTION_NAME,
        description="",
        vector_config=Configure.Vectors.self_provided(),
        properties=[
            Property(name="text", data_type=DataType.TEXT),
            Property(name="source", data_type=DataType.TEXT),
            Property(name="document_id", data_type=DataType.TEXT),
            Property(name="page", data_type=DataType.INT),
            Property(name="chunk_index", data_type=DataType.INT),
        ],
    )


def _object_count(collection) -> int:
    result = collection.aggregate.over_all(total_count=True)
    return int(result.total_count or 0)


def _insert_corpus(collection, root: Path | None = None) -> None:
    pdfs = knowledge_pdfs(root)
    total = 0
    for number, path in enumerate(pdfs, start=1):
        chunks = load_pdf_chunks(path)
        if not chunks:
            logger.warning("No extractable text in %s; skipping", path.name)
            continue
        prepared = []
        for index, chunk in enumerate(chunks):
            page = chunk.metadata.get("page")
            prepared.append(
                {
                    "text": chunk.page_content,
                    "source": path.name,
                    "document_id": path.stem,
                    "page": int(page) + 1 if isinstance(page, int) else 0,
                    "chunk_index": index,
                }
            )
        logger.info(
            "Indexing %s/%s %s (%s chunks)",
            number,
            len(pdfs),
            path.name,
            len(prepared),
        )
        _insert_batch(collection, prepared)
        total += len(prepared)
    if total == 0:
        raise RuntimeError("Knowledge PDFs produced no text chunks")
    logger.info(
        "Indexed %s chunks from %s PDFs into %s",
        total,
        len(pdfs),
        COLLECTION_NAME,
    )


def _insert_batch(collection, prepared: list[dict]) -> None:
    for start in range(0, len(prepared), _INSERT_BATCH):
        batch = prepared[start : start + _INSERT_BATCH]
        vectors = embed_texts([item["text"] for item in batch])
        if len(vectors) != len(batch):
            raise RuntimeError("Embedding batch size did not match the chunk batch")
        result = collection.data.insert_many(
            [
                DataObject(properties=item, vector=vector)
                for item, vector in zip(batch, vectors)
            ]
        )
        if result.has_errors:
            raise RuntimeError(f"Weaviate insert failed: {result.errors}")


def _source(properties: dict) -> str:
    return str(properties.get("source") or "").strip()


def _page(properties: dict) -> int:
    page = properties.get("page")
    if isinstance(page, int) and page > 0:
        return page
    return 0


def _format_hit(properties: dict) -> str:
    text = str(properties.get("text") or "").strip()
    if not text:
        return ""
    page = properties.get("page")
    label = page if isinstance(page, int) and page > 0 else "?"
    source = _source(properties)
    if source:
        return f"{source}, page {label}: {text}"
    return f"page {label}: {text}"
