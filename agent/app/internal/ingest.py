"""PDF ingestion pipeline: load → clean → chunk → embed → store in Weaviate.

The pipeline reads PDF documents, splits them into overlapping chunks,
embeds each chunk via Hugging Face Inference API (same stack as smolagents),
and stores them in Weaviate with rich metadata for cited retrieval.

Usage (CLI)::

    python -m app.scripts.ingest_docs --knowledge-dir /app/knowledge --reset

The schema with author, title, page_number, paragraph, etc. is defined in
:mod:`app.internal.weaviate_schema`.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class PdfIngestor:
    """Ingest PDF documents into a Weaviate collection with rich metadata."""

    def __init__(self, client=None, collection_name=None):
        from ..config import weaviate_collection
        from .weaviate_schema import BOOK_AUTHOR, BOOK_TITLE, create_collection

        self._client = client
        self.collection_name = collection_name or weaviate_collection()
        self._book_title = BOOK_TITLE
        self._book_author = BOOK_AUTHOR
        self._ensure_schema()

    def _ensure_schema(self):
        """Create the Weaviate collection if it does not yet exist."""
        from .weaviate_schema import create_collection

        create_collection(self._client_or_default, self.collection_name)

    @property
    def _client_or_default(self):
        if self._client is not None:
            return self._client
        from .weaviate_client import get_weaviate_client

        return get_weaviate_client()

    def _load_pdf(self, path: Path):
        """Load documents from a PDF file.  Override in tests."""
        from langchain_community.document_loaders import PyPDFLoader

        return PyPDFLoader(str(path)).load()

    def _make_splitter(self):
        """Create a text splitter.  Override in tests."""
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        return RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=256)

    def reset(self) -> None:
        """Delete and recreate the collection (clears all ingested data)."""
        from .weaviate_schema import create_collection, delete_collection

        client = self._client_or_default
        delete_collection(client, self.collection_name)
        create_collection(client, self.collection_name)
        logger.info("Reset collection '%s'.", self.collection_name)

    def ingest_pdf(
        self,
        pdf_path: str | Path,
        title: str | None = None,
        author: str | None = None,
    ) -> int:
        """Ingest a single PDF.  Returns the number of chunks stored."""
        from .book import _clean_text

        path = Path(pdf_path)
        doc_title = title or path.stem
        doc_author = author or self._book_author

        logger.info("Ingesting %s (title=%s, author=%s)", path.name, doc_title, doc_author)

        docs = self._load_pdf(path)
        splitter = self._make_splitter()

        texts: list[str] = []
        metadatas: list[dict] = []
        chunk_idx = 0
        for doc in docs:
            page_num = int(doc.metadata.get("page", 0)) + 1
            cleaned = _clean_text(doc.page_content)
            chunks = splitter.split_text(cleaned)
            for para_idx, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue
                texts.append(chunk)
                metadatas.append(
                    {
                        "title": doc_title,
                        "author": doc_author,
                        "page_number": page_num,
                        "paragraph": para_idx,
                        "source_doc": path.name,
                        "chunk_index": chunk_idx,
                        "content": chunk,
                        "heading": "",
                    }
                )
                chunk_idx += 1

        if not texts:
            logger.warning("No chunks extracted from %s", path)
            return 0

        from .embeddings import embed_texts

        vectors = embed_texts(texts)
        self._batch_insert(metadatas, vectors)
        logger.info("Stored %d chunks into '%s'.", len(texts), self.collection_name)
        return len(texts)

    def ingest_directory(
        self,
        directory: str | Path,
        title: str | None = None,
        author: str | None = None,
    ) -> int:
        """Ingest every ``*.pdf`` in *directory*.  Returns total chunk count."""
        dir_path = Path(directory)
        total = 0
        for pdf_file in sorted(dir_path.glob("*.pdf")):
            total += self.ingest_pdf(pdf_file, title=title, author=author)
        return total

    def _batch_insert(self, metadatas: list[dict], vectors: list[list[float]]) -> None:
        """Insert chunks into Weaviate using a batch request."""
        client = self._client_or_default
        batch_size = 100
        for start in range(0, len(metadatas), batch_size):
            end = min(start + batch_size, len(metadatas))
            batch = client.batch
            batch.batch_size = end - start
            with batch as b:
                for metadata, vector in zip(
                    metadatas[start:end],
                    vectors[start:end],
                ):
                    properties = dict(metadata)
                    b.add_data_object(
                        data_object=properties,
                        class_name=self.collection_name,
                        vector=list(vector),
                    )