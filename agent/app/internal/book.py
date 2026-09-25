"""Lazy search over the clinical companion ebook."""

from __future__ import annotations

import re
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

_EBOOK = (
    Path(__file__).resolve().parents[2]
    / "knowledge"
    / "ebook_pilares_DrCarlosJaramillo_V2.pdf"
)
_TOKEN = re.compile(r"\w+", re.UNICODE)
_LINE_HYPHEN = re.compile(r"(\w)\s*-\s*\n\s*(\w)")
_LINE_BREAK = re.compile(r"\s*\n\s*")


def _tokens(text: str) -> set[str]:
    return {word for word in _TOKEN.findall(text.lower()) if len(word) > 2}


class BookIndex:
    """Load the ebook on first search and return the closest page-tagged passages."""

    def __init__(self, path: Path | None = None):
        self.path = path or _EBOOK
        self._chunks: list | None = None

    def search(self, query: str) -> str:
        scored: list[tuple[int, object]] = []
        query_tokens = _tokens(query)
        for chunk in self._load():
            overlap = len(query_tokens & _tokens(chunk.page_content))
            if overlap:
                scored.append((overlap, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        passages = [
            _format_passage(chunk) for _, chunk in scored[:4]
        ]
        return "\n\n".join(passages)

    def _load(self) -> list:
        if self._chunks is None:
            if not self.path.is_file():
                raise FileNotFoundError(f"Ebook not found: {self.path}")
            docs = PyPDFLoader(str(self.path)).load()
            for doc in docs:
                doc.page_content = _clean_text(doc.page_content)
            splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=256)
            self._chunks = splitter.split_documents(docs)
        return self._chunks


def _clean_text(text: str) -> str:
    text = _LINE_HYPHEN.sub(r"\1\2", text)
    return _LINE_BREAK.sub(" ", text).strip()


def _format_passage(chunk) -> str:
    page = chunk.metadata.get("page")
    label = int(page) + 1 if isinstance(page, int) else "?"
    return f"page {label}: {chunk.page_content.strip()}"
