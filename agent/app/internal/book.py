"""Load every PDF in the knowledge folder for the vector index."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

_KNOWLEDGE = Path(__file__).resolve().parents[2] / "knowledge"
_TOKEN = re.compile(r"\w+", re.UNICODE)
_LINE_HYPHEN = re.compile(r"(\w)\s*-\s*\n\s*(\w)")
_LINE_BREAK = re.compile(r"\s*\n\s*")
_CHUNK_SIZE = 1024
_CHUNK_OVERLAP = 256


def _tokens(text: str) -> set[str]:
    return {word for word in _TOKEN.findall(text.lower()) if len(word) > 2}


def knowledge_pdfs(root: Path | None = None) -> list[Path]:
    """Return the knowledge PDFs in filename order."""
    folder = root or _KNOWLEDGE
    if not folder.is_dir():
        raise FileNotFoundError(f"Knowledge folder not found: {folder}")
    pdfs = sorted(path for path in folder.glob("*.pdf") if path.is_file())
    if not pdfs:
        raise FileNotFoundError(f"No PDFs in {folder}")
    return pdfs


def corpus_fingerprint(root: Path | None = None) -> str:
    """Hash each knowledge PDF so an unchanged folder can skip ingest."""
    digest = hashlib.sha256()
    for path in knowledge_pdfs(root):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def load_pdf_chunks(path: Path) -> list:
    """Split one PDF into cleaned chunks. An unreadable PDF raises."""
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")
    docs = PyPDFLoader(str(path)).load()
    cleaned = []
    for doc in docs:
        doc.page_content = _clean_text(doc.page_content)
        if doc.page_content:
            cleaned.append(doc)
    if not cleaned:
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_CHUNK_SIZE,
        chunk_overlap=_CHUNK_OVERLAP,
    )
    chunks = []
    for chunk in splitter.split_documents(cleaned):
        text = (chunk.page_content or "").strip()
        if not text:
            continue
        chunk.page_content = text
        chunks.append(chunk)
    return chunks


def _clean_text(text: str) -> str:
    text = _LINE_HYPHEN.sub(r"\1\2", text)
    return _LINE_BREAK.sub(" ", text).strip()
