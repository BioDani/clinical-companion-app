#!/usr/bin/env python3
"""Ingest PDF documents into the Weaviate vector store.

Usage::

    python -m app.scripts.ingest_docs --knowledge-dir /app/knowledge
    python -m app.scripts.ingest_docs --pdf /app/knowledge/book.pdf --reset
"""

from __future__ import annotations

import argparse
import logging
import sys

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest PDF documents into Weaviate with rich metadata.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pdf", help="Path to a single PDF file to ingest.")
    group.add_argument(
        "--knowledge-dir",
        help="Directory containing PDF files to ingest (all *.pdf).",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop and recreate the collection before ingesting.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Override document title for all ingested chunks.",
    )
    parser.add_argument(
        "--author",
        default=None,
        help="Override document author for all ingested chunks.",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Override Weaviate collection name.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = _build_parser()
    args = parser.parse_args(argv)

    from app.internal.ingest import PdfIngestor

    ingestor = PdfIngestor(collection_name=args.collection)

    if args.reset:
        ingestor.reset()

    if args.knowledge_dir:
        count = ingestor.ingest_directory(
            args.knowledge_dir, title=args.title, author=args.author
        )
    else:
        count = ingestor.ingest_pdf(
            args.pdf, title=args.title, author=args.author
        )

    print(f"Ingested {count} chunks.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())