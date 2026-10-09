"""Build the Weaviate knowledge index and exit."""

from __future__ import annotations

import logging
import sys

from .vector_store import ensure_indexed

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        ensure_indexed()
    except Exception:
        logger.exception("Knowledge index was not built")
        sys.exit(1)


if __name__ == "__main__":
    main()
