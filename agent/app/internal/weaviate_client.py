"""Lazy Weaviate client factory.

The ``weaviate`` import is deferred so that the module can be imported
in test environments or lightweight contexts where weaviate-client is
not installed.  The client is created on first use via ``get_weaviate_client``.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from ..config import weaviate_api_key, weaviate_url

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_weaviate_client():
    """Return a cached Weaviate client connected to the configured endpoint.

    Raises ImportError if the ``weaviate`` package is not installed, so
    callers can catch it and fall back gracefully.
    """
    import weaviate

    url = weaviate_url()
    api_key = weaviate_api_key()

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    logger.info("Connecting to Weaviate at %s.", url)
    return weaviate.Client(url=url, additional_headers=headers or None)


def close_weaviate_client():
    """Flush the cached client so the next call creates a fresh connection."""
    get_weaviate_client.cache_clear()