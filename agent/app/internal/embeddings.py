"""Embedding service backed by Hugging Face Inference API.

Uses ``langchain_huggingface.HuggingFaceInferenceEmbeddings`` which
wraps the same ``huggingface_hub.InferenceClient`` that ``smolagents``
uses for the LLM, keeping the stack consistent.  The import is deferred
so tests that do not need embeddings are not blocked by a missing
``langchain-huggingface`` dependency.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from ..config import hf_embedding_model, hf_token

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embedding_model():
    """Return a cached HF inference embedding model.

    Requires ``langchain-huggingface`` to be installed and ``HF_TOKEN``
    to start with ``hf_``.
    """
    from langchain_huggingface import HuggingFaceInferenceEmbeddings

    model = HuggingFaceInferenceEmbeddings(
        model_name=hf_embedding_model(),
        huggingfacehub_api_key=hf_token(),
    )
    logger.info("HF embedding model: %s", hf_embedding_model())
    return model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts.  Creates the model on first call."""
    embeddings = get_embedding_model()
    vectors = embeddings.embed_documents(texts)
    return [list(v) for v in vectors]


def embed_query(query: str) -> list[float]:
    """Embed a single query string for retrieval."""
    embeddings = get_embedding_model()
    return list(embeddings.embed_query(query))