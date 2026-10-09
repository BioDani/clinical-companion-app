"""Hugging Face embeddings for knowledge chunks and search queries."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from huggingface_hub import InferenceClient

from ..config import hf_embedding_model, hf_token

_BATCH = 16


@lru_cache(maxsize=1)
def _client() -> InferenceClient:
    return InferenceClient(token=hf_token())


def sentence_vectors(raw: object, count: int) -> list[list[float]]:
    """Collapse each text to one vector. Token matrices are mean-pooled."""
    if count < 1:
        return []
    if count > 1 and isinstance(raw, (list, tuple)) and len(raw) == count:
        return [_one_vector(item) for item in raw]
    return _batch_array(np.asarray(raw, dtype=np.float64), count)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed texts in batches so the knowledge corpus is not one request."""
    if not texts:
        return []
    client = _client()
    model = hf_embedding_model()
    vectors: list[list[float]] = []
    for start in range(0, len(texts), _BATCH):
        batch = texts[start : start + _BATCH]
        raw = client.feature_extraction(batch, model=model)
        vectors.extend(sentence_vectors(raw, len(batch)))
    return vectors


def _one_vector(raw: object) -> list[float]:
    array = np.asarray(raw, dtype=np.float64)
    if array.ndim == 1:
        return array.tolist()
    if array.ndim == 2:
        return array.mean(axis=0).tolist()
    raise ValueError(f"Unexpected embedding shape {array.shape}")


def _batch_array(array: np.ndarray, count: int) -> list[list[float]]:
    if count == 1:
        if array.ndim == 3 and array.shape[0] == 1:
            return [array[0].mean(axis=0).tolist()]
        return [_one_vector(array)]
    if array.ndim == 2 and array.shape[0] == count:
        return [row.tolist() for row in array]
    if array.ndim == 3 and array.shape[0] == count:
        return [row.mean(axis=0).tolist() for row in array]
    raise ValueError(f"Unexpected embedding shape {array.shape} for {count} texts")
