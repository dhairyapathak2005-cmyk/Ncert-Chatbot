"""Local MiniLM embedder. Vectors are L2-normalized, so inner product == cosine."""
from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Load the model once per process."""
    return SentenceTransformer(settings.embed_model, device="cpu")


def dimension() -> int:
    """Embedding size, detected from the loaded model."""
    return int(get_model().get_embedding_dimension())


def embed(texts: str | Sequence[str]) -> np.ndarray:
    """Embed one or many texts. Always returns a 2-D float32 array of shape (n, dim)."""
    if isinstance(texts, str):
        texts = [texts]
    vecs = get_model().encode(
        list(texts), normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
    )
    return np.ascontiguousarray(vecs, dtype=np.float32)
