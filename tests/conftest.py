from __future__ import annotations

import app  # noqa: F401  # macOS OpenMP fix must load before faiss

import faiss
import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.cache import SemanticCache
from app.config import settings
from app.embeddings import dimension, embed
from app.graph import Deps, build_graph

TEXTBOOK = [
    {"chapter": "Life Processes", "page": 3,
     "text": "Photosynthesis is the process by which green plants use sunlight, carbon dioxide "
             "and water to make carbohydrates. Chlorophyll absorbs light energy and oxygen is released."},
    {"chapter": "Life Processes", "page": 12,
     "text": "Xylem tissue transports water and minerals from the roots to the leaves of a plant."},
    {"chapter": "Electricity", "page": 5,
     "text": "Ohm's law states that the current through a conductor is proportional to the potential difference."},
]


class Clock:
    """Controllable time source for TTL tests."""

    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock() -> Clock:
    return Clock()


def make_cache(judge=None, clock=None, ttl=3600.0, max_size=50) -> SemanticCache:
    return SemanticCache(
        dim=dimension(), low=settings.cache_low, high=settings.cache_high,
        ttl_seconds=ttl, max_size=max_size, judge=judge, clock=clock or (lambda: 0.0),
    )


@pytest.fixture
def graph_and_cache():
    vecs = embed([c["text"] for c in TEXTBOOK])
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    cache = make_cache(judge=lambda a, b: False)
    llm = FakeListChatModel(responses=["Plants make food from sunlight (fake answer)."])
    return build_graph(Deps(embed, index, TEXTBOOK, cache, llm)), cache
