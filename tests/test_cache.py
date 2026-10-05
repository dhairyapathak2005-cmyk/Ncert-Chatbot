from __future__ import annotations

import os

import pytest

from app.cache import normalize, numbers_match
from app.config import settings
from app.embeddings import embed
from app.graph import NOT_COVERED, make_equivalence_judge, make_llm, run
from tests.conftest import make_cache

# "What is photosynthesis?" vs "Explain how photosynthesis works" was replaced: Nemotron
# consistently judges a definition and a how-it-works explanation as different answers.
PARAPHRASE = ("State Ohm's law", "What does Ohm's law say?")
NEAR_MISS = ("function of xylem", "structure of xylem")
needs_llm = pytest.mark.skipif(not os.getenv("NVIDIA_API_KEY"), reason="NVIDIA_API_KEY not set")


def _seed(cache, question: str) -> None:
    cache.add(question, embed(question), "answer", [])


def test_normalize() -> None:
    assert normalize("  What IS   photosynthesis?? ") == "what is photosynthesis"


def test_numbers_match() -> None:
    assert numbers_match("2 resistors of 4 ohm", "4 ohm and 2 resistors")
    assert not numbers_match("2 resistors", "3 resistors")


def test_exact_repeat_hits(graph_and_cache) -> None:
    graph, cache = graph_and_cache
    first = run(graph, "What is photosynthesis?")
    assert first["cache_info"]["hit"] is False
    second = run(graph, "  what is PHOTOSYNTHESIS ")
    assert second["cache_info"] == {"hit": True, "type": "exact", "similarity": 1.0}
    assert second["answer"] == first["answer"]
    assert cache.stats() == {"hits": 1, "misses": 1, "size": 1}


def test_out_of_syllabus_returns_not_covered(graph_and_cache) -> None:
    graph, cache = graph_and_cache
    state = run(graph, "Who won the FIFA World Cup in 2018?")
    assert state["answer"] == NOT_COVERED
    assert state["sources"] == []
    assert cache.stats()["size"] == 0  # not-covered replies are not cached


def test_ttl_expiry(clock) -> None:
    cache = make_cache(clock=clock, ttl=60)
    _seed(cache, "What is photosynthesis?")
    vec = embed("What is photosynthesis?")
    assert cache.lookup("What is photosynthesis?", vec)["hit"]
    clock.now += 61
    assert cache.lookup("What is photosynthesis?", vec)["type"] == "none"
    assert cache.stats()["size"] == 0


def test_lru_eviction_keeps_index_consistent() -> None:
    cache = make_cache(max_size=2)
    for q in ["What is photosynthesis?", "What is Ohm's law?", "What is a food chain?"]:
        _seed(cache, q)
    assert cache.stats()["size"] == 2
    # Oldest entry is gone from both the dict and the vector index.
    res = cache.lookup("What is photosynthesis?", embed("What is photosynthesis?"))
    assert res["type"] == "none" and res["similarity"] < settings.cache_high
    assert cache.lookup("What is a food chain?", embed("What is a food chain?"))["type"] == "exact"


@pytest.mark.parametrize("pair", [PARAPHRASE, NEAR_MISS])
def test_gray_band_defers_to_judge(pair) -> None:
    """With MiniLM both pairs land in the gray band, so the judge's verdict is what decides."""
    cached_q, new_q = pair
    for verdict, expected in [(True, "semantic"), (False, "none")]:
        cache = make_cache(judge=lambda a, b, v=verdict: v)
        _seed(cache, cached_q)
        res = cache.lookup(new_q, embed(new_q))
        assert settings.cache_low <= res["similarity"] < settings.cache_high
        assert res["type"] == expected


@needs_llm
def test_paraphrase_hits_with_llm_judge() -> None:
    cache = make_cache(judge=make_equivalence_judge(make_llm()))
    _seed(cache, PARAPHRASE[0])
    assert cache.lookup(PARAPHRASE[1], embed(PARAPHRASE[1]))["type"] == "semantic"


@needs_llm
def test_near_miss_does_not_hit_with_llm_judge() -> None:
    cache = make_cache(judge=make_equivalence_judge(make_llm()))
    _seed(cache, NEAR_MISS[0])
    assert cache.lookup(NEAR_MISS[1], embed(NEAR_MISS[1]))["hit"] is False
