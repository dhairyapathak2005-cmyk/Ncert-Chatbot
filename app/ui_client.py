"""Small UI-facing adapter for the API or the in-process LangGraph app."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import requests
import streamlit as st

_log = logging.getLogger(__name__)


class UIClientError(RuntimeError):
    """A safe, user-facing failure while asking a question or reading cache stats."""


@dataclass
class _InProcessService:
    graph: Any
    cache: Any


@st.cache_resource(show_spinner=False)
def _in_process_service() -> _InProcessService:
    """Build the local graph once for the lifetime of the Streamlit process."""
    from app import embeddings, ingest
    from app.cache import SemanticCache
    from app.config import settings
    from app.graph import Deps, build_graph, make_equivalence_judge, make_llm

    index, chunks = ingest.load()
    dim = embeddings.dimension()
    if index.d != dim:
        raise UIClientError("The textbook index needs to be rebuilt before the helper can start.")

    llm = make_llm()
    cache = SemanticCache(
        dim=dim,
        low=settings.cache_low,
        high=settings.cache_high,
        ttl_seconds=settings.cache_ttl_hours * 3600,
        max_size=settings.cache_max_size,
        judge=make_equivalence_judge(llm),
    )
    graph = build_graph(Deps(embeddings.embed, index, chunks, cache, llm))
    return _InProcessService(graph=graph, cache=cache)


def _api_url() -> str | None:
    value = os.getenv("API_URL", "").strip().rstrip("/")
    return value or None


def _request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
    api_url = _api_url()
    if api_url is None:
        raise UIClientError("No API URL is configured.")
    try:
        response = requests.request(method, f"{api_url}{path}", timeout=60, **kwargs)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        _log.exception("API request failed")
        raise UIClientError("The answer service is unavailable right now.") from exc
    if not isinstance(payload, dict):
        raise UIClientError("The answer service returned an unexpected response.")
    return payload


def ask(question: str) -> dict[str, Any]:
    """Ask the configured API, or run the local LangGraph app when API_URL is absent."""
    clean_question = question.strip()
    if not clean_question:
        raise UIClientError("Please enter a science question.")

    if _api_url():
        return _request("POST", "/ask", json={"question": clean_question})

    try:
        from app.graph import run

        state = run(_in_process_service().graph, clean_question)
        return {
            "answer": state["answer"],
            "sources": state.get("sources", []),
            "cache": state["cache_info"],
            "latency_ms": state["latency_ms"],
        }
    except UIClientError:
        raise
    except Exception as exc:
        _log.exception("UI request failed")  # print the real cause in the terminal
        raise UIClientError("The local answer service is unavailable right now.") from exc


def cache_stats() -> dict[str, int]:
    """Return API cache statistics, or the same data from the local cache."""
    if _api_url():
        payload = _request("GET", "/cache/stats")
        try:
            return {key: int(payload[key]) for key in ("hits", "misses", "size")}
        except (KeyError, TypeError, ValueError) as exc:
            raise UIClientError("Cache statistics are unavailable right now.") from exc
    try:
        return _in_process_service().cache.stats()
    except Exception as exc:
        _log.exception("UI request failed")  # print the real cause in the terminal
        raise UIClientError("Cache statistics are unavailable right now.") from exc
