"""LangGraph flow: check_cache -> (hit: END | miss: retrieve -> generate -> write_cache)."""
from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from typing import Any, TypedDict

import faiss
import numpy as np
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.graph import END, START, StateGraph

from app.cache import LookupResult, SemanticCache, normalize
from app.config import settings
from app.ingest import Chunk

log = logging.getLogger(__name__)

NOT_COVERED = (
    "Sorry, this topic is not covered in the NCERT Class 10 Science textbook, "
    "so I can't answer it here."
)
ANSWER_SYSTEM_PROMPT = (
    "You are a friendly tutor for NCERT Class 10 Science students. "
    "Answer ONLY using the textbook context provided. Use simple language a Class 10 "
    "student can follow. If the context does not contain enough information to answer, "
    "say that the textbook context is insufficient instead of guessing."
)
EQUIVALENCE_PROMPT = "Do these two questions need the same answer? Reply yes or no."
SNIPPET_CHARS = 200
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


class LLMUnavailableError(RuntimeError):
    """The chat model failed after all retries."""


class RAGState(TypedDict, total=False):
    question: str
    normalized_question: str
    question_vec: np.ndarray
    cache_result: LookupResult
    chunks: list[dict[str, Any]]  # retrieved chunks with "score"
    answer: str
    sources: list[dict[str, Any]]
    cache_info: dict[str, Any]
    latency_ms: float


# ---- LLM helpers -------------------------------------------------------------


def make_llm() -> ChatNVIDIA:
    if not settings.nvidia_api_key:
        raise RuntimeError("NVIDIA_API_KEY is not set.")
    return ChatNVIDIA(
        model=settings.chat_model,
        api_key=settings.nvidia_api_key,
        temperature=0.2,
        max_completion_tokens=1024,
        # Nemotron 3.5 reasons by default (hundreds of hidden tokens, seconds per call).
        # Off keeps the gray-band yes/no check fast. with_thinking_mode(False) does not
        # work for this model in langchain-nvidia-ai-endpoints 1.4, so set it directly.
        model_kwargs={"chat_template_kwargs": {"enable_thinking": False}},
    )


def invoke_with_retry(
    llm: BaseChatModel, messages: list[BaseMessage], attempts: int = 3, base_delay: float = 1.0
) -> str:
    """Call the LLM with exponential backoff (1s, 2s, ...); raise LLMUnavailableError at the end."""
    for attempt in range(1, attempts + 1):
        try:
            text = str(llm.invoke(messages).content)
            return _THINK.sub("", text).strip()  # drop reasoning traces if the model emits them
        except Exception as exc:  # network, rate limit, 5xx ...
            log.warning("LLM call failed (attempt %d/%d): %s", attempt, attempts, exc)
            if attempt < attempts:
                time.sleep(base_delay * 2 ** (attempt - 1))
    raise LLMUnavailableError("Chat model unavailable after retries.")


def make_equivalence_judge(llm: BaseChatModel) -> Callable[[str, str], bool]:
    """Gray-band check for the cache. An unavailable LLM counts as 'no' (safe miss)."""

    def judge(cached_q: str, new_q: str) -> bool:
        prompt = f"{EQUIVALENCE_PROMPT}\n\nQuestion 1: {cached_q}\nQuestion 2: {new_q}"
        try:
            reply = invoke_with_retry(llm, [HumanMessage(prompt)])
        except LLMUnavailableError:
            return False
        return reply.lower().lstrip(" \"'*").startswith("yes")

    return judge


# ---- graph -------------------------------------------------------------------


@dataclass
class Deps:
    embed: Callable[[str], np.ndarray]  # text -> (1, dim) normalized vectors
    index: faiss.Index
    chunks: list[Chunk]
    cache: SemanticCache
    llm: BaseChatModel


def _timed(name: str, fn: Callable[[RAGState], dict[str, Any]]) -> Callable[[RAGState], dict[str, Any]]:
    @wraps(fn)
    def wrapper(state: RAGState) -> dict[str, Any]:
        t0 = time.perf_counter()
        try:
            return fn(state)
        finally:
            log.info("step=%s took %.1f ms", name, (time.perf_counter() - t0) * 1000)

    return wrapper


def build_graph(deps: Deps):  # returns a compiled LangGraph
    def check_cache(state: RAGState) -> dict[str, Any]:
        question = state["question"]
        vec = deps.embed(question)  # the ONLY embedding call per request
        result = deps.cache.lookup(question, vec)
        update: dict[str, Any] = {
            "normalized_question": normalize(question),
            "question_vec": vec,
            "cache_result": result,
            "cache_info": {"hit": result["hit"], "type": result["type"], "similarity": result["similarity"]},
        }
        if result["hit"] and result["entry"] is not None:
            update["answer"] = result["entry"].answer
            update["sources"] = result["entry"].sources
        return update

    def retrieve(state: RAGState) -> dict[str, Any]:
        k = min(settings.top_k, deps.index.ntotal)
        scores, rows = deps.index.search(state["question_vec"], k)
        found = [
            {**deps.chunks[int(r)], "score": float(s)}
            for s, r in zip(scores[0], rows[0])
            if r != -1 and s >= settings.relevance_min
        ]
        if not found:  # best score below RELEVANCE_MIN
            return {"chunks": [], "answer": NOT_COVERED, "sources": []}
        return {"chunks": found}

    def generate(state: RAGState) -> dict[str, Any]:
        chunks = state["chunks"]
        context = "\n\n".join(
            f"[{i}] (Chapter: {c['chapter']}, page {c['page']})\n{c['text']}"
            for i, c in enumerate(chunks, start=1)
        )
        messages = [
            SystemMessage(ANSWER_SYSTEM_PROMPT),
            HumanMessage(f"Textbook context:\n{context}\n\nStudent question: {state['question']}"),
        ]
        answer = invoke_with_retry(deps.llm, messages)
        sources = [
            {"chapter": c["chapter"], "page": c["page"], "snippet": c["text"][:SNIPPET_CHARS]}
            for c in chunks
        ]
        return {"answer": answer, "sources": sources}

    def write_cache(state: RAGState) -> dict[str, Any]:
        deps.cache.add(state["question"], state["question_vec"], state["answer"], state["sources"])
        return {}

    g = StateGraph(RAGState)
    g.add_node("check_cache", _timed("check_cache", check_cache))
    g.add_node("retrieve", _timed("retrieve", retrieve))
    g.add_node("generate", _timed("generate", generate))
    g.add_node("write_cache", _timed("write_cache", write_cache))
    g.add_edge(START, "check_cache")
    g.add_conditional_edges(
        "check_cache", lambda s: "hit" if s["cache_result"]["hit"] else "miss",
        {"hit": END, "miss": "retrieve"},
    )
    g.add_conditional_edges(
        "retrieve", lambda s: "relevant" if s["chunks"] else "not_covered",
        {"relevant": "generate", "not_covered": END},
    )
    g.add_edge("generate", "write_cache")
    g.add_edge("write_cache", END)
    return g.compile()


def run(graph, question: str) -> RAGState:
    """Invoke the graph and record total latency."""
    t0 = time.perf_counter()
    state: RAGState = graph.invoke({"question": question})
    state["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    log.info("total latency %.1f ms (cache=%s)", state["latency_ms"], state["cache_info"]["type"])
    return state
