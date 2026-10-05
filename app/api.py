"""FastAPI app. Run: uvicorn app.api:app"""
from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, StringConstraints

from app import embeddings, ingest
from app.cache import SemanticCache
from app.config import settings
from app.graph import Deps, LLMUnavailableError, build_graph, make_equivalence_judge, make_llm, run

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)  # hide per-request HF Hub/LLM HTTP logs


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load index, embedder, LLM client, cache, and graph once."""
    index, chunks = ingest.load()
    dim = embeddings.dimension()
    if index.d != dim:
        raise RuntimeError(f"Index dim {index.d} != embedder dim {dim}; re-run ingest.")
    llm = make_llm()
    cache = SemanticCache(
        dim=dim,
        low=settings.cache_low,
        high=settings.cache_high,
        ttl_seconds=settings.cache_ttl_hours * 3600,
        max_size=settings.cache_max_size,
        judge=make_equivalence_judge(llm),
    )
    app.state.cache = cache
    app.state.index_size = index.ntotal
    app.state.graph = build_graph(Deps(embeddings.embed, index, chunks, cache, llm))
    yield


app = FastAPI(title="NCERT Class 10 Science RAG", lifespan=lifespan)


class AskRequest(BaseModel):
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]


class Source(BaseModel):
    chapter: str
    page: int
    snippet: str


class CacheInfo(BaseModel):
    hit: bool
    type: Literal["exact", "semantic", "none"]
    similarity: float | None


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    cache: CacheInfo
    latency_ms: float


@app.post("/ask", response_model=AskResponse)
def ask(body: AskRequest, request: Request) -> AskResponse:
    # Sync endpoint: FastAPI runs it in a worker thread, so blocking model calls are fine.
    try:
        state = run(request.app.state.graph, body.question)
    except LLMUnavailableError:
        raise HTTPException(status_code=503, detail="The answer service is busy. Please try again shortly.")
    return AskResponse(
        answer=state["answer"],
        sources=state.get("sources", []),
        cache=state["cache_info"],
        latency_ms=state["latency_ms"],
    )


@app.get("/health")
def health(request: Request) -> dict[str, object]:
    return {"status": "ok", "index_chunks": request.app.state.index_size, "chat_model": settings.chat_model}


@app.get("/cache/stats")
def cache_stats(request: Request) -> dict[str, int]:
    return request.app.state.cache.stats()
