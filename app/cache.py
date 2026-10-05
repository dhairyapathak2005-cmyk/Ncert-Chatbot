"""Two-level answer cache: exact (normalized text) then semantic (question vectors)."""
from __future__ import annotations

import re
import string
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

import faiss
import numpy as np

_PUNCT = str.maketrans("", "", string.punctuation)
_NUMBER = re.compile(r"\d+(?:\.\d+)?")

# (cached_question, new_question) -> True if both need the same answer.
EquivalenceJudge = Callable[[str, str], bool]


def normalize(question: str) -> str:
    """Lowercase, strip punctuation, trim, collapse whitespace."""
    return " ".join(question.lower().translate(_PUNCT).split())


def numbers_match(a: str, b: str) -> bool:
    """'Ohm's law for 2 resistors' vs '... 3 resistors' must not share an answer."""
    return sorted(_NUMBER.findall(a)) == sorted(_NUMBER.findall(b))


@dataclass
class CacheEntry:
    question: str
    answer: str
    sources: list[dict[str, Any]]
    created_at: float
    vector: np.ndarray = field(repr=False)


class LookupResult(TypedDict):
    hit: bool
    type: Literal["exact", "semantic", "none"]
    similarity: float | None
    entry: CacheEntry | None


class SemanticCache:
    def __init__(
        self,
        dim: int,
        low: float,
        high: float,
        ttl_seconds: float,
        max_size: int,
        judge: EquivalenceJudge | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.low, self.high = low, high
        self.ttl, self.max_size = ttl_seconds, max_size
        self.judge, self.clock = judge, clock
        # OrderedDict order == LRU order (oldest first). Keyed by normalized question.
        self._entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self._index = faiss.IndexFlatIP(dim)
        self._row_keys: list[str] = []  # FAISS row i -> key in _entries
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    # ---- internal helpers (call with the lock held) -------------------------

    def _rebuild(self) -> None:
        """Re-create the FAISS index from _entries so rows and keys stay aligned."""
        self._index.reset()
        self._row_keys = list(self._entries)
        if self._row_keys:
            self._index.add(np.stack([e.vector for e in self._entries.values()]))

    def _purge_expired(self) -> None:
        now = self.clock()
        expired = [k for k, e in self._entries.items() if now - e.created_at >= self.ttl]
        for k in expired:
            del self._entries[k]
        if expired:
            self._rebuild()

    def _record_hit(self, key: str) -> None:
        self._entries.move_to_end(key)
        self.hits += 1

    # ---- public API ----------------------------------------------------------

    def lookup(self, question: str, vec: np.ndarray) -> LookupResult:
        """Look up a question using its precomputed normalized vector (shape (dim,) or (1, dim))."""
        key = normalize(question)
        with self._lock:
            self._purge_expired()
            if key in self._entries:
                self._record_hit(key)
                return {"hit": True, "type": "exact", "similarity": 1.0, "entry": self._entries[key]}
            if self._index.ntotal == 0:
                self.misses += 1
                return {"hit": False, "type": "none", "similarity": None, "entry": None}
            scores, rows = self._index.search(vec.reshape(1, -1).astype(np.float32), 1)
            sim = float(scores[0][0])
            cand_key = self._row_keys[int(rows[0][0])]
            cand = self._entries[cand_key]

        if sim >= self.high:
            accept = True
        elif sim >= self.low:
            # Gray band: cheap number check first, then the LLM (outside the lock, it is slow).
            accept = (
                numbers_match(cand.question, question)
                and self.judge is not None
                and self.judge(cand.question, question)
            )
        else:
            accept = False

        with self._lock:
            if accept and cand_key in self._entries:  # may have been evicted meanwhile
                self._record_hit(cand_key)
                return {"hit": True, "type": "semantic", "similarity": sim, "entry": cand}
            self.misses += 1
            return {"hit": False, "type": "none", "similarity": sim, "entry": None}

    def add(
        self, question: str, vec: np.ndarray, answer: str, sources: list[dict[str, Any]]
    ) -> None:
        key = normalize(question)
        entry = CacheEntry(
            question=question,
            answer=answer,
            sources=sources,
            created_at=self.clock(),
            vector=vec.reshape(-1).astype(np.float32),
        )
        with self._lock:
            self._purge_expired()
            replaced = key in self._entries
            self._entries[key] = entry
            self._entries.move_to_end(key)
            evicted = False
            while len(self._entries) > self.max_size:
                self._entries.popitem(last=False)  # least recently used
                evicted = True
            if replaced or evicted:
                self._rebuild()
            else:  # fast path: append one row
                self._index.add(entry.vector.reshape(1, -1))
                self._row_keys.append(key)

    def stats(self) -> dict[str, int]:
        with self._lock:
            self._purge_expired()
            return {"hits": self.hits, "misses": self.misses, "size": len(self._entries)}
