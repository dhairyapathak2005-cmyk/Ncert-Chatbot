"""Settings, read once from the environment (and an optional .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT_DIR / "data" / "pdfs"
INDEX_DIR = ROOT_DIR / "data" / "index"


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


@dataclass(frozen=True)
class Settings:
    nvidia_api_key: str | None = field(default_factory=lambda: os.getenv("NVIDIA_API_KEY"))
    # The API catalog ID is namespaced ("nvidia/...").
    chat_model: str = field(
        default_factory=lambda: os.getenv("CHAT_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b")
    )
    embed_model: str = field(
        default_factory=lambda: os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )
    cache_low: float = field(default_factory=lambda: _float("CACHE_LOW", 0.80))
    cache_high: float = field(default_factory=lambda: _float("CACHE_HIGH", 0.92))
    # 0.40 from measured scores: in-syllabus best-chunk >= 0.495, off-topic <= 0.331.
    relevance_min: float = field(default_factory=lambda: _float("RELEVANCE_MIN", 0.40))
    cache_ttl_hours: float = field(default_factory=lambda: _float("CACHE_TTL_HOURS", 24))
    cache_max_size: int = field(default_factory=lambda: _int("CACHE_MAX_SIZE", 500))
    top_k: int = field(default_factory=lambda: _int("TOP_K", 4))


settings = Settings()
