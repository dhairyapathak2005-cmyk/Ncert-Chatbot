# NCERT Class 10 Science RAG chatbot: backend

FastAPI + LangGraph service that answers questions using only the NCERT Class 10 Science textbook. It has an exact and semantic answer cache.

```
check_cache ──hit──▶ END
    │ miss
retrieve ──not relevant──▶ END ("not covered" reply)
    │
generate ─▶ write_cache ─▶ END
```

The question is embedded once in `check_cache`. `retrieve` and `write_cache` reuse that vector.

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then set NVIDIA_API_KEY
```

MiniLM runs locally. It downloads from Hugging Face on first use, and no key is needed.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `NVIDIA_API_KEY` | (required) | Key for build.nvidia.com |
| `CHAT_MODEL` | `nvidia/nemotron-3.5-lightning-30b-a3b` | Chat model ID in the NVIDIA API catalog |
| `EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedder |
| `CACHE_LOW` / `CACHE_HIGH` | `0.80` / `0.92` | Semantic cache bands |
| `RELEVANCE_MIN` | `0.40` | Minimum textbook-chunk score for an answer |
| `CACHE_TTL_HOURS` | `24` | Cache entry lifetime |
| `CACHE_MAX_SIZE` | `500` | Maximum cache entries (LRU eviction) |
| `TOP_K` | `4` | Chunks retrieved per question |

## Ingest the textbook

Put the chapter PDFs in `data/pdfs/`, then run:

```bash
python -m app.ingest
```

This writes `data/index/textbook.faiss` and `data/index/chunks.json`. The chapter name comes from the filename (`life_processes.pdf` becomes "Life Processes"). For NCERT code names such as `jesc106.pdf`, it comes from the largest heading on page 1.

## Run the API

```bash
uvicorn app.api:app --port 8000
```

```bash
curl -s -X POST localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "What is photosynthesis?"}'
```

```json
{
  "answer": "...",
  "sources": [{"chapter": "Life Processes", "page": 3, "snippet": "..."}],
  "cache": {"hit": false, "type": "none", "similarity": null},
  "latency_ms": 1840.2
}
```

Other endpoints are `GET /health` and `GET /cache/stats` (returns `hits`, `misses`, `size`). After 3 failed LLM attempts with backoff, `/ask` returns `503`.

## Run the UI

Install the requirements, set `NVIDIA_API_KEY`, and run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

By default, the UI runs the LangGraph app in the same process. To use a separately running FastAPI service instead, set its base URL first:

```bash
API_URL=http://localhost:8000 streamlit run streamlit_app.py
```

For Streamlit Community Cloud, add `NVIDIA_API_KEY` in the app's **Settings → Secrets** as a TOML secret. The UI reads that secret before it loads the local graph. Set `API_URL` in the app's environment variables when deploying against a separate API service.

## Cache

1. **Exact:** dictionary lookup on the normalized question (lowercase, trimmed, whitespace collapsed, punctuation removed).
2. **Semantic:** nearest past question in a separate FAISS `IndexFlatIP`.
   - Similarity `>= CACHE_HIGH`: accept.
   - `CACHE_LOW <= similarity < CACHE_HIGH`: the numbers in both questions must match, then the LLM answers "Do these two questions need the same answer?" The hit is accepted only on "yes".
   - Below `CACHE_LOW`: miss.
3. **Eviction:** TTL and LRU size cap. The FAISS index is rebuilt on eviction, so it stays aligned with the dictionary.

## Tune thresholds

```bash
python -m scripts.eval_cache
```

This prints each labeled pair's similarity, plus hit rate and false-hit count for thresholds from 0.60 to 0.98. Set `CACHE_HIGH` above the highest score of any "different" pair you care about. Set `CACHE_LOW` near the lowest score of the "same" pairs you want to catch. The LLM check handles pairs in between.

## Tests

```bash
pytest -q
```

Offline tests cover exact repeats, gray-band behavior for the paraphrase and near-miss pairs, the out-of-syllabus reply, TTL, and LRU. The two LLM-judge tests run only when `NVIDIA_API_KEY` is set.
