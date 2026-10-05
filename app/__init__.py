"""Package init. Set threading env vars before faiss/torch load.

faiss-cpu and torch each bundle OpenMP; on macOS (Apple Silicon) loading both can segfault.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
