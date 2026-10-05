"""PDFs in data/pdfs/ -> chunks -> FAISS index in data/index/.

Run: python -m app.ingest
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TypedDict

import faiss
import numpy as np
import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import INDEX_DIR, PDF_DIR
from app.embeddings import embed

log = logging.getLogger(__name__)

INDEX_FILE = "textbook.faiss"
CHUNKS_FILE = "chunks.json"


class Chunk(TypedDict):
    text: str
    chapter: str
    page: int


def _printed_page_number(page: pymupdf.Page) -> int | None:
    """Extract the real printed page number from a PDF page.

    NCERT pages have a small-font (≈8.5pt) standalone number at the header or footer.
    Returns None if no printed page number is detected (e.g. title pages).
    """
    page_height = page.rect.height
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            text = " ".join(s["text"] for s in line["spans"]).strip()
            bbox = line["bbox"]
            y_pos = bbox[1]
            size = max((s["size"] for s in line["spans"]), default=0.0)
            at_edge = y_pos < page_height * 0.15 or y_pos > page_height * 0.85
            if re.fullmatch(r"\d{1,3}", text) and size < 15 and at_edge:
                return int(text)
    return None


def _page_offset(doc: pymupdf.Document) -> int:
    """Determine the book-page offset for a chapter PDF.

    Scans up to the first 5 pages to find a printed page number, then computes
    offset such that: real_book_page = pdf_page_index + offset.
    Falls back to 1 (i.e. PDF page 0 == book page 1) if nothing is found.
    """
    for page_idx in range(min(5, len(doc))):
        printed = _printed_page_number(doc[page_idx])
        if printed is not None:
            return printed - page_idx
    return 1


def _first_heading(doc: pymupdf.Document) -> str | None:
    """Chapter title on page 1: all lines in the largest font, joined (titles often wrap).

    Lines of 3 characters or fewer are skipped (chapter numbers, drop caps).
    """
    lines: list[tuple[float, str]] = []
    for block in doc[0].get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            text = " ".join(s["text"] for s in line["spans"]).strip()
            size = max((s["size"] for s in line["spans"]), default=0.0)
            if len(text) > 3:
                lines.append((size, text))
    if not lines:
        return None
    top = max(size for size, _ in lines)
    return " ".join(text for size, text in lines if abs(size - top) < 0.5)


def chapter_name(path: Path, doc: pymupdf.Document) -> str:
    """Readable filenames ("life_processes.pdf") win; NCERT codes ("jesc106.pdf") use the heading."""
    stem = path.stem
    if not re.fullmatch(r"[A-Za-z]+\d+", stem):
        return re.sub(r"[_\-]+", " ", stem).strip().title()
    return _first_heading(doc) or stem


def load_chunks(pdf_dir: Path = PDF_DIR) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks: list[Chunk] = []
    for path in sorted(pdf_dir.glob("*.pdf")):
        with pymupdf.open(path) as doc:
            chapter = chapter_name(path, doc)
            offset = _page_offset(doc)
            num_pages = len(doc)
            for page_idx, page in enumerate(doc):
                book_page = page_idx + offset
                text = page.get_text().strip()
                for piece in splitter.split_text(text):
                    chunks.append({"text": piece, "chapter": chapter, "page": book_page})
        log.info("Loaded %s as chapter %r (book pages %d–%d)", path.name, chapter, offset, offset + num_pages - 1)
    return chunks


def build_index(chunks: list[Chunk]) -> faiss.IndexFlatIP:
    vecs = embed([c["text"] for c in chunks])
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    return index


def save(index: faiss.Index, chunks: list[Chunk], index_dir: Path = INDEX_DIR) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / INDEX_FILE))
    (index_dir / CHUNKS_FILE).write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")


def load(index_dir: Path = INDEX_DIR) -> tuple[faiss.Index, list[Chunk]]:
    """Load the saved index and its chunk metadata (row i of the index == chunks[i])."""
    index = faiss.read_index(str(index_dir / INDEX_FILE))
    chunks: list[Chunk] = json.loads((index_dir / CHUNKS_FILE).read_text(encoding="utf-8"))
    if index.ntotal != len(chunks):
        raise RuntimeError("Index and chunk metadata are out of sync; re-run ingest.")
    return index, chunks


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    chunks = load_chunks()
    if not chunks:
        raise SystemExit(f"No PDF text found in {PDF_DIR}")
    index = build_index(chunks)
    save(index, chunks)
    log.info("Saved %d chunks (dim=%d) to %s", index.ntotal, index.d, INDEX_DIR)


if __name__ == "__main__":
    main()
