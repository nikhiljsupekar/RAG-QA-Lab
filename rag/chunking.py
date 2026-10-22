"""Configurable chunking with overlap and stable, human-readable chunk IDs."""
from __future__ import annotations

from dataclasses import dataclass

from .loaders import Document

_BREAKS = ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source: str
    text: str
    start: int  # character offsets into the source document text
    end: int
    page: int | None = None


def validate_params(chunk_size: int, overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must not be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")


def _find_end(text: str, start: int, chunk_size: int) -> int:
    """End offset for a chunk starting at `start`, preferring natural boundaries."""
    hard_end = min(start + chunk_size, len(text))
    if hard_end >= len(text):
        return len(text)
    window_floor = start + chunk_size // 2  # never cut the chunk below half its size
    for sep in _BREAKS:
        idx = text.rfind(sep, window_floor, hard_end)
        if idx != -1:
            return idx + len(sep)
    return hard_end


def chunk_text(text: str, source: str, chunk_size: int = 800, overlap: int = 120,
               doc: Document | None = None) -> list[Chunk]:
    validate_params(chunk_size, overlap)
    if not text or not text.strip():
        return []
    chunks: list[Chunk] = []
    start = 0
    while start < len(text):
        end = _find_end(text, start, chunk_size)
        raw = text[start:end]
        body = raw.strip()
        if body:
            real_start = start + (len(raw) - len(raw.lstrip()))
            chunks.append(Chunk(
                chunk_id=f"{source}::chunk-{len(chunks):04d}",
                source=source,
                text=body,
                start=real_start,
                end=real_start + len(body),
                page=doc.page_at(real_start) if doc else None,
            ))
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)  # always make forward progress
    return chunks


def chunk_document(doc: Document, chunk_size: int = 800, overlap: int = 120) -> list[Chunk]:
    return chunk_text(doc.text, doc.filename, chunk_size, overlap, doc)
