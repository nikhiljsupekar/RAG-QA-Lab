"""Top-K retrieval plus an explicit evidence-sufficiency decision."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .chunking import Chunk
from .embeddings import Embedder
from .vector_store import VectorStore


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    rank: int


@dataclass
class RetrievalResult:
    question: str
    hits: list[RetrievedChunk] = field(default_factory=list)
    sufficient: bool = False
    reason: str = ""
    latency_ms: float = 0.0
    min_score: float = 0.0

    @property
    def top_score(self) -> float:
        return self.hits[0].score if self.hits else 0.0


class Retriever:
    def __init__(self, embedder: Embedder, store: VectorStore, min_score: float | None = None):
        self.embedder = embedder
        self.store = store
        self.min_score = embedder.default_min_score if min_score is None else min_score

    def retrieve(self, question: str, top_k: int = 4) -> RetrievalResult:
        started = time.perf_counter()
        result = RetrievalResult(question=question, min_score=self.min_score)
        if not question or not question.strip():
            result.reason = "Empty question."
        elif len(self.store) == 0:
            result.reason = "No indexed content (no documents, or all documents were empty)."
        else:
            vector = self.embedder.embed([question])[0]
            hits = self.store.search(vector, top_k)
            result.hits = [RetrievedChunk(c, s, i + 1) for i, (c, s) in enumerate(hits)]
            if not result.hits or result.top_score < self.min_score:
                result.reason = (
                    f"Best match score {result.top_score:.3f} is below the threshold {self.min_score:.3f}."
                )
            else:
                result.sufficient = True
                result.reason = f"Best match score {result.top_score:.3f} >= threshold {self.min_score:.3f}."
        result.latency_ms = (time.perf_counter() - started) * 1000
        return result
