"""Ingest + query orchestration. Answer generation is injected, so nothing here knows about any LLM."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from .answering import (Answer, AnswerGenerator, ExtractiveAnswerer, GroundingResult,
                        check_grounding, insufficient_answer)
from .chunking import Chunk, chunk_document
from .config import Settings
from .embeddings import Embedder, get_embedder
from .loaders import Document
from .retriever import RetrievalResult, Retriever
from .vector_store import VectorStore


@dataclass
class IngestReport:
    documents_indexed: list[str] = field(default_factory=list)
    documents_skipped: list[str] = field(default_factory=list)  # "name: reason"
    chunk_count: int = 0
    embed_seconds: float = 0.0


@dataclass
class RAGResponse:
    question: str
    answer: Answer
    retrieval: RetrievalResult
    retrieval_ms: float
    generation_ms: float
    total_ms: float
    grounding: GroundingResult | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def sources(self) -> list[str]:
        seen: list[str] = []
        for h in self.retrieval.hits:
            if h.chunk.source not in seen:
                seen.append(h.chunk.source)
        return seen


class RAGPipeline:
    def __init__(self, settings: Settings | None = None, embedder: Embedder | None = None):
        self.settings = settings or Settings()
        self.fallback_reason: str | None = None
        if embedder is None:
            choice = get_embedder(self.settings.embedding_backend, self.settings.embedding_model)
            embedder, self.fallback_reason = choice.embedder, choice.fallback_reason
        self.embedder = embedder
        self.store: VectorStore | None = None
        self.retriever: Retriever | None = None
        self.chunks: list[Chunk] = []

    # ------------------------------------------------------------------ indexing
    def ingest(self, documents: list[Document],
               progress: Callable[[float, str], None] | None = None) -> IngestReport:
        """(Re)build the index from scratch from `documents`."""
        started = time.perf_counter()
        report = IngestReport()
        chunks: list[Chunk] = []
        for i, doc in enumerate(documents):
            if doc.is_empty:
                report.documents_skipped.append(
                    f"{doc.filename}: no extractable text (empty file, or a scanned PDF without OCR)")
                continue
            doc_chunks = chunk_document(doc, self.settings.chunk_size, self.settings.chunk_overlap)
            if not doc_chunks:
                report.documents_skipped.append(f"{doc.filename}: produced no chunks")
                continue
            chunks.extend(doc_chunks)
            report.documents_indexed.append(doc.filename)
            if progress:
                progress((i + 1) / max(len(documents), 1) * 0.3, f"Chunked {doc.filename}")

        self.chunks = chunks
        self.store = None
        self.retriever = None
        if chunks:
            vectors = self.embedder.embed([c.text for c in chunks])
            self.store = VectorStore(vectors.shape[1])
            self.store.add(chunks, vectors)
            self.retriever = Retriever(self.embedder, self.store, self.settings.min_score)
        report.chunk_count = len(chunks)
        report.embed_seconds = time.perf_counter() - started
        if progress:
            progress(1.0, "Index ready")
        return report

    @property
    def is_ready(self) -> bool:
        return self.retriever is not None

    # ------------------------------------------------------------------ querying
    def retrieve(self, question: str, top_k: int | None = None) -> RetrievalResult:
        k = top_k or self.settings.top_k
        if self.retriever is None:
            return RetrievalResult(question=question, reason="No documents have been indexed yet.")
        return self.retriever.retrieve(question, k)

    def ask(self, question: str, generator: AnswerGenerator | None = None,
            top_k: int | None = None) -> RAGResponse:
        started = time.perf_counter()
        retrieval = self.retrieve(question, top_k)
        warnings: list[str] = []
        gen_ms = 0.0
        grounding: GroundingResult | None = None

        if not retrieval.sufficient:
            answer = insufficient_answer()  # the generator is never called without evidence
        else:
            generator = generator or ExtractiveAnswerer(self.embedder)
            t0 = time.perf_counter()
            try:
                answer = generator.generate(question, retrieval.hits)
            except Exception as exc:  # e.g. network / API failure in an optional LLM provider
                warnings.append(f"Generator failed ({type(exc).__name__}: {exc}); used extractive fallback.")
                answer = ExtractiveAnswerer(self.embedder).generate(question, retrieval.hits)
            gen_ms = (time.perf_counter() - t0) * 1000
            if not answer.insufficient:
                grounding = check_grounding(answer.text, [h.chunk.text for h in retrieval.hits])
                if not grounding.supported:
                    warnings.append("Grounding check failed: " + grounding.detail)

        return RAGResponse(
            question=question, answer=answer, retrieval=retrieval,
            retrieval_ms=retrieval.latency_ms, generation_ms=gen_ms,
            total_ms=(time.perf_counter() - started) * 1000,
            grounding=grounding, warnings=warnings,
        )
