"""FAISS-backed vector store (exact inner-product search over L2-normalised vectors)."""
from __future__ import annotations

import faiss
import numpy as np

from .chunking import Chunk


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: list[Chunk] = []

    def __len__(self) -> int:
        return len(self.chunks)

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        if not chunks:
            return
        vectors = np.ascontiguousarray(vectors, dtype="float32")
        if vectors.shape[1] != self.dim:
            raise ValueError(f"expected vectors of dim {self.dim}, got {vectors.shape[1]}")
        self.index.add(vectors)
        self.chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[Chunk, float]]:
        if not self.chunks or k <= 0:
            return []
        q = np.ascontiguousarray(query_vector.reshape(1, -1), dtype="float32")
        scores, ids = self.index.search(q, min(k, len(self.chunks)))
        return [(self.chunks[i], float(s)) for s, i in zip(scores[0], ids[0]) if i != -1]
