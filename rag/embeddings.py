"""Embedding backends.

* ``SentenceTransformerEmbedder`` - the real semantic model (downloads weights on first use).
* ``HashingEmbedder`` - a deterministic, offline, *lexical* fallback. It is NOT semantic: it only
  matches shared words. It exists so the app and the test-suite run with no network access, and the
  UI always shows which embedder is active.
"""
from __future__ import annotations

import zlib
from typing import Protocol

import numpy as np

from .textutils import content_terms


class Embedder(Protocol):
    name: str
    default_min_score: float
    semantic: bool  # False for lexical-only embedders: they cannot match paraphrases

    def embed(self, texts: list[str]) -> np.ndarray: ...


def _normalise_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype("float32")


class HashingEmbedder:
    """Signed feature hashing of unigrams + bigrams, L2-normalised (cosine == inner product)."""

    default_min_score = 0.12
    semantic = False

    def __init__(self, dim: int = 1024):
        self.dim = dim
        self.name = f"hashing-{dim} (lexical fallback, not semantic)"

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for row, text in enumerate(texts):
            terms = content_terms(text)
            features = terms + [f"{a}_{b}" for a, b in zip(terms, terms[1:])]
            for feat in features:
                h = zlib.crc32(feat.encode("utf-8"))
                sign = 1.0 if (h >> 31) & 1 else -1.0
                out[row, h % self.dim] += sign
        # sub-linear tf damping keeps long chunks from dominating
        out = np.sign(out) * np.log1p(np.abs(out))
        return _normalise_rows(out)


class SentenceTransformerEmbedder:
    default_min_score = 0.30  # cosine similarity for MiniLM-class models; tune in the UI
    semantic = True

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # heavy import, keep lazy

        self.model = SentenceTransformer(model_name)
        self.name = model_name

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts, batch_size=32, convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype="float32")


class EmbedderChoice:
    """Result of backend selection, including why a fallback happened."""

    def __init__(self, embedder: Embedder, fallback_reason: str | None = None):
        self.embedder = embedder
        self.fallback_reason = fallback_reason


def get_embedder(backend: str = "auto", model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> EmbedderChoice:
    backend = backend.lower()
    if backend == "hashing":
        return EmbedderChoice(HashingEmbedder())
    if backend not in ("auto", "sentence-transformers"):
        raise ValueError(f"Unknown embedding backend: {backend}")
    try:
        return EmbedderChoice(SentenceTransformerEmbedder(model_name))
    except Exception as exc:  # missing package, no network, bad model name ...
        if backend == "sentence-transformers":
            raise
        return EmbedderChoice(HashingEmbedder(), f"{type(exc).__name__}: {str(exc)[:200]}")
