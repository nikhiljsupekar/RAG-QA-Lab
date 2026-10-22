from pathlib import Path

import pytest

from rag.config import Settings
from rag.embeddings import HashingEmbedder
from rag.loaders import load_many
from rag.pipeline import RAGPipeline

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_docs"


@pytest.fixture(scope="session")
def sample_docs():
    files = [(p.name, p.read_bytes()) for p in sorted(SAMPLE_DIR.iterdir()) if p.suffix in (".txt", ".pdf")]
    docs, errors = load_many(files)
    assert not errors, errors
    return docs


@pytest.fixture(scope="session")
def pipeline(sample_docs):
    """Offline, deterministic pipeline: hashing embedder, so tests never need network access."""
    pipe = RAGPipeline(Settings(embedding_backend="hashing", chunk_size=500, chunk_overlap=80), HashingEmbedder())
    pipe.ingest(sample_docs)
    return pipe
