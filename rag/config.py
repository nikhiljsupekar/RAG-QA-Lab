"""Central settings. Secrets are read from the environment only - never hardcoded."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

try:  # python-dotenv is optional at import time so tests never depend on a .env file
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

INSUFFICIENT_EVIDENCE_MESSAGE = (
    "I could not find enough evidence in the uploaded documents to answer this question."
)

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class Settings:
    chunk_size: int = 800  # characters
    chunk_overlap: int = 120  # characters
    top_k: int = 4
    # None -> use the embedder's own recommended threshold (cosine similarity).
    min_score: float | None = None
    embedding_backend: str = "auto"  # auto | sentence-transformers | hashing
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    latency_budget_ms: float = 2000.0
    claude_model: str = "claude-opus-5-5"
    # Lets the UI index files by typing a local path (fallback when the browser blocks the file dialog).
    # Reads from the machine running the app, so set ALLOW_LOCAL_PATHS=0 for any shared deployment.
    allow_local_paths: bool = True
    # Read lazily from the environment; kept out of repr so it never lands in logs.
    anthropic_api_key: str | None = field(default=None, repr=False)

    @classmethod
    def from_env(cls) -> "Settings":
        if load_dotenv is not None:
            load_dotenv()
        defaults = cls()
        return cls(
            chunk_size=int(os.getenv("CHUNK_SIZE", defaults.chunk_size)),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", defaults.chunk_overlap)),
            top_k=int(os.getenv("TOP_K", defaults.top_k)),
            embedding_backend=os.getenv("EMBEDDING_BACKEND", defaults.embedding_backend),
            embedding_model=os.getenv("EMBEDDING_MODEL", defaults.embedding_model),
            claude_model=os.getenv("ANTHROPIC_MODEL", defaults.claude_model),
            allow_local_paths=os.getenv("ALLOW_LOCAL_PATHS", "1").strip().lower() not in ("0", "false", "no"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY") or None,
        )
