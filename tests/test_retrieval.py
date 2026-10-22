import pytest

CASES = [
    ("How many days per week can employees work remotely?", "remote_work_policy.txt"),
    ("What is the deadline for submitting an expense report?", "expense_policy.txt"),
    ("What is the minimum password length?", "security_policy.txt"),
    ("What is the hotel nightly limit in high-cost cities?", "travel_policy.pdf"),
]


@pytest.mark.parametrize("question,source", CASES)
def test_relevant_source_is_in_top_k(pipeline, question, source):
    result = pipeline.retrieve(question, top_k=4)
    assert result.sufficient
    assert source in [h.chunk.source for h in result.hits]


@pytest.mark.parametrize("question,source", CASES)
def test_best_hit_is_from_the_right_document(pipeline, question, source):
    assert pipeline.retrieve(question, top_k=4).hits[0].chunk.source == source


def test_hits_are_ranked_by_descending_score(pipeline):
    scores = [h.score for h in pipeline.retrieve("remote work equipment allowance", top_k=5).hits]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_top_k_is_respected(pipeline, k):
    assert len(pipeline.retrieve("security incident reporting", top_k=k).hits) == k


def test_top_k_larger_than_index_returns_everything(pipeline):
    assert len(pipeline.retrieve("policy", top_k=10_000).hits) == len(pipeline.chunks)


def test_retrieval_is_deterministic(pipeline):
    q = "How long are backups retained?"
    a = [(h.chunk.chunk_id, h.score) for h in pipeline.retrieve(q).hits]
    b = [(h.chunk.chunk_id, h.score) for h in pipeline.retrieve(q).hits]
    assert a == b


def test_retrieval_latency_is_recorded(pipeline):
    assert pipeline.retrieve("meal limit").latency_ms > 0


def test_empty_question_is_not_sufficient(pipeline):
    result = pipeline.retrieve("   ")
    assert not result.sufficient and result.hits == []


@pytest.mark.semantic
def test_semantic_model_handles_paraphrase(sample_docs):
    """Runs only when the real Sentence Transformers model can be loaded (needs network on first run)."""
    from rag.config import Settings
    from rag.embeddings import get_embedder
    from rag.pipeline import RAGPipeline

    try:
        choice = get_embedder("sentence-transformers")
    except Exception as exc:
        pytest.skip(f"Sentence Transformers model unavailable: {exc}")
    pipe = RAGPipeline(Settings(chunk_size=500, chunk_overlap=80), choice.embedder)
    pipe.ingest(sample_docs)
    hits = pipe.retrieve("What is the minimum password length?", top_k=3).hits
    assert "security_policy.txt" in [h.chunk.source for h in hits]
    # The paraphrase ('length' vs 'characters') must also produce a grounded, cited answer
    resp = pipe.ask("What is the minimum password length?")
    assert not resp.answer.insufficient and "14 characters" in resp.answer.text
    assert pipe.ask("What is the company's parental leave policy?").answer.insufficient
