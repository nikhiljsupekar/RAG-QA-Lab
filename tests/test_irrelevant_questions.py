import pytest

from rag.config import INSUFFICIENT_EVIDENCE_MESSAGE

IRRELEVANT = [
    "Who won the football world cup in 2018?",
    "How do I configure a Kubernetes cluster?",
    "What is the capital of France?",
    "Give me a recipe for chocolate cake",
    "What is the salary of the CEO?",
    "What is the company's parental leave policy?",
]


@pytest.mark.parametrize("question", IRRELEVANT)
def test_irrelevant_question_gets_explicit_insufficient_message(pipeline, question):
    resp = pipeline.ask(question)
    assert resp.answer.insufficient
    assert resp.answer.text == INSUFFICIENT_EVIDENCE_MESSAGE
    assert resp.answer.citations == []


def test_generator_is_never_called_without_evidence(pipeline):
    class Exploding:
        def generate(self, question, hits):
            raise AssertionError("generator must not run when evidence is insufficient")

    resp = pipeline.ask("What is the capital of France?", generator=Exploding())
    assert resp.answer.insufficient and resp.generation_ms == 0.0


def test_threshold_is_configurable(pipeline):
    q = "remote work equipment"
    base = pipeline.retrieve(q)
    assert base.sufficient
    pipeline.retriever.min_score = 0.99
    try:
        assert not pipeline.retrieve(q).sufficient
    finally:
        pipeline.retriever.min_score = base.min_score
