from rag.config import INSUFFICIENT_EVIDENCE_MESSAGE, Settings
from rag.embeddings import HashingEmbedder
from rag.loaders import Document
from rag.pipeline import RAGPipeline


def new_pipeline():
    return RAGPipeline(Settings(embedding_backend="hashing"), HashingEmbedder())


def test_ask_before_any_ingest_is_a_clean_refusal():
    resp = new_pipeline().ask("What is the policy?")
    assert resp.answer.insufficient
    assert resp.answer.text == INSUFFICIENT_EVIDENCE_MESSAGE
    assert resp.retrieval.hits == []


def test_ingest_with_no_documents():
    pipe = new_pipeline()
    report = pipe.ingest([])
    assert report.chunk_count == 0 and not pipe.is_ready
    assert pipe.ask("anything").answer.insufficient


def test_empty_and_whitespace_documents_are_skipped_with_reason():
    pipe = new_pipeline()
    report = pipe.ingest([Document("empty.txt", ""), Document("blank.txt", "  \n\n ")])
    assert report.documents_indexed == []
    assert len(report.documents_skipped) == 2
    assert all("no extractable text" in s for s in report.documents_skipped)
    assert pipe.ask("anything at all").answer.text == INSUFFICIENT_EVIDENCE_MESSAGE


def test_empty_documents_do_not_block_valid_ones():
    pipe = new_pipeline()
    report = pipe.ingest([Document("empty.txt", ""),
                          Document("rules.txt", "Employees must wear safety helmets on the construction site.")])
    assert report.documents_indexed == ["rules.txt"]
    assert not pipe.ask("Must employees wear safety helmets on site?").answer.insufficient


def test_reingest_replaces_previous_index():
    pipe = new_pipeline()
    pipe.ingest([Document("a.txt", "The vault code rotates every Monday morning.")])
    pipe.ingest([Document("b.txt", "Lunch is served in the cafeteria at noon.")])
    assert {c.source for c in pipe.chunks} == {"b.txt"}
    assert pipe.ask("When does the vault code rotate?").answer.insufficient


def test_empty_question_is_refused(pipeline):
    assert pipeline.ask("").answer.insufficient
