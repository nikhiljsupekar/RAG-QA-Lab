from rag.answering import ExtractiveAnswerer
from qa.deterministic import check_source_attribution


def test_every_hit_carries_filename_and_chunk_id(pipeline):
    for hit in pipeline.retrieve("expense receipt required", top_k=4).hits:
        assert hit.chunk.source.endswith((".txt", ".pdf"))
        assert hit.chunk.chunk_id.startswith(hit.chunk.source + "::chunk-")
        assert hit.chunk.text.strip()


def test_answer_citations_are_a_subset_of_retrieved_chunks(pipeline):
    resp = pipeline.ask("How quickly must a security incident be reported?")
    assert not resp.answer.insufficient
    retrieved = {h.chunk.chunk_id for h in resp.retrieval.hits}
    assert resp.answer.citations and set(resp.answer.citations) <= retrieved


def test_answer_text_is_copied_from_the_cited_chunk(pipeline):
    resp = pipeline.ask("What is the daily meal limit?")
    cited_text = " ".join(h.chunk.text for h in resp.retrieval.hits if h.chunk.chunk_id in resp.answer.citations)
    assert "60 USD" in resp.answer.text and "60 USD" in cited_text


def test_answer_names_the_source_file(pipeline):
    resp = pipeline.ask("What is the minimum number of characters in a password?")
    assert "security_policy.txt" in resp.answer.text
    assert resp.sources[0] == "security_policy.txt"


def test_pdf_hits_carry_page_numbers(pipeline):
    hits = pipeline.retrieve("hotel limit per night", top_k=3).hits
    pdf_pages = [h.chunk.page for h in hits if h.chunk.source == "travel_policy.pdf"]
    assert pdf_pages and all(p in (1, 2) for p in pdf_pages)


def test_attribution_check_rejects_citation_of_unretrieved_chunk(pipeline):
    resp = pipeline.ask("What is the minimum number of characters in a password?")
    resp.answer.citations = ["ghost.txt::chunk-0042"]
    assert not check_source_attribution(resp).passed


def test_attribution_check_rejects_uncited_answer(pipeline):
    resp = pipeline.ask("What is the minimum number of characters in a password?")
    resp.answer.citations = []
    assert not check_source_attribution(resp).passed


def test_extractive_answerer_is_verbatim(pipeline):
    hits = pipeline.retrieve("How many days per week can employees work remotely?").hits
    answer = ExtractiveAnswerer().generate("How many days per week can employees work remotely?", hits)
    evidence = " ".join(h.chunk.text for h in hits)
    for line in answer.text.splitlines():
        sentence = line.rsplit(" [", 1)[0]
        assert sentence in evidence
