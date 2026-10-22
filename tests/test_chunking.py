import pytest

from rag.chunking import chunk_document, chunk_text
from rag.loaders import Document

TEXT = ("Sentence number one is here. " * 60).strip()


def test_chunks_respect_size_limit():
    chunks = chunk_text(TEXT, "a.txt", chunk_size=200, overlap=40)
    assert len(chunks) > 1
    assert all(len(c.text) <= 200 for c in chunks)


def test_consecutive_chunks_overlap():
    chunks = chunk_text(TEXT, "a.txt", chunk_size=200, overlap=60)
    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt.start < prev.end, "chunks should overlap"


def test_zero_overlap_does_not_overlap():
    chunks = chunk_text(TEXT, "a.txt", chunk_size=200, overlap=0)
    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt.start >= prev.end


def test_offsets_point_back_to_source_text():
    for c in chunk_text(TEXT, "a.txt", chunk_size=150, overlap=30):
        assert TEXT[c.start:c.end] == c.text


def test_chunks_cover_whole_document():
    chunks = chunk_text(TEXT, "a.txt", chunk_size=150, overlap=30)
    assert chunks[0].start == 0
    assert chunks[-1].end == len(TEXT)


def test_chunk_ids_unique_and_include_source():
    chunks = chunk_text(TEXT, "policy.txt", chunk_size=150, overlap=30)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert all(i.startswith("policy.txt::chunk-") for i in ids)


def test_text_without_whitespace_still_terminates():
    chunks = chunk_text("x" * 1000, "a.txt", chunk_size=100, overlap=20)
    assert len(chunks) >= 10
    assert chunks[-1].end == 1000


def test_short_text_is_single_chunk():
    chunks = chunk_text("Short text.", "a.txt", chunk_size=800, overlap=100)
    assert [c.text for c in chunks] == ["Short text."]


@pytest.mark.parametrize("text", ["", "   \n\n\t "])
def test_empty_text_gives_no_chunks(text):
    assert chunk_text(text, "a.txt") == []


@pytest.mark.parametrize("size,overlap", [(0, 0), (-5, 0), (100, -1), (100, 100), (100, 150)])
def test_invalid_parameters_rejected(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("hello world", "a.txt", size, overlap)


def test_page_numbers_follow_page_starts():
    doc = Document("p.pdf", "A" * 100 + "\n\n" + "B" * 100, page_starts=[0, 102])
    chunks = chunk_document(doc, chunk_size=60, overlap=10)
    first_b = next(c for c in chunks if c.text.startswith("B"))
    assert chunks[0].page == 1
    assert first_b.page == 2
