import pymupdf
import pytest

from rag.loaders import DocumentLoadError, clean_text, load_document, load_many, read_path


def make_pdf(pages: list[str]) -> bytes:
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


def test_txt_is_loaded_and_cleaned():
    doc = load_document("n.txt", b"Hello   world\r\n\r\n\r\n\r\nSecond  paragraph")
    assert doc.text == "Hello world\n\nSecond paragraph"
    assert doc.page_starts == []


def test_txt_handles_non_utf8_bytes():
    assert "caf" in load_document("n.txt", "café".encode("cp1252")).text


def test_pdf_text_and_pages_are_extracted():
    doc = load_document("d.pdf", make_pdf(["First page text", "Second page text"]))
    assert "First page text" in doc.text and "Second page text" in doc.text
    assert len(doc.page_starts) == 2
    assert doc.page_at(doc.text.index("Second")) == 2


def test_pdf_without_text_is_empty_not_an_error():
    assert load_document("blank.pdf", make_pdf([""])).is_empty


def test_corrupt_pdf_raises_load_error():
    with pytest.raises(DocumentLoadError):
        load_document("bad.pdf", b"this is not a pdf")


def test_unsupported_extension_rejected():
    with pytest.raises(DocumentLoadError, match="Unsupported"):
        load_document("slides.pptx", b"x")


def test_load_many_reports_failures_without_aborting():
    docs, errors = load_many([("ok.txt", b"fine"), ("bad.pdf", b"nope"), ("x.docx", b"x")])
    assert [d.filename for d in docs] == ["ok.txt"]
    assert len(errors) == 2


def test_read_path_single_file_and_quoted_path(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hello", encoding="utf-8")
    files, errors = read_path(f'  "{f}"  ')  # paths pasted from Explorer often carry quotes
    assert files == [("a.txt", b"hello")] and not errors


def test_read_path_folder_reads_only_supported_files(tmp_path):
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    (tmp_path / "a.pdf").write_bytes(make_pdf(["page"]))
    (tmp_path / "skip.docx").write_bytes(b"x")
    files, errors = read_path(str(tmp_path))
    assert [n for n, _ in files] == ["a.pdf", "b.txt"] and not errors


def test_read_path_reports_missing_empty_and_unsupported(tmp_path):
    assert "not found" in read_path(str(tmp_path / "nope"))[1][0]
    assert "No .pdf or .txt" in read_path(str(tmp_path))[1][0]
    assert read_path("   ")[0] == []
    bad = tmp_path / "x.docx"
    bad.write_bytes(b"x")
    assert read_path(str(bad))[0] == [] and "unsupported" in read_path(str(bad))[1][0]


def test_clean_text_rejoins_hyphenated_words():
    assert clean_text("inter-\nnational") == "international"
