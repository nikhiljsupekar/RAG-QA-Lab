"""Document loading: PDF (PyMuPDF) and TXT extraction plus text cleaning."""
from __future__ import annotations

import bisect
import re
from dataclasses import dataclass, field
from pathlib import PurePath

SUPPORTED_EXTENSIONS = (".pdf", ".txt")


class DocumentLoadError(Exception):
    """Raised when a file cannot be turned into text."""


@dataclass
class Document:
    filename: str
    text: str
    # Character offset at which each page starts in `text` (PDFs only).
    page_starts: list[int] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()

    def page_at(self, offset: int) -> int | None:
        """1-based page number containing `offset`, or None for non-paged documents."""
        if not self.page_starts:
            return None
        return max(bisect.bisect_right(self.page_starts, offset), 1)


def clean_text(text: str) -> str:
    """Normalise whitespace while keeping paragraph breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = text.replace("­", "")  # soft hyphen
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # re-join words hyphenated across lines
    text = re.sub(r"[ \t\f\v ]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def unwrap_lines(text: str) -> str:
    """Join hard-wrapped lines (typical of PDF extraction) back into sentences.

    A line break is kept after a short line (a heading), after terminal punctuation, or before a
    numbered heading; otherwise the line is considered a mid-sentence wrap and joined with a space.
    """
    out: list[str] = []
    for line in text.split("\n"):
        if out and out[-1] and line and len(out[-1]) >= 50 \
                and out[-1][-1] not in ".!?:" and not re.match(r"\d+\.\s", line):
            out[-1] += " " + line
        else:
            out.append(line)
    return "\n".join(out)


def extract_txt(data: bytes) -> str:
    # UTF-16 decodes almost any even-length byte string, so only trust it when a BOM says so.
    encodings = ["utf-8-sig"]
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        encodings.append("utf-16")
    encodings.append("cp1252")
    for encoding in encodings:
        try:
            return data.decode(encoding)
        except UnicodeError:
            continue
    return data.decode("utf-8", errors="replace")


def extract_pdf(data: bytes) -> tuple[str, list[int]]:
    """Return (text, page_start_offsets). Raises DocumentLoadError for unreadable PDFs."""
    try:
        import pymupdf  # PyMuPDF
    except ImportError as exc:  # pragma: no cover
        raise DocumentLoadError("PyMuPDF is not installed (pip install PyMuPDF).") from exc
    try:
        pdf = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise DocumentLoadError(f"Could not open PDF: {exc}") from exc
    try:
        if pdf.needs_pass:
            raise DocumentLoadError("PDF is password protected.")
        parts: list[str] = []
        starts: list[int] = []
        offset = 0
        for page in pdf:
            page_text = clean_text(unwrap_lines(clean_text(page.get_text("text"))))
            starts.append(offset)
            parts.append(page_text)
            offset += len(page_text) + 2  # the "\n\n" joiner below
        return "\n\n".join(parts), starts
    except DocumentLoadError:
        raise
    except Exception as exc:
        raise DocumentLoadError(f"Could not read PDF text: {exc}") from exc
    finally:
        pdf.close()


def load_document(filename: str, data: bytes) -> Document:
    name = PurePath(filename).name
    ext = PurePath(name).suffix.lower()
    if ext == ".pdf":
        text, starts = extract_pdf(data)
        return Document(name, text, starts)
    if ext == ".txt":
        return Document(name, clean_text(extract_txt(data)))
    raise DocumentLoadError(
        f"Unsupported file type '{ext or 'none'}' for {name}. Supported: {', '.join(SUPPORTED_EXTENSIONS)}."
    )


def read_path(path_text: str, limit: int = 200) -> tuple[list[tuple[str, bytes]], list[str]]:
    """Read one file, or every .pdf/.txt directly inside a folder, from the local disk.

    Returns ([(filename, bytes)], errors). Meant for a local single-user app; callers that run on a
    shared server should not expose it (see Settings.allow_local_paths).
    """
    from pathlib import Path

    text = path_text.strip().strip('"').strip("'")
    if not text:
        return [], ["Enter a file or folder path."]
    p = Path(text).expanduser()
    if not p.exists():
        return [], [f"Path not found: {p}"]
    if p.is_file():
        candidates = [p]
    else:
        candidates = sorted(f for f in p.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS)
        if not candidates:
            return [], [f"No .pdf or .txt files in {p}"]
        if len(candidates) > limit:
            candidates = candidates[:limit]
    files, errors = [], []
    for f in candidates:
        if f.suffix.lower() not in SUPPORTED_EXTENSIONS:
            errors.append(f"{f.name}: unsupported file type (supported: {', '.join(SUPPORTED_EXTENSIONS)})")
            continue
        try:
            files.append((f.name, f.read_bytes()))
        except OSError as exc:
            errors.append(f"{f.name}: {exc}")
    return files, errors


def load_many(files: list[tuple[str, bytes]]) -> tuple[list[Document], list[str]]:
    """Load several files; failures are collected as messages instead of aborting the batch."""
    docs: list[Document] = []
    errors: list[str] = []
    for filename, data in files:
        try:
            docs.append(load_document(filename, data))
        except DocumentLoadError as exc:
            errors.append(f"{filename}: {exc}")
    return docs, errors
