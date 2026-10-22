"""Small shared text helpers (tokenising, sentence splitting) used by retrieval, answering and QA."""
from __future__ import annotations

import re

STOPWORDS = frozenset(
    """a an and are as at be been but by can could did do does for from had has have how i if in into is it its
    me my of on or our should so than that the their them then there these they this to us was we were what when
    where which who whom why will with would you your about any per up out also""".split()
)

_WORD = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")


def stem(token: str) -> str:
    """Very light suffix stripping so 'days'/'day' and 'approved'/'approve' line up."""
    for suffix in ("ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]
    return token


def content_terms(text: str) -> list[str]:
    """Lower-cased, stopword-free, lightly stemmed terms."""
    return [stem(t) for t in _WORD.findall(text.lower()) if t not in STOPWORDS]


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p and p.strip()]


def md_escape(text: str) -> str:
    """Escape text so a Markdown renderer shows it literally (and keeps its line breaks).

    Document text rendered unescaped would misrepresent the evidence: '3. Heading' becomes a list item,
    '*x*' italics, '$..$' maths, and single newlines collapse.
    """
    escaped = re.sub(r"([\\`*_{}\[\]<>#|~$])", r"\\\1", text)
    escaped = re.sub(r"(?m)^(\s*\d+)\.", r"\1\\.", escaped)
    return escaped.replace("\n", "  \n")


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()
