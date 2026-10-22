"""Answer generation contracts, the no-LLM extractive answerer, and a lexical grounding check."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

from .config import INSUFFICIENT_EVIDENCE_MESSAGE
from .retriever import RetrievedChunk
from .textutils import content_terms, split_sentences

_CITATION = re.compile(r"\[[^\]]*chunk-\d+\]")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


@dataclass
class Answer:
    text: str
    mode: str  # "extractive" | "llm" | "none"
    citations: list[str] = field(default_factory=list)  # chunk_ids
    insufficient: bool = False


def insufficient_answer() -> Answer:
    return Answer(INSUFFICIENT_EVIDENCE_MESSAGE, mode="none", insufficient=True)


class AnswerGenerator(Protocol):
    """Anything that can turn retrieved evidence into an answer (extractive, Claude, a test stub...)."""

    def generate(self, question: str, hits: list[RetrievedChunk]) -> Answer: ...


def citation_label(hit: RetrievedChunk) -> str:
    return f"[{hit.chunk.source} · {hit.chunk.chunk_id.split('::')[-1]}]"


class ExtractiveAnswerer:
    """Demo / retrieval-only mode: quotes the best-matching sentences from the evidence.

    It cannot invent text because every sentence it returns is copied verbatim from a retrieved chunk.
    """

    def __init__(self, embedder=None, max_sentences: int = 3, min_overlap: float = 0.34,
                 sentence_min_score: float = 0.50):
        # 0.50 was chosen from the score gap observed with all-MiniLM-L6-v2 on the 17 bundled eval cases
        # (answerable best-sentence cosine 0.65-0.91, unanswerable <= 0.44). Re-calibrate for other models/corpora.
        self.embedder = embedder
        self.max_sentences = max_sentences
        self.min_overlap = min_overlap  # lexical mode: share of question terms found in the sentence
        self.sentence_min_score = sentence_min_score  # semantic mode: cosine(question, sentence)

    def _score_sentences(self, question: str, sentences: list[str]) -> list[float]:
        """Relevance of each sentence to the question, in [0, 1]-ish. Semantic when a semantic
        embedder is available, otherwise lexical term overlap."""
        if self.embedder is not None and getattr(self.embedder, "semantic", False):
            vectors = self.embedder.embed([question] + sentences)
            return [float(v @ vectors[0]) for v in vectors[1:]]
        q_terms = set(content_terms(question))
        return [len(q_terms & set(content_terms(s))) / len(q_terms) for s in sentences]

    def generate(self, question: str, hits: list[RetrievedChunk]) -> Answer:
        if not content_terms(question) or not hits:
            return insufficient_answer()
        semantic = self.embedder is not None and getattr(self.embedder, "semantic", False)
        threshold = self.sentence_min_score if semantic else self.min_overlap
        flat: list[tuple[int, int, str, RetrievedChunk]] = []
        seen: set[str] = set()
        for hit in hits:
            for pos, sentence in enumerate(split_sentences(hit.chunk.text)):
                key = sentence.lower()
                # Headings / titles / list labels carry no claim: require a sentence-like line.
                if key in seen or len(sentence) < 12 or sentence[-1] not in ".!?":
                    continue
                seen.add(key)
                flat.append((hit.rank, pos, sentence, hit))
        scores = self._score_sentences(question, [f[2] for f in flat]) if flat else []
        candidates = [(s, rank, pos, sentence, hit) for s, (rank, pos, sentence, hit) in zip(scores, flat)
                      if s >= threshold]
        if not candidates:
            return insufficient_answer()
        best = sorted(candidates, key=lambda c: (-c[0], c[1], c[2]))[: self.max_sentences]
        best.sort(key=lambda c: (c[1], c[2]))  # present in document order
        lines, citations = [], []
        for _, _, _, sentence, hit in best:
            lines.append(f"{sentence} {citation_label(hit)}")
            if hit.chunk.chunk_id not in citations:
                citations.append(hit.chunk.chunk_id)
        return Answer("\n".join(lines), mode="extractive", citations=citations)


# --------------------------------------------------------------------------- grounding check
@dataclass
class GroundingResult:
    supported: bool
    sentence_support: list[tuple[str, float]]  # (sentence, fraction of its terms found in evidence)
    unsupported_sentences: list[str]
    unsupported_numbers: list[str]

    @property
    def detail(self) -> str:
        if self.supported:
            return "Every answer sentence and number is lexically supported by the retrieved evidence."
        bits = []
        if self.unsupported_sentences:
            bits.append(f"{len(self.unsupported_sentences)} sentence(s) with low evidence overlap")
        if self.unsupported_numbers:
            bits.append(f"numbers not in evidence: {', '.join(self.unsupported_numbers)}")
        return "; ".join(bits)


def check_grounding(answer_text: str, evidence_texts: list[str], min_support: float = 0.6) -> GroundingResult:
    """Deterministic lexical grounding check (no LLM).

    It catches invented facts and numbers, but a paraphrase that uses different words can score low,
    so treat a failure as 'needs review', and a pass as 'lexically supported' - not proof of truth.
    """
    evidence = " ".join(evidence_texts)
    evidence_terms = set(content_terms(evidence))
    evidence_numbers = {n.replace(",", "") for n in _NUMBER.findall(evidence)}
    cleaned = _CITATION.sub("", answer_text)
    support, bad_sentences = [], []
    for sentence in split_sentences(cleaned):
        terms = content_terms(sentence)
        if not terms:
            continue
        ratio = sum(t in evidence_terms for t in terms) / len(terms)
        support.append((sentence, ratio))
        if ratio < min_support:
            bad_sentences.append(sentence)
    bad_numbers = sorted({n for n in (x.replace(",", "") for x in _NUMBER.findall(cleaned))
                          if n not in evidence_numbers})
    return GroundingResult(
        supported=not bad_sentences and not bad_numbers,
        sentence_support=support,
        unsupported_sentences=bad_sentences,
        unsupported_numbers=bad_numbers,
    )
