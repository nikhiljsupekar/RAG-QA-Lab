"""Deterministic (rule-based, reproducible, no LLM) checks.

Every check returns a CheckResult with kind="deterministic". Same input -> same output.
"""
from __future__ import annotations

from dataclasses import dataclass

from rag.answering import check_grounding
from rag.config import INSUFFICIENT_EVIDENCE_MESSAGE
from rag.pipeline import RAGResponse
from rag.textutils import normalise

KIND = "deterministic"


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str
    kind: str = KIND


def evidence_coverage(expected_evidence: list[str], retrieved_texts: list[str]) -> tuple[float, list[str]]:
    """Fraction of expected evidence phrases found verbatim (whitespace/case-insensitive) in the retrieved
    chunks, plus the list of missing phrases. 1.0 when nothing is expected."""
    if not expected_evidence:
        return 1.0, []
    haystack = normalise(" ".join(retrieved_texts))
    missing = [p for p in expected_evidence if normalise(p) not in haystack]
    return (len(expected_evidence) - len(missing)) / len(expected_evidence), missing


def check_retrieval_source(resp: RAGResponse, expected_source: str | None) -> CheckResult:
    if not expected_source:
        return CheckResult("source_retrieved", True, "No source expected (unanswerable case).")
    found = expected_source in resp.sources
    return CheckResult("source_retrieved", found,
                       f"Expected '{expected_source}' in top-K sources; got {resp.sources or 'none'}.")


def check_evidence_coverage(resp: RAGResponse, expected_evidence: list[str], min_coverage: float = 1.0) -> CheckResult:
    cov, missing = evidence_coverage(expected_evidence, [h.chunk.text for h in resp.retrieval.hits])
    return CheckResult("evidence_coverage", cov >= min_coverage,
                       f"Coverage {cov:.0%}" + (f"; missing: {missing}" if missing else ""))


def check_answer_facts(resp: RAGResponse, must_contain: list[str]) -> CheckResult:
    if not must_contain:
        return CheckResult("answer_contains_facts", True, "No facts required.")
    text = normalise(resp.answer.text)
    missing = [f for f in must_contain if normalise(f) not in text]
    return CheckResult("answer_contains_facts", not missing,
                       "All required facts present." if not missing else f"Missing in answer: {missing}")


def check_refusal(resp: RAGResponse, answerable: bool) -> CheckResult:
    refused = resp.answer.insufficient and resp.answer.text == INSUFFICIENT_EVIDENCE_MESSAGE
    if answerable:
        return CheckResult("correct_abstention", not refused,
                           "Answered as expected." if not refused else "Wrongly refused an answerable question.")
    return CheckResult("correct_abstention", refused,
                       "Correctly declined to answer." if refused else "Answered a question the documents cannot answer.")


def check_source_attribution(resp: RAGResponse) -> CheckResult:
    """Every cited chunk must be one of the retrieved chunks, and a non-refusal answer must cite something."""
    if resp.answer.insufficient:
        return CheckResult("source_attribution", True, "No claims made, nothing to attribute.")
    retrieved = {h.chunk.chunk_id for h in resp.retrieval.hits}
    cited = resp.answer.citations
    if not cited:
        return CheckResult("source_attribution", False, "Answer cites no sources.")
    unknown = [c for c in cited if c not in retrieved]
    return CheckResult("source_attribution", not unknown,
                       f"{len(cited)} citation(s), all retrieved." if not unknown else f"Cited chunks not retrieved: {unknown}")


def check_grounded(resp: RAGResponse) -> CheckResult:
    if resp.answer.insufficient:
        return CheckResult("grounded_in_evidence", True, "No claims made.")
    g = check_grounding(resp.answer.text, [h.chunk.text for h in resp.retrieval.hits])
    return CheckResult("grounded_in_evidence", g.supported, g.detail)


def check_latency(resp: RAGResponse, budget_ms: float) -> CheckResult:
    return CheckResult("latency_budget", resp.total_ms <= budget_ms,
                       f"{resp.total_ms:.1f} ms vs budget {budget_ms:.0f} ms")


def run_all(resp: RAGResponse, *, answerable: bool, expected_source: str | None,
            expected_evidence: list[str], answer_must_contain: list[str],
            latency_budget_ms: float) -> list[CheckResult]:
    checks = [check_refusal(resp, answerable)]
    if answerable:
        checks += [
            check_retrieval_source(resp, expected_source),
            check_evidence_coverage(resp, expected_evidence),
            check_answer_facts(resp, answer_must_contain),
            check_source_attribution(resp),
        ]
    checks += [check_grounded(resp), check_latency(resp, latency_budget_ms)]
    return checks
