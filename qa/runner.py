"""Runs the evaluation dataset against a pipeline and computes summary metrics from real measurements."""
from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from rag.answering import AnswerGenerator
from rag.pipeline import RAGPipeline

from . import deterministic as det
from .llm_judge import Completer, JudgeResult, judge_faithfulness

DEFAULT_DATASET = Path(__file__).resolve().parent.parent / "data" / "eval_dataset.json"
REQUIRED_FIELDS = ("id", "question", "answerable")


@dataclass
class CaseResult:
    case_id: str
    question: str
    answerable: bool
    answer: str
    sources: list[str]
    checks: list[det.CheckResult]
    coverage: float
    total_ms: float
    retrieval_ms: float
    top_score: float
    llm_judgement: JudgeResult | None = None
    skipped_reason: str | None = None  # set when the case was not run; never counted as a pass
    contexts: list[str] = field(default_factory=list)  # retrieved chunk texts (input to RAGAS)
    reference: str | None = None  # ground-truth answer from the dataset, if any

    @property
    def skipped(self) -> bool:
        return self.skipped_reason is not None

    @property
    def passed(self) -> bool:
        """Deterministic verdict only. The optional LLM judgement never gates this."""
        return not self.skipped and all(c.passed for c in self.checks)


@dataclass
class EvalReport:
    results: list[CaseResult] = field(default_factory=list)
    embedder: str = ""
    generator_mode: str = ""

    @property
    def summary(self) -> dict:
        ran = [r for r in self.results if not r.skipped]
        n = len(ran)
        skipped = len(self.results) - n
        if n == 0:
            return {"cases": 0, "skipped": skipped}
        answerable = [r for r in ran if r.answerable]
        unanswerable = [r for r in ran if not r.answerable]
        lat = sorted(r.total_ms for r in ran)
        p95 = lat[min(n - 1, max(0, round(0.95 * n) - 1))]
        judged = [r.llm_judgement for r in ran if r.llm_judgement and r.llm_judgement.passed is not None]
        return {
            "cases": n,
            "skipped": skipped,
            "passed": sum(r.passed for r in ran),
            "pass_rate": sum(r.passed for r in ran) / n,
            "mean_evidence_coverage": statistics.mean(r.coverage for r in answerable) if answerable else None,
            "abstention_accuracy": (sum(r.passed for r in unanswerable) / len(unanswerable)) if unanswerable else None,
            "mean_latency_ms": statistics.mean(lat),
            "p95_latency_ms": p95,
            "llm_judged": len(judged),
            "llm_faithful": sum(bool(j.passed) for j in judged),
        }


def load_dataset(path: str | Path = DEFAULT_DATASET) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = data["cases"] if isinstance(data, dict) else data
    ids = set()
    for case in cases:
        missing = [f for f in REQUIRED_FIELDS if f not in case]
        if missing:
            raise ValueError(f"Case {case.get('id', '?')} is missing fields: {missing}")
        if case["id"] in ids:
            raise ValueError(f"Duplicate case id {case['id']}")
        ids.add(case["id"])
    return cases


def run_eval(pipeline: RAGPipeline, cases: list[dict], generator: AnswerGenerator | None = None,
             top_k: int | None = None, latency_budget_ms: float | None = None,
             judge: Completer | None = None) -> EvalReport:
    budget = latency_budget_ms if latency_budget_ms is not None else pipeline.settings.latency_budget_ms
    report = EvalReport(embedder=pipeline.embedder.name,
                        generator_mode=type(generator).__name__ if generator else "ExtractiveAnswerer")
    semantic = getattr(pipeline.embedder, "semantic", False)
    for case in cases:
        if case.get("requires_semantic") and not semantic:
            report.results.append(CaseResult(
                case_id=case["id"], question=case["question"], answerable=case["answerable"], answer="",
                sources=[], checks=[], coverage=0.0, total_ms=0.0, retrieval_ms=0.0, top_score=0.0,
                skipped_reason="Needs a semantic embedder (active embedder is lexical-only)."))
            continue
        resp = pipeline.ask(case["question"], generator=generator, top_k=top_k)
        evidence = case.get("expected_evidence", [])
        coverage, _ = det.evidence_coverage(evidence, [h.chunk.text for h in resp.retrieval.hits])
        checks = det.run_all(
            resp, answerable=case["answerable"], expected_source=case.get("expected_source"),
            expected_evidence=evidence, answer_must_contain=case.get("answer_must_contain", []),
            latency_budget_ms=budget,
        )
        report.results.append(CaseResult(
            case_id=case["id"], question=case["question"], answerable=case["answerable"],
            answer=resp.answer.text, sources=resp.sources, checks=checks, coverage=coverage,
            total_ms=resp.total_ms, retrieval_ms=resp.retrieval_ms, top_score=resp.retrieval.top_score,
            llm_judgement=judge_faithfulness(judge, resp) if judge else None,
            contexts=[h.chunk.text for h in resp.retrieval.hits], reference=case.get("reference"),
        ))
    return report
