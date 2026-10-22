"""Tests for the optional RAGAS integration.

No real LLM is called. A stub judge reads the JSON schema that RAGAS puts in every prompt and answers it with
all-"yes" (1) or all-"no" (0) verdicts. That proves the *plumbing*: adapters, threading, parsing, mapping of the
judge's verdicts to scores. It says nothing about how good Claude would be as a judge.
"""
import json
import re
import threading

import pytest

from qa.ragas_eval import (RagasUnavailable, exclusion_reason, ragas_available, run_ragas, strip_citations)
from qa.runner import CaseResult, EvalReport, load_dataset, run_eval

ok, _reason = ragas_available()
needs_ragas = pytest.mark.skipif(not ok, reason=f"RAGAS not importable: {_reason}")

_SCHEMA = re.compile(r"JSON Schema:\s*(\{.*?\})\s*Do not use single quotes", re.S)


def _fabricate(schema, defs, verdict):
    if "$ref" in schema:
        return _fabricate(defs[schema["$ref"].split("/")[-1]], defs, verdict)
    kind = schema.get("type")
    if kind == "object":
        return {k: _fabricate(v, defs, verdict) for k, v in schema.get("properties", {}).items()}
    if kind == "array":
        return [_fabricate(schema["items"], defs, verdict)]
    if kind == "integer":
        return verdict
    if kind == "number":
        return float(verdict)
    if kind == "boolean":
        return bool(verdict)
    return "stub"


class SchemaStubJudge:
    """Answers any RAGAS prompt with a schema-valid JSON in which every verdict is `verdict` (1 or 0)."""

    def __init__(self, verdict):
        self.verdict, self.calls = verdict, 0

    def complete(self, system, user):
        self.calls += 1
        match = _SCHEMA.search(user)
        assert match, "RAGAS prompt without a JSON schema"
        schema = json.loads(match.group(1))
        return json.dumps(_fabricate(schema, schema.get("$defs", {}), self.verdict))


class BrokenJudge:
    def complete(self, system, user):
        raise ConnectionError("API down")


def small_report(pipeline, n=2):
    return run_eval(pipeline, [c for c in load_dataset() if c["answerable"]][:n])


# ------------------------------------------------------------------------------ no RAGAS needed
def test_strip_citations_removes_only_the_markers():
    assert strip_citations("Up to 3 days. [remote_work_policy.txt · chunk-0001]\nMore. [a.pdf · chunk-0002]") == \
        "Up to 3 days.\nMore."


def _case(**kw):
    base = dict(case_id="X", question="q", answerable=True, answer="An answer.", sources=[], checks=[],
                coverage=1.0, total_ms=1.0, retrieval_ms=1.0, top_score=0.5, contexts=["ctx"], reference="ref")
    base.update(kw)
    return CaseResult(**base)


@pytest.mark.parametrize("kw,expected", [
    ({}, None),
    ({"skipped_reason": "needs semantic"}, "skipped"),
    ({"answerable": False}, "unanswerable"),
    ({"reference": None}, "no reference"),
    ({"contexts": []}, "nothing was retrieved"),
    ({"answer": "I could not find enough evidence in the uploaded documents to answer this question."}, "abstained"),
])
def test_only_answerable_answered_cases_are_sent_to_ragas(kw, expected):
    reason = exclusion_reason(_case(**kw))
    assert (reason is None) if expected is None else (expected in reason)


def test_dataset_references_exist_for_every_answerable_case():
    for c in load_dataset():
        assert bool(c.get("reference")) == c["answerable"], c["id"]


def test_runner_keeps_contexts_and_reference(pipeline):
    r = small_report(pipeline, 1).results[0]
    assert r.contexts and r.reference


def test_unavailable_ragas_raises_clear_error(pipeline, monkeypatch):
    monkeypatch.setattr("qa.ragas_eval.ragas_available", lambda: (False, "blocked by policy"))
    with pytest.raises(RagasUnavailable, match="blocked by policy"):
        run_ragas(small_report(pipeline, 1), SchemaStubJudge(1), pipeline.embedder)


def test_report_with_nothing_eligible_makes_no_judge_calls(pipeline):
    report = run_eval(pipeline, [c for c in load_dataset() if not c["answerable"]][:2])
    judge = SchemaStubJudge(1)
    out = run_ragas(report, judge, pipeline.embedder) if ok else None
    if ok:
        assert judge.calls == 0 and out.scored_cases == 0 and out.notes


# ------------------------------------------------------------------------------ with RAGAS (stub judge)
@needs_ragas
def test_judge_verdicts_flow_through_to_scores_all_yes(pipeline):
    out = run_ragas(small_report(pipeline), SchemaStubJudge(1), pipeline.embedder)
    assert out.kind == "ragas-llm" and out.scored_cases == 2
    for case in out.cases:
        for metric in ("faithfulness", "llm_context_precision_with_reference", "context_recall"):
            assert case.scores[metric] == pytest.approx(1.0), (case.case_id, metric)


@needs_ragas
def test_judge_verdicts_flow_through_to_scores_all_no(pipeline):
    out = run_ragas(small_report(pipeline), SchemaStubJudge(0), pipeline.embedder)
    for case in out.cases:
        for metric in ("faithfulness", "llm_context_precision_with_reference", "context_recall"):
            assert case.scores[metric] == pytest.approx(0.0), (case.case_id, metric)
    assert all(v == pytest.approx(0.0) for v in out.means.values() if v is not None)


@needs_ragas
def test_lexical_embedder_skips_answer_relevancy_with_a_note(pipeline):
    out = run_ragas(small_report(pipeline, 1), SchemaStubJudge(1), pipeline.embedder)
    assert "answer_relevancy" not in out.metrics_run
    assert any("Answer relevancy skipped" in n for n in out.notes)


@needs_ragas
def test_judge_failure_gives_none_scores_not_fake_numbers(pipeline):
    out = run_ragas(small_report(pipeline, 1), BrokenJudge(), pipeline.embedder)
    assert all(v is None for c in out.cases for v in c.scores.values())
    assert any("could not be computed" in n for n in out.notes)
    assert all(v is None for v in out.means.values())


@needs_ragas
def test_runs_from_a_non_main_thread_like_streamlit(pipeline):
    box = {}
    t = threading.Thread(target=lambda: box.update(out=run_ragas(small_report(pipeline, 1), SchemaStubJudge(1),
                                                                 pipeline.embedder)))
    t.start()
    t.join(timeout=240)
    assert not t.is_alive() and box["out"].scored_cases == 1


@needs_ragas
@pytest.mark.semantic
def test_answer_relevancy_runs_with_a_semantic_embedder(sample_docs):
    from rag.config import Settings
    from rag.embeddings import get_embedder
    from rag.pipeline import RAGPipeline

    try:
        emb = get_embedder("sentence-transformers").embedder
    except Exception as exc:
        pytest.skip(f"semantic model unavailable: {exc}")
    pipe = RAGPipeline(Settings(), emb)
    pipe.ingest(sample_docs)
    out = run_ragas(small_report(pipe, 1), SchemaStubJudge(1), emb)
    assert "answer_relevancy" in out.metrics_run
    assert isinstance(out.cases[0].scores["answer_relevancy"], float)
