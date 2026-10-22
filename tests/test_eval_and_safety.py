import json
import re
from pathlib import Path

import pytest

from llm.claude_provider import ClaudeAnswerer, LLMUnavailable, claude_available
from qa.llm_judge import judge_faithfulness, parse_verdict
from qa.runner import DEFAULT_DATASET, load_dataset, run_eval
from rag.answering import Answer

ROOT = Path(__file__).resolve().parent.parent


# ----------------------------------------------------------------------------- dataset / runner
def test_dataset_is_valid_and_references_real_documents(sample_docs):
    cases = load_dataset(DEFAULT_DATASET)
    names = {d.filename for d in sample_docs}
    assert len(cases) >= 10
    assert any(not c["answerable"] for c in cases) and any(c["answerable"] for c in cases)
    for c in cases:
        if c["answerable"]:
            assert c["expected_source"] in names and c["expected_evidence"]


def test_dataset_evidence_phrases_really_exist_in_source_documents(sample_docs):
    """Guards against a dataset that cannot possibly be satisfied."""
    by_name = {d.filename: " ".join(d.text.lower().split()) for d in sample_docs}
    for c in load_dataset():
        for phrase in c["expected_evidence"]:
            assert phrase.lower() in by_name[c["expected_source"]], (c["id"], phrase)


def test_full_eval_passes_on_sample_documents(pipeline):
    report = run_eval(pipeline, load_dataset())
    ran = [r for r in report.results if not r.skipped]
    failed = [(r.case_id, [c.name for c in r.checks if not c.passed]) for r in ran if not r.passed]
    assert not failed, failed
    # The lexical fallback cannot match paraphrases: those cases must be SKIPPED, never counted as passes.
    assert [r.case_id for r in report.results if r.skipped] == ["SEC-01P"]
    s = report.summary
    assert s["pass_rate"] == 1.0 and s["abstention_accuracy"] == 1.0 and s["skipped"] == 1
    assert s["cases"] == len(report.results) - 1


def test_eval_detects_failures_instead_of_always_passing(pipeline):
    bad = [{"id": "X-1", "question": "What is the minimum number of characters in a password?", "answerable": True,
            "expected_source": "expense_policy.txt", "expected_evidence": ["a phrase that is nowhere"],
            "answer_must_contain": ["999"]},
           # labelled unanswerable although the documents DO answer it -> the system answers -> must fail
           {"id": "X-2", "question": "What is the minimum number of characters in a password?", "answerable": False}]
    report = run_eval(pipeline, bad)
    assert [r.passed for r in report.results] == [False, False]
    assert report.summary["pass_rate"] == 0.0


def test_summary_metrics_are_measured_not_constants(pipeline):
    s = run_eval(pipeline, load_dataset()).summary
    assert s["mean_latency_ms"] > 0 and s["p95_latency_ms"] >= s["mean_latency_ms"] * 0.5
    assert s["llm_judged"] == 0  # no judge supplied -> no LLM scores are reported


def test_load_dataset_rejects_missing_fields_and_duplicates(tmp_path):
    p = tmp_path / "d.json"
    p.write_text(json.dumps({"cases": [{"id": "a", "question": "q"}]}))
    with pytest.raises(ValueError, match="missing"):
        load_dataset(p)
    p.write_text(json.dumps({"cases": [{"id": "a", "question": "q", "answerable": True}] * 2}))
    with pytest.raises(ValueError, match="Duplicate"):
        load_dataset(p)


# ----------------------------------------------------------------------------- LLM judge (stubbed)
class StubJudge:
    def __init__(self, reply):
        self.reply, self.calls = reply, 0

    def complete(self, system, user):
        self.calls += 1
        return self.reply


def test_llm_judge_is_labelled_and_separate_from_deterministic_verdict(pipeline):
    judge = StubJudge('{"faithful": false, "rationale": "stub says no"}')
    report = run_eval(pipeline, load_dataset()[:2], judge=judge)
    for r in report.results:
        assert r.llm_judgement.kind == "llm"
        assert r.passed  # deterministic verdict unaffected by the (stubbed) LLM opinion
    assert report.summary["llm_judged"] == 2 and report.summary["llm_faithful"] == 0


def test_judge_skips_abstentions(pipeline):
    resp = pipeline.ask("What is the capital of France?")
    judge = StubJudge('{"faithful": true, "rationale": ""}')
    assert judge_faithfulness(judge, resp).passed is None and judge.calls == 0


def test_judge_parsing_handles_garbage():
    assert parse_verdict("no json here").passed is None
    assert parse_verdict('Sure! {"faithful": true, "rationale": "ok"}').passed is True


# ----------------------------------------------------------------------------- Claude provider / secrets
def test_claude_is_unavailable_without_a_key(monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    assert not claude_available()
    from llm.claude_provider import ClaudeClient
    with pytest.raises(LLMUnavailable):
        ClaudeClient()


def test_claude_answerer_maps_sentinel_and_citations(pipeline):
    hits = pipeline.retrieve("What is the minimum password length?").hits

    class Client:
        def __init__(self, text): self.text = text
        def complete(self, system, user): return self.text

    assert ClaudeAnswerer(Client("INSUFFICIENT_EVIDENCE")).generate("q", hits).insufficient
    label = f"{hits[0].chunk.source} · {hits[0].chunk.chunk_id.split('::')[-1]}"
    ans = ClaudeAnswerer(Client(f"14 characters. [{label}]")).generate("q", hits)
    assert isinstance(ans, Answer) and ans.mode == "llm" and ans.citations == [hits[0].chunk.chunk_id]
    fake = ClaudeAnswerer(Client("14 characters. [ghost.txt · chunk-0009]")).generate("q", hits)
    assert fake.citations == []


def test_retrieval_package_does_not_import_llm_code():
    for path in (ROOT / "rag").glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"^\s*(from|import)\s+(llm|anthropic)\b", text, re.M), path.name


def test_no_api_keys_are_hardcoded():
    pattern = re.compile(r"sk-ant-[A-Za-z0-9_\-]{10,}")
    for path in ROOT.rglob("*"):
        if path.is_file() and path.suffix in {".py", ".md", ".txt", ".json", ".example", ".ini", ".toml"} \
                and ".venv" not in path.parts and ".pytest_cache" not in path.parts:
            assert not pattern.search(path.read_text(encoding="utf-8", errors="ignore")), path


def test_env_files_are_gitignored():
    ignore = (ROOT / ".gitignore").read_text()
    assert re.search(r"^\.env$", ignore, re.M)
    assert (ROOT / ".env.example").exists()
