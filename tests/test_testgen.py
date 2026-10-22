import json

import pytest

from qa.runner import run_eval
from qa.testgen import (ROW_COLUMNS, case_to_row, cases_from_rows, draft_cases, parse_draft)


# ------------------------------------------------------------------------------ table rows
def row(**kw):
    base = {"Question": "How many leave days?", "Answerable": True, "Reference answer": "25 days.",
            "Evidence phrase": "25 days", "Required fact in answer": "25", "Source file": "hr.pdf"}
    base.update(kw)
    return base


def test_rows_become_cases_with_sequential_ids():
    cases, warnings = cases_from_rows([row(), row(Question="Second?")])
    assert [c["id"] for c in cases] == ["CUS-001", "CUS-002"] and not warnings
    c = cases[0]
    assert (c["expected_source"], c["expected_evidence"], c["answer_must_contain"], c["reference"]) == \
        ("hr.pdf", ["25 days"], ["25"], "25 days.")


def test_blank_rows_and_nan_cells_are_handled():
    nan = float("nan")
    cases, _ = cases_from_rows([
        {c: None for c in ROW_COLUMNS}, row(Question="  "),
        row(**{"Answerable": nan, "Source file": nan, "Evidence phrase": None, "Required fact in answer": nan}),
    ])
    assert len(cases) == 1
    assert cases[0]["answerable"] is True and cases[0]["expected_source"] is None
    assert cases[0]["expected_evidence"] == [] and cases[0]["answer_must_contain"] == []


def test_unanswerable_row_drops_expectations():
    (c,), _ = cases_from_rows([row(Answerable=False)])
    assert c["answerable"] is False
    assert c["expected_source"] is None and c["reference"] is None and c["expected_evidence"] == []


def test_answerable_row_without_reference_warns_that_ragas_will_skip_it():
    cases, warnings = cases_from_rows([row(**{"Reference answer": ""})])
    assert len(cases) == 1 and cases[0]["reference"] is None
    assert "RAGAS will skip" in warnings[0]


def test_case_round_trips_through_a_row():
    (c,), _ = cases_from_rows([row()])
    (again,), _ = cases_from_rows([case_to_row(c)])
    assert {k: v for k, v in again.items() if k != "id"} == {k: v for k, v in c.items() if k != "id"}


def test_custom_cases_run_in_the_normal_evaluation(pipeline):
    cases, _ = cases_from_rows([
        row(Question="What is the daily meal limit when travelling on business?",
            **{"Reference answer": "The daily meal limit is 60 USD.", "Evidence phrase": "60 USD per person",
               "Required fact in answer": "60 USD", "Source file": "expense_policy.txt"}),
        row(Question="What is the capital of France?", Answerable=False),
    ])
    report = run_eval(pipeline, cases)
    assert [r.passed for r in report.results] == [True, True]
    assert report.results[0].reference == "The daily meal limit is 60 USD."


# ------------------------------------------------------------------------------ drafting with a stub model
class Stub:
    def __init__(self, reply_for):
        self.reply_for, self.calls = reply_for, 0

    def complete(self, system, user):
        self.calls += 1
        return self.reply_for(user)


def good_reply(user):
    quote = "The daily meal limit is 60 USD per person"
    return json.dumps({"question": "What is the daily meal limit?", "reference": "It is 60 USD per person.",
                       "evidence": quote})


def test_parse_draft_handles_wrapped_and_broken_output():
    assert parse_draft('Sure:\n```json\n{"question":"q","reference":"r","evidence":"e"}\n```') == \
        {"question": "q", "reference": "r", "evidence": "e"}
    assert parse_draft("no json") is None
    assert parse_draft('{"question":"q"}') is None
    assert parse_draft("[1, 2]") is None


def test_draft_keeps_only_verified_quotes(pipeline):
    meal_chunks = [c for c in pipeline.chunks if "daily meal limit" in c.text]
    assert meal_chunks
    result = draft_cases(Stub(good_reply), meal_chunks, n=1)
    assert len(result.cases) == 1 and not result.dropped
    case = result.cases[0]
    assert case["id"] == "GEN-001" and case["generated"] and case["expected_source"] == "expense_policy.txt"


def test_draft_drops_a_quote_that_is_not_in_the_chunk(pipeline):
    chunk = next(c for c in pipeline.chunks if "daily meal limit" in c.text)
    fabricated = Stub(lambda u: json.dumps({"question": "Q?", "reference": "R.", "evidence": "invented quote"}))
    result = draft_cases(fabricated, [chunk], n=1)
    assert result.cases == [] and "not word-for-word" in result.dropped[0]


@pytest.mark.parametrize("reply,reason", [
    ("garbage", "unparseable"),
    (json.dumps({"question": "What does the passage say?", "reference": "r", "evidence": "The daily meal limit"}),
     "passage"),
])
def test_draft_drops_unusable_replies(pipeline, reply, reason):
    chunk = next(c for c in pipeline.chunks if "daily meal limit" in c.text)
    result = draft_cases(Stub(lambda u: reply), [chunk], n=1)
    assert result.cases == [] and reason in result.dropped[0]


def test_draft_survives_model_errors_and_short_chunks(pipeline):
    class Down:
        def complete(self, system, user):
            raise ConnectionError("API down")

    chunk = next(c for c in pipeline.chunks if "daily meal limit" in c.text)
    assert "model call failed" in draft_cases(Down(), [chunk], n=1).dropped[0]
    short = type(chunk)("s.txt::chunk-0000", "s.txt", "tiny", 0, 4)
    assert "long enough" in draft_cases(Stub(good_reply), [short], n=1).dropped[0]


def test_draft_sampling_is_seeded_and_bounded(pipeline):
    stub = Stub(good_reply)
    draft_cases(stub, pipeline.chunks, n=3, seed=1)
    assert stub.calls == 3
    seen = []
    for _ in range(2):
        s = Stub(lambda u, seen=seen: (seen.append(u), good_reply(u))[1])
        draft_cases(s, pipeline.chunks, n=3, seed=7)
    assert seen[:3] == seen[3:]  # same seed -> same chunks -> same prompts
