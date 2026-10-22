"""OPTIONAL RAGAS evaluation (https://github.com/explodinggradients/ragas).

LLM-BASED and NON-DETERMINISTIC: the scores come from a judge model (Claude) and can change from run to
run. They are reported next to, never mixed into, the deterministic PASS/FAIL of qa.deterministic.

Metrics (per answerable case):
  faithfulness       - are the claims in the answer supported by the retrieved contexts?
  context_precision  - are the useful chunks ranked at the top of the retrieved list? (needs `reference`)
  context_recall     - does the retrieved context contain what the reference answer says? (needs `reference`)
  answer_relevancy   - does the answer address the question? (needs a semantic embedder)

RAGAS is imported lazily, so the rest of the project works when it is not installed. The judge LLM and the
embeddings are plugged in through small adapters around this project's own Claude client and embedder, which
keeps RAGAS from choosing (or requiring) a different provider such as OpenAI.
"""
from __future__ import annotations

import asyncio
import math
import re
import warnings
from dataclasses import dataclass, field

from .runner import CaseResult, EvalReport

KIND = "ragas-llm"
_CITATION = re.compile(r"\s*\[[^\]]*chunk-\d+\]")
JUDGE_SYSTEM_PROMPT = ("You are a careful evaluator. Follow the output format requested in the user message "
                       "exactly. Reply with the requested output only.")

METRIC_LABELS = {
    "faithfulness": "Faithfulness",
    "llm_context_precision_with_reference": "Context precision",
    "context_recall": "Context recall",
    "answer_relevancy": "Answer relevancy",
}


class RagasUnavailable(Exception):
    """RAGAS is not installed or cannot be imported on this machine."""


def ragas_available() -> tuple[bool, str]:
    """(True, version) when RAGAS imports cleanly, else (False, reason). Never raises."""
    try:
        from importlib.metadata import version
        import ragas  # noqa: F401
        return True, version("ragas")
    except Exception as exc:  # ImportError, a blocked DLL, a dependency version clash ...
        return False, f"{type(exc).__name__}: {str(exc)[:300]}"


# --------------------------------------------------------------------------------------------- adapters
def build_adapters(completer, embedder):
    """Create RAGAS-compatible (judge LLM, embeddings) from a `.complete(system, user)` object and an embedder.

    Defined inside a function because the base classes only exist once RAGAS is imported.
    """
    try:
        from langchain_core.outputs import Generation, LLMResult
        from ragas.embeddings.base import BaseRagasEmbeddings
        from ragas.llms.base import BaseRagasLLM
    except Exception as exc:
        raise RagasUnavailable(f"{type(exc).__name__}: {exc}") from exc

    class CompleterLLM(BaseRagasLLM):
        def __init__(self, completer_):
            super().__init__()
            self.completer = completer_

        def generate_text(self, prompt, n=1, temperature=0.01, stop=None, callbacks=None):
            text = self.completer.complete(JUDGE_SYSTEM_PROMPT, prompt.to_string())
            return LLMResult(generations=[[Generation(text=text) for _ in range(max(n, 1))]])

        async def agenerate_text(self, prompt, n=1, temperature=0.01, stop=None, callbacks=None):
            return await asyncio.to_thread(self.generate_text, prompt, n, temperature, stop, callbacks)

        def is_finished(self, response) -> bool:
            return True

    class EmbedderAdapter(BaseRagasEmbeddings):
        def __init__(self, embedder_):
            super().__init__()
            self.embedder = embedder_

        def embed_query(self, text: str) -> list[float]:
            return self.embedder.embed([text])[0].tolist()

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return self.embedder.embed(list(texts)).tolist()

        async def aembed_query(self, text: str) -> list[float]:
            return await asyncio.to_thread(self.embed_query, text)

        async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
            return await asyncio.to_thread(self.embed_documents, texts)

    return CompleterLLM(completer), EmbedderAdapter(embedder)


# --------------------------------------------------------------------------------------------- results
@dataclass
class RagasCaseScore:
    case_id: str
    question: str
    scores: dict[str, float | None] = field(default_factory=dict)
    excluded_reason: str | None = None  # set when the case was not sent to RAGAS


@dataclass
class RagasReport:
    cases: list[RagasCaseScore] = field(default_factory=list)
    metrics_run: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    ragas_version: str = ""
    kind: str = KIND

    @property
    def means(self) -> dict[str, float | None]:
        out: dict[str, float | None] = {}
        for m in self.metrics_run:
            vals = [c.scores[m] for c in self.cases if c.scores.get(m) is not None]
            out[m] = sum(vals) / len(vals) if vals else None
        return out

    @property
    def scored_cases(self) -> int:
        return sum(1 for c in self.cases if c.excluded_reason is None)


def strip_citations(answer: str) -> str:
    """Remove '[file · chunk-0001]' markers: they are formatting, not claims, and add noise to the judge."""
    return _CITATION.sub("", answer).strip()


def exclusion_reason(r: CaseResult) -> str | None:
    if r.skipped:
        return "case was skipped"
    if not r.answerable:
        return "unanswerable case (a refusal has no claims to score)"
    if not r.reference:
        return "no reference answer in the dataset"
    if not r.contexts:
        return "nothing was retrieved"
    if not r.answer or r.answer.startswith("I could not find enough evidence"):
        return "the system abstained"
    return None


def _clean(value) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


# --------------------------------------------------------------------------------------------- runner
def run_ragas(report: EvalReport, completer, embedder) -> RagasReport:
    """Score the answerable, answered cases of `report` with RAGAS, using `completer` as the judge LLM.

    Costs real API calls (several per case per metric). Raises RagasUnavailable if RAGAS cannot be imported.
    """
    ok, info = ragas_available()
    if not ok:
        raise RagasUnavailable(info)
    with warnings.catch_warnings():
        # ragas 0.4 marks this (still working) API as deprecated in favour of ragas.metrics.collections
        warnings.simplefilter("ignore", DeprecationWarning)
        from ragas import EvaluationDataset, SingleTurnSample, evaluate
        from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall, ResponseRelevancy
    from ragas.run_config import RunConfig

    out = RagasReport(ragas_version=info)
    eligible: list[CaseResult] = []
    for r in report.results:
        why = exclusion_reason(r)
        if why:
            out.cases.append(RagasCaseScore(r.case_id, r.question, excluded_reason=why))
        else:
            eligible.append(r)
    if not eligible:
        out.notes.append("No eligible cases to score.")
        return out

    metrics = [Faithfulness(), LLMContextPrecisionWithReference(), LLMContextRecall()]
    if getattr(embedder, "semantic", False):
        metrics.append(ResponseRelevancy())
    else:
        out.notes.append("Answer relevancy skipped: it needs a semantic embedder (the lexical fallback is active).")
    out.metrics_run = [m.name for m in metrics]

    llm, embeddings = build_adapters(completer, embedder)
    dataset = EvaluationDataset(samples=[
        SingleTurnSample(user_input=r.question, response=strip_citations(r.answer),
                         retrieved_contexts=list(r.contexts), reference=r.reference)
        for r in eligible
    ])
    # RAGAS defaults (10 retries, up to 60 s back-off, 16 parallel calls) would hang the UI for minutes during an
    # API outage and invite rate-limit errors, so use a tighter budget.
    run_config = RunConfig(timeout=180, max_retries=3, max_wait=8, max_workers=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        result = evaluate(dataset=dataset, metrics=metrics, llm=llm, embeddings=embeddings,
                          run_config=run_config, raise_exceptions=False, show_progress=False)
    frame = result.to_pandas()
    for i, r in enumerate(eligible):
        row = frame.iloc[i]
        out.cases.append(RagasCaseScore(r.case_id, r.question,
                                        scores={m: _clean(row.get(m)) for m in out.metrics_run}))
    failed = sum(1 for c in out.cases if c.excluded_reason is None
                 for v in c.scores.values() if v is None)
    if failed:
        out.notes.append(f"{failed} metric value(s) could not be computed (judge output unparseable or API error).")
    out.cases.sort(key=lambda c: [x.case_id for x in report.results].index(c.case_id))
    return out
