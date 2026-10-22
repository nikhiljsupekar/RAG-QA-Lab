"""Unsupported-answer handling: fabricated generator output must be detected, never silently trusted."""
from rag.answering import Answer, check_grounding
from rag.config import INSUFFICIENT_EVIDENCE_MESSAGE

EVIDENCE = ["Passwords must contain at least 14 characters. Passwords must be changed every 180 days."]


def test_grounded_answer_passes():
    g = check_grounding("Passwords must contain at least 14 characters.", EVIDENCE)
    assert g.supported


def test_invented_number_is_flagged():
    g = check_grounding("Passwords must contain at least 20 characters.", EVIDENCE)
    assert not g.supported and "20" in g.unsupported_numbers


def test_invented_claim_is_flagged():
    g = check_grounding("Employees receive a free lunch voucher every Friday.", EVIDENCE)
    assert not g.supported and g.unsupported_sentences


def test_citation_markers_are_ignored_by_grounding():
    g = check_grounding("Passwords need 14 characters. [security_policy.txt · chunk-0000]", EVIDENCE)
    assert g.supported


class HallucinatingGenerator:
    def generate(self, question, hits):
        return Answer("Passwords must be 32 characters and include a pet's name.", mode="llm",
                      citations=[hits[0].chunk.chunk_id])


class AbstainingGenerator:
    def generate(self, question, hits):
        from rag.answering import insufficient_answer
        return insufficient_answer()


class BrokenGenerator:
    def generate(self, question, hits):
        raise ConnectionError("network down")


def test_pipeline_flags_hallucinating_generator(pipeline):
    resp = pipeline.ask("What is the minimum number of characters in a password?", generator=HallucinatingGenerator())
    assert resp.grounding is not None and not resp.grounding.supported
    assert any("Grounding check failed" in w for w in resp.warnings)


def test_qa_check_fails_on_hallucinated_answer(pipeline):
    from qa.deterministic import check_grounded
    resp = pipeline.ask("What is the minimum number of characters in a password?", generator=HallucinatingGenerator())
    assert not check_grounded(resp).passed


def test_abstaining_generator_gives_standard_message(pipeline):
    resp = pipeline.ask("What is the minimum number of characters in a password?", generator=AbstainingGenerator())
    assert resp.answer.text == INSUFFICIENT_EVIDENCE_MESSAGE


def test_generator_failure_falls_back_to_extractive_with_warning(pipeline):
    resp = pipeline.ask("What is the minimum number of characters in a password?", generator=BrokenGenerator())
    assert resp.answer.mode == "extractive" and not resp.answer.insufficient
    assert any("Generator failed" in w for w in resp.warnings)


def test_answerable_question_requires_the_right_fact(pipeline):
    from qa import deterministic as det
    resp = pipeline.ask("What is the minimum number of characters in a password?")
    assert det.check_answer_facts(resp, ["14 characters"]).passed
    assert not det.check_answer_facts(resp, ["99 characters"]).passed
