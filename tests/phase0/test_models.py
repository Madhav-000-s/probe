"""Schema round-trips for every model, plus the validators that carry design
commitments rather than mere type-checking."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from probe.models import (
    BeliefSnapshot,
    Competency,
    EvidenceSpan,
    GRMParams,
    Grade,
    GradeFlag,
    LLMCallRecord,
    LLMRole,
    Persona,
    ProbeFamily,
    RunRecord,
    StopReason,
    StyleProfile,
    Transcript,
    Turn,
)


def test_taxonomy_node_round_trip(taxonomy):
    for node in taxonomy:
        assert node.model_validate_json(node.model_dump_json()) == node


def test_rubric_round_trip(rubric):
    from probe.models import Rubric

    assert Rubric.model_validate_json(rubric.model_dump_json()) == rubric


def test_question_round_trip(bank):
    from probe.models import Question, QuestionBank

    for q in bank.questions:
        assert Question.model_validate_json(q.model_dump_json()) == q
    assert QuestionBank.model_validate_json(bank.model_dump_json()) == bank


def test_grade_round_trip():
    g = Grade(
        competency_id="databases.indexing",
        score=4,
        confidence=0.8,
        evidence_spans=[EvidenceSpan(start=0, end=11, text="composite i")],
        flags=[GradeFlag.UNSUPPORTED_CLAIM],
        rationale="names composite index column order",
    )
    assert Grade.model_validate_json(g.model_dump_json()) == g


def test_persona_round_trip():
    p = Persona(
        id="p01",
        theta_star={"databases.indexing": 1.25},
        style=StyleProfile(id="neutral", verbosity=1.0, hedging=0.1, assertiveness=0.1, l1_transfer=0.0),
        behavior="honest",
        resume="Built things.",
        jd_id="jd-backend",
        split="calibration",
    )
    assert Persona.model_validate_json(p.model_dump_json()) == p


def test_turn_and_transcript_round_trip(bank):
    snap = BeliefSnapshot(means={"a": 0.1}, sds={"a": 0.9}, entropies={"a": 1.3})
    turn = Turn(
        run_id="r1",
        turn_idx=0,
        question_id=bank.questions[0].id,
        competency_id=bank.questions[0].competency_id,
        question_text=bank.questions[0].text,
        answer="an answer",
        grade=None,
        belief_after=snap,
    )
    t = Transcript(run_id="r1", candidate_id="p01", arm="fixed", turns=[turn])
    assert Transcript.model_validate_json(t.model_dump_json()) == t


def test_run_and_call_records_round_trip():
    r = RunRecord(
        run_id="r1",
        arm="fixed",
        persona_id="p01",
        style_id="neutral",
        bank_version="v0",
        population_version="v0",
        code_commit="abc1234",
        seed=7,
        stop_reason=StopReason.CONFIDENCE,
    )
    assert RunRecord.model_validate_json(r.model_dump_json()) == r

    c = LLMCallRecord(
        call_id="c1",
        run_id="r1",
        role=LLMRole.GRADE,
        prompt="p",
        prompt_hash="h",
        model="m",
        seed=7,
        temperature=0.0,
        raw_output="{}",
        parsed_ok=True,
    )
    assert LLMCallRecord.model_validate_json(c.model_dump_json()) == c


# --------------------------------------------------------------- validators


def test_span_must_be_ordered_and_non_empty():
    with pytest.raises(ValidationError):
        EvidenceSpan(start=5, end=5, text="x")
    with pytest.raises(ValidationError):
        EvidenceSpan(start=0, end=3, text="   ")


def test_span_verifies_against_source():
    source = "quorum reads are the fix"
    good = EvidenceSpan(start=0, end=12, text="quorum reads")
    bad = EvidenceSpan(start=0, end=12, text="quorum write")
    out_of_range = EvidenceSpan(start=100, end=120, text="quorum reads")
    assert good.verify_against(source)
    assert not bad.verify_against(source)
    assert not out_of_range.verify_against(source)


def test_evidence_in_resume_must_be_span_backed():
    """The compiler's core commitment: an evidence score with no citation is a
    number somebody made up."""
    with pytest.raises(ValidationError, match="no resume span"):
        Competency(
            id="databases.indexing",
            label="Index design",
            required_level=4,
            evidence_in_resume=0.9,
            prior_mean=0.5,
            prior_var=0.4,
            probe_families=[ProbeFamily.DEBUG],
            resume_spans=[],
        )


def test_grm_thresholds_must_be_increasing():
    with pytest.raises(ValidationError, match="strictly increasing"):
        GRMParams(a=1.0, b=[0.5, 0.2, 1.0, 1.5])
    with pytest.raises(ValidationError):
        GRMParams(a=0.0, b=[-1.0, 0.0, 1.0, 2.0])


def test_grade_score_is_bounded():
    for bad in (0, 6, -1):
        with pytest.raises(ValidationError):
            Grade(competency_id="x", score=bad, confidence=0.5, evidence_spans=[])


def test_grade_schema_cannot_express_bulk_score_setting():
    """Injection defence at the schema level: there is no field a compromised
    grader could use to touch anything but the one competency it was asked
    about."""
    fields = set(Grade.model_fields)
    assert fields == {
        "competency_id",
        "score",
        "confidence",
        "evidence_spans",
        "flags",
        "rationale",
    }
    assert Grade.model_config.get("extra") in (None, "ignore")


def test_spans_valid_for_helper():
    answer = "use an idempotency key"
    g = Grade(
        competency_id="x",
        score=3,
        confidence=0.5,
        evidence_spans=[EvidenceSpan(start=7, end=22, text="idempotency key")],
    )
    assert g.spans_valid_for(answer)
    assert not g.spans_valid_for("a completely different answer entirely")
