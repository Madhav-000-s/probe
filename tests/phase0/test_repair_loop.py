"""The repair ladder must hit all four rungs on demand.

Everything downstream leans on this: if the ladder is flaky, an 800-run sweep
dies overnight on one malformed response. The phase gate says explicitly not
to proceed past here with it half-working.
"""

from __future__ import annotations

import json

from probe.grader.base import LLMGrader
from probe.grader.fixtures import (
    bad_spans,
    constant_grade,
    malformed_then_valid,
    never_valid,
)
from probe.models import Grade, LLMRole, Transcript
from probe.runtime.llm import FakeLLM, LLMRequest
from probe.runtime.retry import (
    ParseOutcome,
    RepairStats,
    extract_json,
    parse_model,
    structured_call,
)


def _grade(bank, client, answer="I would use an idempotency key and exponential backoff here."):
    grader = LLMGrader(client)
    return grader.grade(
        bank.questions[0], answer, Transcript(run_id="r", candidate_id="c", arm="fixed")
    )


# ------------------------------------------------------------------- rung 1


def test_valid_json_parses_first_time(bank):
    client = FakeLLM(by_role={LLMRole.GRADE: constant_grade(score=4)}, strict=False)
    out = _grade(bank, client)
    assert out.outcome is ParseOutcome.OK
    assert out.attempts == 1
    assert out.grade.score == 4


def test_json_wrapped_in_prose_and_fences_is_recovered():
    payload = {"competency_id": "x", "score": 3, "confidence": 0.5, "evidence_spans": []}
    fenced = f"Sure, here you go:\n```json\n{json.dumps(payload)}\n```\nHope that helps!"
    assert json.loads(extract_json(fenced))["score"] == 3

    value, err = parse_model(Grade, fenced)
    assert err is None and value.score == 3


# ------------------------------------------------------------------- rung 2


def test_malformed_then_valid_is_repaired(bank):
    client = FakeLLM(by_role={LLMRole.GRADE: malformed_then_valid(valid_score=5)}, strict=False)
    out = _grade(bank, client)
    assert out.outcome is ParseOutcome.REPAIRED
    assert out.attempts == 2
    assert out.grade.score == 5
    # The repair prompt must carry the validator's own message, not a generic
    # nudge — that is the whole reason the second attempt works.
    repair_prompt = client.calls[1].prompt
    assert "--- REPAIR ---" in repair_prompt
    assert "not valid JSON" in repair_prompt


def test_schema_valid_but_fabricated_spans_are_rejected(bank):
    """Offsets that do not quote the answer are a citation the audit trail
    cannot check. The postcheck must treat that as a parse failure."""
    client = FakeLLM(by_role={LLMRole.GRADE: bad_spans()}, strict=False)
    out = _grade(bank, client)
    assert out.outcome is ParseOutcome.DEGRADED
    assert any("do not quote the answer" in e for e in out.errors)


# ------------------------------------------------------------------- rung 3


def test_never_valid_falls_back_to_degraded_path(bank):
    client = FakeLLM(by_role={LLMRole.GRADE: never_valid()}, strict=False)
    out = _grade(bank, client)
    assert out.outcome is ParseOutcome.DEGRADED
    assert out.grade is not None
    assert out.grade.confidence == 0.0, "a degraded grade must not claim confidence"
    assert out.grade.score == 3
    assert out.grade.spans_valid_for(
        "I would use an idempotency key and exponential backoff here."
    )


# ------------------------------------------------------------------- rung 4


def test_no_degraded_path_yields_unrecoverable(bank):
    client = FakeLLM(by_role={LLMRole.GRADE: never_valid()}, strict=False)
    request = LLMRequest(
        role=LLMRole.GRADE, prompt="grade this", context={"competency_id": "x", "answer": "hi"}
    )
    result = structured_call(client, request, Grade, degraded=None, max_repairs=1)
    assert result.outcome is ParseOutcome.UNRECOVERABLE
    assert result.value is None
    assert result.attempts == 2


def test_unrecoverable_never_raises(bank):
    """A bad model output is a data condition, not an exception."""
    client = FakeLLM(by_role={LLMRole.GRADE: never_valid()}, strict=False)
    request = LLMRequest(role=LLMRole.GRADE, prompt="x", context={"answer": ""})
    result = structured_call(client, request, Grade, degraded=None)
    assert not result.ok and not result.usable


# ------------------------------------------------------------- instrumentation


def test_repair_stats_track_violation_and_recovery_rates(bank):
    stats = RepairStats()
    for responder, _expected in (
        (constant_grade(), ParseOutcome.OK),
        (constant_grade(), ParseOutcome.OK),
        (malformed_then_valid(), ParseOutcome.REPAIRED),
        (never_valid(), ParseOutcome.DEGRADED),
    ):
        client = FakeLLM(by_role={LLMRole.GRADE: responder}, strict=False)
        request = LLMRequest(
            role=LLMRole.GRADE,
            prompt="p",
            context={"competency_id": bank.questions[0].competency_id, "answer": "an answer here"},
        )
        stats.observe(
            structured_call(
                client,
                request,
                Grade,
                degraded=lambda: Grade(
                    competency_id="x", score=3, confidence=0.0, evidence_spans=[]
                ),
            )
        )

    assert stats.total == 4
    assert stats.ok == 2 and stats.repaired == 1 and stats.degraded == 1
    assert stats.violation_rate == 0.5
    assert stats.repair_success_rate == 0.5


def test_parse_errors_name_the_offending_field():
    value, err = parse_model(Grade, json.dumps({"competency_id": "x", "score": 9}))
    assert value is None
    assert "score" in err and "confidence" in err
