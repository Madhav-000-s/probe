"""Canned grader responses for hermetic tests and ``probe run --fake``.

These are fixtures, not a grading strategy. They exist so the runtime spine can
be exercised end to end — repair ladder, span validation, persistence,
resumption — without a model and without network. Every one of them echoes
real character offsets back out of the answer it was handed, because a fixture
whose spans do not validate would quietly route every turn down the degraded
path and the integration test would prove nothing.
"""

from __future__ import annotations

import json

from probe.runtime.llm import LLMRequest


def _span_over(answer: str, max_len: int = 60) -> dict[str, object]:
    body = answer.strip()
    start = answer.index(body[0]) if body else 0
    end = min(len(answer), start + max(1, min(len(body), max_len)))
    return {"start": start, "end": end, "text": answer[start:end]}


def constant_grade(score: int = 3, confidence: float = 0.7):
    """A grader that always returns ``score`` with a valid span."""

    def responder(request: LLMRequest) -> str:
        answer = str(request.context.get("answer", ""))
        return json.dumps(
            {
                "competency_id": request.context.get("competency_id", "unknown"),
                "score": score,
                "confidence": confidence,
                "evidence_spans": [_span_over(answer)],
                "flags": [],
                "rationale": "fixture: constant grade",
            }
        )

    return responder


def length_proportional_grade():
    """Score rises with answer length.

    Crude on purpose: it gives the Phase 0 tests a grader whose output varies
    with its input without pretending to measure anything. Notably it is
    *pure style* — which makes it a useful negative control when the fairness
    suite lands, since a real grader scoring like this is exactly the failure
    Q3 is looking for.
    """

    def responder(request: LLMRequest) -> str:
        answer = str(request.context.get("answer", ""))
        words = len(answer.split())
        score = 1 + min(4, words // 12)
        return json.dumps(
            {
                "competency_id": request.context.get("competency_id", "unknown"),
                "score": score,
                "confidence": 0.5,
                "evidence_spans": [_span_over(answer)],
                "flags": [],
                "rationale": f"fixture: {words} words",
            }
        )

    return responder


def malformed_then_valid(valid_score: int = 4):
    """First call returns unparseable text, second returns a valid grade.

    Drives the repair ladder's middle rung in tests: stage one fails, the
    repair prompt succeeds, and the outcome should be ``REPAIRED`` rather than
    ``DEGRADED``.
    """
    state = {"n": 0}

    def responder(request: LLMRequest) -> str:
        state["n"] += 1
        if state["n"] % 2 == 1:
            return "Sure! Here's my assessment: the candidate did quite well overall."
        answer = str(request.context.get("answer", ""))
        return json.dumps(
            {
                "competency_id": request.context.get("competency_id", "unknown"),
                "score": valid_score,
                "confidence": 0.6,
                "evidence_spans": [_span_over(answer)],
                "flags": [],
                "rationale": "fixture: repaired on second attempt",
            }
        )

    return responder


def never_valid():
    """Always unparseable. Exercises the degraded rung and, with the degraded
    path disabled, the ``unrecoverable`` terminal state."""

    def responder(request: LLMRequest) -> str:
        return "I'm not able to produce that format."

    return responder


def bad_spans():
    """Schema-valid, meaning-invalid: offsets that do not quote the answer.

    This is the failure class the ``postcheck`` hook exists for, and the one
    that would otherwise put a fabricated citation into the audit trail.
    """

    def responder(request: LLMRequest) -> str:
        return json.dumps(
            {
                "competency_id": request.context.get("competency_id", "unknown"),
                "score": 5,
                "confidence": 0.99,
                "evidence_spans": [
                    {"start": 0, "end": 25, "text": "text that is not present!"}
                ],
                "flags": [],
                "rationale": "fixture: fabricated citation",
            }
        )

    return responder
