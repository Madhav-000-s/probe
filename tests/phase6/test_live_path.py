"""Defects that only a real model could surface.

Every one of these was invisible under the offline backends, because those
compose their output directly instead of writing JSON and counting characters.
The suite exists so they stay fixed without needing to spend money to notice.
"""

from __future__ import annotations

import pytest

from probe.grader.base import degraded_grade, relocate_spans
from probe.models import Behavior, Grade, LLMRole
from probe.runtime.llm import DEFAULT_MAX_TOKENS, ROLE_MAX_TOKENS, LLMRequest, LLMResponse
from probe.runtime.retry import ParseOutcome, structured_call
from probe.sim.candidate import AnswerEnvelope, extract_prose

# ------------------------------------------------------- evidence relocation

ANSWER = (
    "Cursor pagination avoids scanning large offsets, which matters at depth. "
    "Offset pagination is simpler but degrades as the offset grows."
)


def _grade(spans: list[dict]) -> Grade:
    return Grade(
        competency_id="api_design.pagination",
        score=4,
        confidence=0.9,
        evidence_spans=spans,
        rationale="test",
    )


def test_a_correct_quote_with_wrong_offsets_is_repaired_not_rejected():
    """The live failure, in miniature. Haiku quoted the answer verbatim and
    reported offsets that were off by tens of characters; the postcheck
    rejected 100% of its grades and the interview scored everything 3 at zero
    confidence."""
    quote = "Cursor pagination avoids scanning large offsets"
    grade = _grade([{"start": 999, "end": 999 + len(quote), "text": quote}])

    fixed = relocate_spans(grade, ANSWER)

    span = fixed.evidence_spans[0]
    assert span.verify_against(ANSWER)
    assert ANSWER[span.start : span.end] == quote


def test_fabricated_evidence_is_still_rejected():
    """The whole point of the span. Relocation must not become a way for a
    grader to cite something the candidate never said."""
    grade = _grade([{"start": 0, "end": 20, "text": "I have a PhD in distributed systems."}])

    fixed = relocate_spans(grade, ANSWER)

    assert not fixed.evidence_spans[0].verify_against(ANSWER)


def test_a_paraphrase_does_not_count_as_a_quotation():
    """Exact substring only — accepting near-matches would let a summary pass
    as evidence, which is the failure the span exists to catch."""
    grade = _grade([{"start": 0, "end": 30, "text": "cursor pagination avoids scanning"}])  # case
    assert relocate_spans(grade, ANSWER).evidence_spans[0].relocated_in(ANSWER) is None


def test_already_correct_spans_are_returned_untouched():
    """The offline backend computes exact offsets, so this path has to be a
    no-op or every committed number moves."""
    quote = "Offset pagination is simpler"
    start = ANSWER.index(quote)
    grade = _grade([{"start": start, "end": start + len(quote), "text": quote}])
    assert relocate_spans(grade, ANSWER) is grade


def test_degraded_grade_is_still_reachable_and_marked():
    """When the grade really is unusable the fallback must stay distinguishable
    from a real one — zero confidence is what the belief update keys on."""
    from probe.bank.loader import load_bank

    fallback = degraded_grade(load_bank("v2").questions[0], "no idea, sorry")
    assert fallback.confidence == 0.0
    assert fallback.score == 3


# ------------------------------------------------------------ token budgets


def test_rubric_compilation_gets_a_budget_that_fits_a_rubric():
    """A 14-competency rubric is ~3k tokens of JSON. The flat 1024 default
    truncated it mid-object, and the repair reproduced the identical
    truncation."""
    assert ROLE_MAX_TOKENS[LLMRole.RUBRIC_COMPILE] >= 4096
    assert ROLE_MAX_TOKENS[LLMRole.RUBRIC_COMPILE] > DEFAULT_MAX_TOKENS


def test_every_role_has_a_budget():
    missing = [r.value for r in LLMRole if r not in ROLE_MAX_TOKENS]
    assert not missing, f"roles with no output budget: {missing}"


def test_explicit_max_tokens_wins_over_the_role_default():
    role_default = LLMRequest(role=LLMRole.GRADE, prompt="x")
    explicit = LLMRequest(role=LLMRole.GRADE, prompt="x", max_tokens=77)
    assert role_default.token_budget == ROLE_MAX_TOKENS[LLMRole.GRADE]
    assert explicit.token_budget == 77


def test_truncation_is_reported_as_truncation_not_as_bad_json():
    """A truncated object parses as invalid JSON, so the ladder asks for
    well-formed JSON, the model obliges, and the reply is cut off at exactly
    the same place. The error has to name the real cause."""

    class Truncating:
        name = "stub"
        model = "stub"

        def complete(self, request):
            return LLMResponse(
                text='{"answer": "half a sen',
                model="stub",
                completion_tokens=request.token_budget,
                truncated=True,
            )

    result = structured_call(
        Truncating(), LLMRequest(role=LLMRole.PERSONA_ANSWER, prompt="x"), AnswerEnvelope
    )

    assert result.outcome is ParseOutcome.UNRECOVERABLE
    assert any("output budget" in e for e in result.errors)


def test_repair_tokens_are_charged_to_the_caller():
    """Counting only the successful attempt makes a flaky role look as cheap
    as a clean one."""

    class Flaky:
        name = "stub"
        model = "stub"

        def __init__(self):
            self.n = 0

        def complete(self, request):
            self.n += 1
            text = "not json" if self.n == 1 else '{"answer": "fine"}'
            return LLMResponse(text=text, model="stub", prompt_tokens=10, completion_tokens=5)

    result = structured_call(
        Flaky(), LLMRequest(role=LLMRole.PERSONA_ANSWER, prompt="x"), AnswerEnvelope
    )

    assert result.outcome is ParseOutcome.REPAIRED
    assert result.tokens == 30, "the repair call's tokens are missing"


# ------------------------------------------------------------ prose salvage


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('{"answer": "hello"}', "hello"),
        ('```json\n{"answer": "fenced"}\n```', "fenced"),
        ("Just the answer, no envelope.", "Just the answer, no envelope."),
        ("", ""),
    ],
)
def test_extract_prose_salvages_what_it_can(raw, expected):
    assert extract_prose(raw) == expected


# ------------------------------------------------- ground truth stays ours

def test_plan_answer_leaves_the_composer_bit_identical():
    """``compose_answer`` was split so the live path could share the content
    decision. If the split shifted the RNG stream by one draw, every committed
    sim answer would change."""
    from probe.bank.loader import load_bank
    from probe.sim.answers import compose_answer, plan_answer
    from probe.sim.style import style_by_id

    question = load_bank("v2").questions[3]
    style = style_by_id("neutral")
    kwargs = dict(
        question=question,
        theta=0.4,
        behavior=Behavior.HONEST,
        distractor_pool=["mutex", "quorum"],
        seed=4242,
    )

    level, plan, _ = plan_answer(**kwargs)
    _, composed_level, composed_plan = compose_answer(style=style, **kwargs)

    assert level == composed_level
    assert plan == composed_plan


def test_the_live_prompt_never_carries_the_number():
    """The persona may know theta; the *prompt* must express depth as coverage.
    'Answer as a level-4 candidate' invites the model to apply its own idea of
    what a 4 is, which would make the recovery metric score the model's
    self-report."""
    from probe.bank.loader import load_bank
    from probe.sim.answers import plan_answer
    from probe.sim.candidate import PersonaCandidate
    from probe.sim.persona import load_population

    personas, _ = load_population("v2")
    question = load_bank("v2").questions[0]
    candidate = PersonaCandidate(personas[0], client=None, seed=1)

    theta = personas[0].ability(question.competency_id)
    level, plan, _ = plan_answer(
        question=question,
        theta=theta,
        behavior=personas[0].behavior,
        distractor_pool=[],
        seed=7,
    )
    prompt = candidate._prompt(question, plan)

    assert f"{theta:.1f}" not in prompt
    assert f"level {level}" not in prompt.lower()
    assert "theta" not in prompt.lower()


def test_the_live_answer_is_recorded_against_our_drawn_level(monkeypatch):
    """A live model returns prose and nothing else. The drawn level and concept
    counts in the record must come from the harness's own plan, never from
    whatever the model says about itself."""
    from probe.bank.loader import load_bank
    from probe.models import Transcript
    from probe.sim.candidate import PersonaCandidate
    from probe.sim.persona import load_population

    personas, _ = load_population("v2")
    question = load_bank("v2").questions[0]

    class ProseOnly:
        name = "stub"
        model = "stub"

        def complete(self, request):
            # Deliberately lies about its own depth in the envelope.
            return LLMResponse(
                text='{"answer": "a real answer", "drawn_level": 5, "n_concepts": 99}',
                model="stub",
                completion_tokens=7,
            )

    candidate = PersonaCandidate(personas[0], client=ProseOnly(), seed=1)
    candidate.answer(question, Transcript(run_id="r", candidate_id=personas[0].id, arm="fixed"))

    record = candidate.records[-1]
    assert record.drawn_level != 5 or record.n_concepts != 99, (
        "the model's self-report was trusted as ground truth"
    )
    assert record.n_concepts < 99
