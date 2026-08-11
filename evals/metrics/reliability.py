"""Grader reliability — research question two.

Four properties, all of which a grader can fail while looking perfectly
reasonable on any single answer:

* **test-retest** — regrade the same answer under five seeds. The SD is how
  much of a score is the grader's mood rather than the candidate.
* **position bias** — does an identical answer score differently at turn 2
  than at turn 10?
* **anchor-order sensitivity** — does presenting the rubric levels in reverse
  change the verdict? It should not; the anchors are a set, not a ranking cue.
* **span entailment** — does the cited evidence actually support the score, or
  is the citation decorative?

Under the offline backend these measure a simulator whose noise parameters were
chosen here, and the README says so before quoting any of them. The machinery
is what transfers: point it at a real model and the same four numbers come out
meaning something else.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from probe.grader.base import LLMGrader, build_grade_prompt
from probe.models import Question, Transcript
from probe.sim.textfeatures import match_concepts

EMPTY = Transcript(run_id="reliability", candidate_id="-", arm="reliability")

#: Seeds used for the test-retest measurement.
RETEST_SEEDS = (1, 2, 3, 4, 5)


@dataclass
class ReliabilityReport:
    test_retest_sd: float
    exact_agreement: float
    position_bias: float
    anchor_order_disagreement: float
    span_entailment_rate: float
    n_answers: int
    #: False when the backend cannot resample — see :func:`test_retest`. The
    #: two test-retest cells are meaningless then, and reporting them anyway
    #: produces a spectacularly flattering number for the wrong reason.
    retest_is_meaningful: bool = True

    def to_dict(self) -> dict[str, float]:
        payload = {
            "position_bias": round(self.position_bias, 4),
            "anchor_order_disagreement": round(self.anchor_order_disagreement, 4),
            "span_entailment_rate": round(self.span_entailment_rate, 4),
            "n_answers": self.n_answers,
            "retest_is_meaningful": self.retest_is_meaningful,
        }
        if self.retest_is_meaningful:
            payload["test_retest_sd"] = round(self.test_retest_sd, 4)
            payload["exact_agreement"] = round(self.exact_agreement, 4)
        else:
            payload["test_retest_sd"] = None
            payload["exact_agreement"] = None
        return payload


def _score(grader: LLMGrader, question: Question, answer: str, seed: int, transcript=EMPTY):
    outcome = grader.grade(question, answer, transcript, seed=seed)
    return outcome.grade


def retest_is_meaningful(grader: LLMGrader) -> bool:
    """Whether re-grading under different seeds actually resamples anything.

    The seed is a knob on the offline backends' RNG. A provider API has no such
    knob — ``AnthropicClient`` never sends ``request.seed``, and the grader runs
    at temperature 0 — so the five "retest seeds" become five byte-identical
    requests, and the resulting SD measures provider nondeterminism rather than
    grader noise.

    That number is small and looks wonderful: the first live run produced a
    test-retest SD of 0.014 against the offline grader's 0.321, which would have
    read as a twenty-fold reliability improvement and was an artefact of asking
    the same question five times. Reported as not-applicable instead.
    """
    return bool(getattr(grader.client, "honours_seed", True))


def test_retest(
    grader: LLMGrader, samples: Sequence[tuple[Question, str]], seeds=RETEST_SEEDS
) -> tuple[float, float]:
    """``(mean per-answer SD, exact agreement across seeds)``."""
    sds, agreements = [], []
    for question, answer in samples:
        scores = [
            g.score for s in seeds if (g := _score(grader, question, answer, s)) is not None
        ]
        if len(scores) < 2:
            continue
        sds.append(float(np.std(scores, ddof=1)))
        modal = max(set(scores), key=scores.count)
        agreements.append(sum(s == modal for s in scores) / len(scores))
    return (
        float(np.mean(sds)) if sds else float("nan"),
        float(np.mean(agreements)) if agreements else float("nan"),
    )


def position_bias(
    grader: LLMGrader,
    samples: Sequence[tuple[Question, str]],
    early: int = 1,
    late: int = 10,
) -> float:
    """Mean score change from grading the same answer late rather than early.

    Constructed by handing the grader transcripts of different lengths with
    identical content, so the only thing that varies is where in the interview
    the answer sits.
    """
    from probe.models import BeliefSnapshot, Turn

    def transcript_of_length(n: int) -> Transcript:
        snapshot = BeliefSnapshot(means={}, sds={}, entropies={})
        return Transcript(
            run_id="pos",
            candidate_id="-",
            arm="reliability",
            turns=[
                Turn(
                    run_id="pos",
                    turn_idx=i,
                    question_id=f"filler::{i}::scenario",
                    competency_id="filler",
                    question_text="q",
                    answer="a",
                    grade=None,
                    belief_after=snapshot,
                )
                for i in range(n)
            ],
        )

    deltas = []
    early_t, late_t = transcript_of_length(early), transcript_of_length(late)
    for question, answer in samples:
        a = _score(grader, question, answer, seed=7, transcript=early_t)
        b = _score(grader, question, answer, seed=7, transcript=late_t)
        if a and b:
            deltas.append(b.score - a.score)
    return float(np.mean(deltas)) if deltas else float("nan")


def anchor_order_sensitivity(
    grader: LLMGrader, samples: Sequence[tuple[Question, str]]
) -> float:
    """Disagreement rate when the rubric anchors are presented in reverse.

    The anchors describe five levels; the order they appear in is presentation,
    not content. A grader sensitive to it is reading the prompt's shape rather
    than the rubric.
    """
    disagreements = []
    for question, answer in samples:
        reversed_question = question.model_copy(
            update={"anchors": list(reversed(question.anchors))}
        )
        a = _score(grader, question, answer, seed=9)
        b = _score(grader, reversed_question, answer, seed=9)
        if a and b:
            disagreements.append(a.score != b.score)
    return float(np.mean(disagreements)) if disagreements else float("nan")


def span_entailment(
    grader: LLMGrader, samples: Sequence[tuple[Question, str]]
) -> float:
    """Fraction of grades whose cited spans support the score they carry.

    A high score citing text that names none of the anchor concepts is a
    decorative citation. This is the check that keeps mandatory spans from
    becoming a formality — the schema forces a span, and this asks whether the
    span means anything.
    """
    supported = total = 0
    for question, answer in samples:
        grade = _score(grader, question, answer, seed=11)
        if grade is None or not grade.evidence_spans:
            continue
        total += 1
        cited = " ".join(s.text for s in grade.evidence_spans)
        pool = list(question.anchor(5).required_concepts)
        n_cited = len(match_concepts(cited, pool))
        # A grade of 3+ claims the answer named at least two anchor concepts,
        # so its citation should contain at least one of them.
        supported += n_cited >= 1 if grade.score >= 3 else True
    return supported / total if total else float("nan")


def measure(grader: LLMGrader, samples: Sequence[tuple[Question, str]]) -> ReliabilityReport:
    meaningful = retest_is_meaningful(grader)
    sd, agreement = test_retest(grader, samples) if meaningful else (float("nan"), float("nan"))
    return ReliabilityReport(
        retest_is_meaningful=meaningful,
        test_retest_sd=sd,
        exact_agreement=agreement,
        position_bias=position_bias(grader, samples),
        anchor_order_disagreement=anchor_order_sensitivity(grader, samples),
        span_entailment_rate=span_entailment(grader, samples),
        n_answers=len(samples),
    )


def prompt_is_stable_under_anchor_order(question: Question, answer: str) -> bool:
    """The prompt genuinely differs when anchors are reordered — otherwise the
    sensitivity metric would be measuring nothing."""
    reversed_question = question.model_copy(update={"anchors": list(reversed(question.anchors))})
    return build_grade_prompt(question, answer, EMPTY) != build_grade_prompt(
        reversed_question, answer, EMPTY
    )
