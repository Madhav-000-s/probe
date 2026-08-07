"""Administering the whole bank to a calibration sample.

Standard calibration design: everyone in the calibration sample answers every
item, so each item accumulates enough responses to estimate five parameters
from. This is not an interview — no policy, no stop rule, no budget — it is
the psychometric pass that turns authoring defaults into fitted quantities.

The split is enforced at the call site and tested: an eval-split persona must
never appear in a calibration input, or the held-out design that kills the
circularity objection is a fiction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from probe.grader.base import LLMGrader
from probe.models import Persona, QuestionBank, Transcript
from probe.sim.answers import answer_seed, compose_answer

EMPTY = Transcript(run_id="calibration", candidate_id="-", arm="calibration")


@dataclass
class AdministrationResult:
    #: persona_id -> {question_id: score}
    responses: dict[str, dict[str, int]] = field(default_factory=dict)
    #: persona_id -> {competency_id: mean score}, the input to the empirical
    #: correlation matrix.
    competency_scores: dict[str, dict[str, float]] = field(default_factory=dict)
    n_graded: int = 0
    n_unrecoverable: int = 0

    def score_matrix(self, competency_ids: list[str]) -> np.ndarray:
        """``(n_personas, n_competencies)`` of mean scores, NaN where absent."""
        rows = []
        for pid in sorted(self.competency_scores):
            row = self.competency_scores[pid]
            rows.append([row.get(cid, np.nan) for cid in competency_ids])
        return np.asarray(rows, dtype=float)


def administer(
    personas: list[Persona],
    bank: QuestionBank,
    client,
    *,
    seed: int = 20260807,
    style_neutral: bool = True,
    include_quarantined: bool = True,
) -> AdministrationResult:
    """Every persona answers every item; every answer is graded.

    Grading goes through the ordinary :class:`LLMGrader`, not a shortcut, so
    the parameters are fitted against the same measurement process the
    interviews will use. Calibrating against a cleaner grader than the one in
    production would make every fitted difficulty optimistic.
    """
    from probe.sim.style import STYLE_PRESETS

    grader = LLMGrader(client)
    out = AdministrationResult()
    items = bank.questions if include_quarantined else bank.live()

    for persona in personas:
        style = STYLE_PRESETS["neutral"] if style_neutral else persona.style
        row: dict[str, int] = {}
        by_competency: dict[str, list[int]] = {}

        for question in items:
            if question.competency_id not in persona.theta_star:
                continue
            text, _level, _plan = compose_answer(
                question=question,
                theta=persona.ability(question.competency_id),
                behavior=persona.behavior,
                style=style,
                distractor_pool=[],
                seed=answer_seed(persona.id, question.id, style.id, seed),
            )
            outcome = grader.grade(question, text, EMPTY, seed=seed)
            if outcome.grade is None:
                out.n_unrecoverable += 1
                continue
            row[question.id] = outcome.grade.score
            by_competency.setdefault(question.competency_id, []).append(outcome.grade.score)
            out.n_graded += 1

        out.responses[persona.id] = row
        out.competency_scores[persona.id] = {
            cid: float(np.mean(scores)) for cid, scores in by_competency.items()
        }
    return out
