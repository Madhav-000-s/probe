"""Expected information gain — the claim under test.

For a candidate item ``q`` targeting competency ``c``:

.. math::

    \\mathrm{EIG}(q) = H[p(\\theta_c)]
                     - \\mathbb{E}_{k \\sim p(k \\mid q)} H[p(\\theta_c \\mid q, k)]

    p(k \\mid q) = \\int P(\\text{score}=k \\mid \\theta)\\, p(\\theta)\\, d\\theta

Selection is not on bits alone:

.. math::

    \\arg\\max_q \\; \\frac{\\mathrm{EIG}(q)}{\\mathrm{cost}(q)}
                 - \\lambda \\cdot \\text{repeat-family penalty}

``cost(q)`` is the item's expected answer time. Dividing by it is what makes
the policy budget-aware rather than greedy: an item worth 0.4 nats in thirty
seconds beats one worth 0.5 nats in three minutes, and a candidate's time is
the scarce resource an interview actually spends. The repeat-family penalty
stops an interview degenerating into six consecutive "tell me about a time
when" — informative, and unbearable.

Everything here is closed-form over the grid. There is no sampling, so the EIG
of a given (posterior, item) pair is exact up to grid resolution, and the
Phase 2 tests check it against a Monte-Carlo estimate to prove it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from probe.belief.grid import GridBelief, pmf_entropy
from probe.belief.grm import category_probs
from probe.belief.state import BeliefState
from probe.config import ExperimentConfig
from probe.models import ProbeFamily, Question, QuestionBank, Rubric, StopReason, Transcript
from probe.policy.base import Ask, Decision, Policy, Stop

#: Cost normaliser, in seconds. Roughly one short answer, so a typical item has
#: cost near 1 and the EIG-per-cost ratio stays on a readable scale.
COST_UNIT = 60.0

#: How many recent turns the repeat-family penalty looks back over.
FAMILY_WINDOW = 4


def expected_information_gain(pmf: np.ndarray, grid: np.ndarray, a: float, b) -> float:
    """Closed-form EIG for one item against one posterior, in nats.

    Non-negative by construction: this is a mutual information, and conditioning
    cannot increase expected entropy. A negative value out of here means the
    posterior or the category probabilities are malformed, which is why the
    property test asserting it is worth keeping.
    """
    probs = category_probs(grid, a, b)  # (n_grid, 5)
    joint = probs * pmf[:, None]  # p(theta) P(k | theta)
    p_k = joint.sum(axis=0)  # (5,) marginal over categories

    prior_entropy = pmf_entropy(pmf)
    expected_posterior_entropy = 0.0
    for k in range(probs.shape[1]):
        if p_k[k] <= 1e-15:
            # A category this posterior considers impossible contributes
            # nothing; normalising by it would divide by ~zero.
            continue
        posterior_k = joint[:, k] / p_k[k]
        expected_posterior_entropy += float(p_k[k]) * pmf_entropy(posterior_k)

    return max(0.0, prior_entropy - expected_posterior_entropy)


def repeat_family_penalty(
    family: ProbeFamily, transcript: Transcript, window: int = FAMILY_WINDOW
) -> float:
    """Fraction of the last ``window`` turns that used the same probe family.

    A ratio rather than a raw count so the penalty means the same thing at
    turn 3 and at turn 11.
    """
    recent = transcript.turns[-window:]
    if not recent:
        return 0.0
    same = sum(1 for t in recent if t.question_id.endswith(f"::{family.value}"))
    return same / len(recent)


@dataclass
class Scored:
    question: Question
    eig: float
    cost: float
    penalty: float

    @property
    def objective(self) -> float:
        return self.eig / self.cost


class EIGPolicy(Policy):
    """Belief-driven selection. The arm the project exists to evaluate."""

    name = "eig"

    def __init__(
        self,
        rubric: Rubric,
        config: ExperimentConfig,
        *,
        required_only: bool = True,
        skip_resolved: bool = True,
    ) -> None:
        self.rubric = rubric
        self.config = config
        #: Restrict candidates to competencies the role actually requires.
        #: Spending a scarce question budget resolving something the job does
        #: not need is a loss even when it is informative.
        self.required_only = required_only
        #: Drop competencies whose posterior SD is already below tau.
        #:
        #: Alignment with the stop rule, not an optimisation. The interview
        #: terminates when *every* required competency is under tau, so a
        #: competency already under it contributes exactly nothing to
        #: termination and every further question spent on it is wasted, no
        #: matter how many nats it returns. Pure entropy-greedy selection has
        #: no notion of the threshold it is being measured against and will
        #: happily keep mining an already-resolved competency because it is
        #: still the widest thing on the board.
        self.skip_resolved = skip_resolved
        self.last_scores: list[Scored] = []

    def _candidate_ids(self, belief: BeliefState) -> list[str]:
        comps = self.rubric.required if self.required_only else self.rubric.competencies
        ids = [c.id for c in comps] or self.rubric.ids
        if not self.skip_resolved:
            return ids
        unresolved = [cid for cid in ids if belief.sd(cid) >= self.config.tau]
        # If everything is resolved the stop rule is about to fire anyway;
        # returning the full set keeps this from being the thing that ends the
        # interview, so the recorded stop reason stays honest.
        return unresolved or ids

    def score_all(
        self, belief: BeliefState, transcript: Transcript, bank: QuestionBank
    ) -> list[Scored]:
        if not isinstance(belief, GridBelief):
            raise TypeError(
                f"{self.name} needs a grid posterior; got {type(belief).__name__}. "
                "Running it on the prior-only belief would score at chance and "
                "silently invalidate the arm comparison."
            )
        pool = self.available(bank, transcript, self._candidate_ids(belief))

        scored: list[Scored] = []
        for question in pool:
            eig = expected_information_gain(
                belief.pmf(question.competency_id),
                belief.grid,
                question.grm.a,
                question.grm.b,
            )
            scored.append(
                Scored(
                    question=question,
                    eig=eig,
                    cost=max(question.expected_seconds / COST_UNIT, 1e-6),
                    penalty=repeat_family_penalty(question.probe_family, transcript),
                )
            )
        return scored

    def next_question(
        self, belief: BeliefState, transcript: Transcript, bank: QuestionBank
    ) -> Decision:
        scored = self.score_all(belief, transcript, bank)
        self.last_scores = scored
        if not scored:
            return Stop(StopReason.BANK_EXHAUSTED, "no unasked items for required competencies")

        lam = self.config.repeat_family_lambda
        best = max(scored, key=lambda s: s.objective - lam * s.penalty)

        # The epsilon test is against raw information, not information per
        # second. "Nothing left worth asking" is a statement about what can
        # still be learned; dividing by cost first would let an expensive but
        # genuinely informative item look exhausted.
        if best.eig < self.config.epsilon:
            return Stop(
                StopReason.NO_INFORMATIVE_QUESTION,
                f"best EIG {best.eig:.5f} < epsilon {self.config.epsilon}",
            )

        return Ask(
            question=best.question,
            eig=best.eig,
            reason=(
                f"EIG={best.eig:.4f} nats, cost={best.cost:.2f}, "
                f"family_penalty={best.penalty:.2f}"
            ),
            diagnostics={
                "eig": best.eig,
                "cost": best.cost,
                "penalty": best.penalty,
                "objective": best.objective - lam * best.penalty,
                "n_candidates": float(len(scored)),
                "posterior_sd": belief.sd(best.question.competency_id),
            },
        )
