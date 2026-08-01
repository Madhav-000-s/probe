"""The belief-state contract.

Everything the policy and the stop rule need to know about a candidate lives
behind this interface: a posterior mean, a posterior SD, and an entropy, per
competency. Both the prior-only Phase 0 implementation and the Phase 2 grid
posterior satisfy it, so swapping them is a one-line change in the loop and
the arms stay comparable.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

from probe.models import BeliefSnapshot, Question, Rubric


class BeliefState(ABC):
    """Per-competency posterior over latent ability ``theta``."""

    def __init__(self, rubric: Rubric) -> None:
        self.rubric = rubric
        self.n_observations: dict[str, int] = {c.id: 0 for c in rubric.competencies}

    # ---------------------------------------------------------- read surface

    @abstractmethod
    def mean(self, competency_id: str) -> float: ...

    @abstractmethod
    def sd(self, competency_id: str) -> float: ...

    @abstractmethod
    def entropy(self, competency_id: str) -> float: ...

    @abstractmethod
    def credible_interval(self, competency_id: str, mass: float = 0.8) -> tuple[float, float]: ...

    # --------------------------------------------------------- write surface

    @abstractmethod
    def update(self, question: Question, score: int) -> None:
        """Fold one graded response into the posterior for the question's
        target competency."""

    # -------------------------------------------------------------- derived

    @property
    def competency_ids(self) -> list[str]:
        return self.rubric.ids

    def snapshot(self) -> BeliefSnapshot:
        """Persisted every turn. Accuracy-vs-budget curves are computed from
        these post hoc rather than by re-running interviews per budget."""
        return BeliefSnapshot(
            means={c: self.mean(c) for c in self.competency_ids},
            sds={c: self.sd(c) for c in self.competency_ids},
            entropies={c: self.entropy(c) for c in self.competency_ids},
        )

    def resolved(self, tau: float, required_only: bool = True) -> dict[str, bool]:
        comps = self.rubric.required if required_only else self.rubric.competencies
        return {c.id: self.sd(c.id) < tau for c in comps}

    def all_resolved(self, tau: float) -> bool:
        flags = self.resolved(tau)
        return bool(flags) and all(flags.values())

    def widest(self) -> str:
        """Competency with the largest posterior SD. Used by the fixed arm's
        tie-breaking and by diagnostics."""
        return max(self.competency_ids, key=self.sd)


class PriorOnlyBelief(BeliefState):
    """Phase 0 scaffolding: holds the prior and never learns from evidence.

    It exists so the loop, the trace store and the report generator can be
    built and tested before the inference machinery lands. It is deliberately
    *not* a plausible policy input — an arm running on this would score at
    chance, which is exactly what the Phase 2 tests will demonstrate when the
    grid posterior replaces it.
    """

    def __init__(self, rubric: Rubric) -> None:
        super().__init__(rubric)
        self._mean = {c.id: c.prior_mean for c in rubric.competencies}
        self._var = {c.id: c.prior_var for c in rubric.competencies}
        self._scores: dict[str, list[int]] = {c.id: [] for c in rubric.competencies}

    def mean(self, competency_id: str) -> float:
        return self._mean[competency_id]

    def sd(self, competency_id: str) -> float:
        return math.sqrt(self._var[competency_id])

    def entropy(self, competency_id: str) -> float:
        # Differential entropy of a Gaussian, in nats.
        return 0.5 * math.log(2 * math.pi * math.e * self._var[competency_id])

    def credible_interval(self, competency_id: str, mass: float = 0.8) -> tuple[float, float]:
        z = {0.5: 0.674, 0.8: 1.282, 0.9: 1.645, 0.95: 1.960}.get(mass, 1.282)
        m, s = self.mean(competency_id), self.sd(competency_id)
        return (m - z * s, m + z * s)

    def update(self, question: Question, score: int) -> None:
        # Records the observation for auditability but moves no probability
        # mass. Superseded in Phase 2.
        self._scores[question.competency_id].append(score)
        self.n_observations[question.competency_id] += 1

    def observed_scores(self, competency_id: str) -> list[int]:
        return list(self._scores[competency_id])
