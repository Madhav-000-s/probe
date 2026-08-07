"""Cross-competency correlation — the ``eig+corr`` ablation.

Evidence about ``databases.indexing`` should say something about
``databases.query_optimization``. The independent grid posteriors cannot
express that, so this module layers a Gaussian copula on top: each grid
posterior is summarised by its mean and SD, the multivariate-normal
conditional update is applied to the *other* competencies, and the result is
folded back onto their grids as a Gaussian reweighting.

It is an approximation and is documented as one. A true joint posterior over
fourteen competencies would need a fourteen-dimensional grid, which is the
cost the architecture explicitly declined to pay.

**Provenance matters here more than anywhere else.** The correlation matrix is
estimated from the *calibration split only*. Estimating it from the eval split
would let the arm exploit structure measured on the data it is being scored
against, which is the circularity objection this project is at pains to kill.
The estimate carries its provenance and a test asserts it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from probe.belief.grid import GridBelief, normalise
from probe.models import Question

#: Correlations weaker than this are treated as zero. Sampling noise on a
#: 36-respondent calibration split produces plenty of small spurious values,
#: and propagating them adds variance without adding signal.
MIN_ABS_CORRELATION = 0.20
#: Ceiling on how much of another competency's update is borrowed. Even a
#: correlation of 0.9 does not make one competency a substitute for another,
#: and an uncapped copula will happily talk itself into confidence it has not
#: earned.
MAX_BORROW = 0.60


@dataclass
class CompetencyCorrelation:
    competency_ids: list[str]
    matrix: np.ndarray
    #: "calibration" — recorded so a results table can prove where it came from.
    provenance: str
    n_respondents: int

    def rho(self, a: str, b: str) -> float:
        try:
            i, j = self.competency_ids.index(a), self.competency_ids.index(b)
        except ValueError:
            return 0.0
        return float(self.matrix[i, j])

    def neighbours(self, competency_id: str, among: list[str]) -> list[tuple[str, float]]:
        out = []
        for other in among:
            if other == competency_id:
                continue
            r = self.rho(competency_id, other)
            if abs(r) >= MIN_ABS_CORRELATION:
                out.append((other, r))
        return sorted(out, key=lambda kv: -abs(kv[1]))

    def to_dict(self) -> dict:
        return {
            "competency_ids": self.competency_ids,
            "matrix": self.matrix.tolist(),
            "provenance": self.provenance,
            "n_respondents": self.n_respondents,
        }

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> CompetencyCorrelation:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            competency_ids=raw["competency_ids"],
            matrix=np.asarray(raw["matrix"], dtype=float),
            provenance=raw["provenance"],
            n_respondents=int(raw["n_respondents"]),
        )


def estimate_correlation(
    score_matrix: np.ndarray,
    competency_ids: list[str],
    provenance: str = "calibration",
) -> CompetencyCorrelation:
    """Empirical Pearson correlation over mean competency scores.

    Pairwise-complete, because not every persona answers every competency and
    dropping incomplete rows would discard most of a modest calibration split.
    """
    n = len(competency_ids)
    matrix = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            xi, xj = score_matrix[:, i], score_matrix[:, j]
            mask = np.isfinite(xi) & np.isfinite(xj)
            if mask.sum() < 8 or np.std(xi[mask]) < 1e-9 or np.std(xj[mask]) < 1e-9:
                continue
            r = float(np.corrcoef(xi[mask], xj[mask])[0, 1])
            matrix[i, j] = matrix[j, i] = 0.0 if not np.isfinite(r) else r
    return CompetencyCorrelation(
        competency_ids=list(competency_ids),
        matrix=matrix,
        provenance=provenance,
        n_respondents=int(score_matrix.shape[0]),
    )


class CorrelatedGridBelief(GridBelief):
    """Grid posteriors coupled by a Gaussian copula.

    On each update the targeted competency gets the exact grid update; every
    correlated neighbour then receives the multivariate-normal conditional
    implied by the observed shift, applied as a Gaussian reweighting of its own
    grid.
    """

    def __init__(self, rubric, correlation: CompetencyCorrelation | None = None) -> None:
        super().__init__(rubric)
        self.correlation = correlation
        self.borrowed_updates = 0

    def update(self, question: Question, score: int) -> None:
        cid = question.competency_id
        if self.correlation is None or cid not in self._pmf:
            super().update(question, score)
            return

        before_mean, before_sd = self.mean(cid), self.sd(cid)
        super().update(question, score)
        after_mean, after_sd = self.mean(cid), self.sd(cid)

        if before_sd <= 1e-9:
            return
        shift = after_mean - before_mean
        variance_ratio = (after_sd / before_sd) ** 2

        for other, rho in self.correlation.neighbours(cid, self.competency_ids):
            weight = min(abs(rho), MAX_BORROW) * np.sign(rho)
            other_sd = self.sd(other)

            # MVN conditional: the neighbour's mean moves by rho times the
            # observed shift, rescaled by the ratio of their SDs, and its
            # variance contracts by rho^2 times the contraction just observed.
            delta = float(weight) * shift * (other_sd / before_sd)
            contraction = 1.0 - (weight**2) * (1.0 - variance_ratio)
            implied_var = max((other_sd**2) * contraction, 1e-4)

            target_mean = self.mean(other) + delta
            message = -0.5 * (self.grid - target_mean) ** 2 / max(implied_var, 1e-4)
            # Blend rather than replace: the neighbour's own evidence is not
            # overwritten by a borrowed message.
            blended = normalise(np.log(np.maximum(self._pmf[other], 1e-300)) + 0.5 * message)
            self.set_pmf(other, blended)
            self.borrowed_updates += 1
