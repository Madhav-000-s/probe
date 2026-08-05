"""The grid posterior.

Per competency, ability ``theta`` lives on a fixed 61-point grid over
[-3, 3]. A graded response updates the posterior by one vectorised
multiply-and-normalise. That is the entire inference engine.

Why a grid rather than MCMC or a Laplace approximation: in one dimension a
61-point grid is *exact enough* — 0.1 resolution in theta, far finer than the
standard error any realistic number of items achieves — it has no sampler to
tune or diagnose, and an update costs microseconds, which matters when the EIG
policy evaluates every item in the bank at every turn. The cost is that it does
not extend to a joint multivariate posterior; the ``eig+corr`` arm layers a
copula on top rather than gridding a 14-dimensional space.

Numerical work happens in log space. Multiplying many small likelihoods in
linear space underflows quietly, and a posterior that has silently become all
zeros still returns a mean — it is just the wrong one.
"""

from __future__ import annotations

import numpy as np

from probe.belief.grm import log_likelihood
from probe.belief.state import BeliefState
from probe.config import THETA_MAX, THETA_MIN, THETA_POINTS
from probe.models import Question, Rubric

#: The grid, shared by every competency and every run.
THETA_GRID: np.ndarray = np.linspace(THETA_MIN, THETA_MAX, THETA_POINTS)
GRID_STEP: float = float(THETA_GRID[1] - THETA_GRID[0])


def gaussian_log_prior(mean: float, var: float, grid: np.ndarray = THETA_GRID) -> np.ndarray:
    """Unnormalised log density. The constant term is dropped because the
    posterior is normalised immediately afterwards."""
    return -0.5 * (grid - mean) ** 2 / var


def normalise(log_weights: np.ndarray) -> np.ndarray:
    """Log weights -> a probability mass function over the grid.

    The max-subtraction is not optional. After a dozen updates the raw log
    weights are around -300, and ``exp`` of that is zero in float64.
    """
    shifted = log_weights - log_weights.max()
    weights = np.exp(shifted)
    total = weights.sum()
    if total <= 0 or not np.isfinite(total):  # pragma: no cover - defensive
        return np.full_like(weights, 1.0 / weights.size)
    return weights / total


def pmf_mean(pmf: np.ndarray, grid: np.ndarray = THETA_GRID) -> float:
    return float(pmf @ grid)


def pmf_sd(pmf: np.ndarray, grid: np.ndarray = THETA_GRID) -> float:
    mean = pmf_mean(pmf, grid)
    var = float(pmf @ (grid - mean) ** 2)
    return float(np.sqrt(max(var, 0.0)))


def pmf_entropy(pmf: np.ndarray) -> float:
    """Shannon entropy of the grid pmf, in nats.

    Discrete rather than differential entropy. Every posterior in the system
    lives on the same grid, so the ``log(step)`` offset between the two is a
    constant that cancels in every difference the policy takes. Using the
    discrete form keeps entropies non-negative, which makes a negative EIG an
    unambiguous bug rather than a plausible artefact.
    """
    nz = pmf[pmf > 0]
    return float(-(nz * np.log(nz)).sum())


def pmf_interval(pmf: np.ndarray, mass: float = 0.8, grid: np.ndarray = THETA_GRID):
    """Central credible interval by linear interpolation of the CDF.

    The grid points are bin *centres*, so ``cumsum(pmf)[i]`` is the probability
    of falling at or below the **right edge** of bin ``i``, not at its centre.
    Interpolating against the centres therefore shifts both endpoints down by
    half a step — a systematic bias, not a rounding wobble, and one that would
    have quietly skewed every credible interval in every report and shown up
    later as a calibration failure with no obvious cause. Interpolate against
    the edges.
    """
    step = float(grid[1] - grid[0])
    edges = grid + step / 2.0
    cdf = np.cumsum(pmf)
    lo_target = (1.0 - mass) / 2.0
    hi_target = 1.0 - lo_target
    return (
        float(np.interp(lo_target, cdf, edges)),
        float(np.interp(hi_target, cdf, edges)),
    )


class GridBelief(BeliefState):
    """Independent grid posteriors, one per competency.

    Independence is a modelling choice, and a limitation stated in the README:
    evidence about ``algorithms.complexity`` does not move
    ``algorithms.data_structures`` here. The ``eig+corr`` arm relaxes it.
    """

    def __init__(self, rubric: Rubric, grid: np.ndarray = THETA_GRID) -> None:
        super().__init__(rubric)
        self.grid = grid
        self._log_post: dict[str, np.ndarray] = {
            c.id: gaussian_log_prior(c.prior_mean, c.prior_var, grid)
            for c in rubric.competencies
        }
        self._pmf: dict[str, np.ndarray] = {
            cid: normalise(lp) for cid, lp in self._log_post.items()
        }

    # ------------------------------------------------------------ read side

    def pmf(self, competency_id: str) -> np.ndarray:
        return self._pmf[competency_id]

    def mean(self, competency_id: str) -> float:
        return pmf_mean(self._pmf[competency_id], self.grid)

    def sd(self, competency_id: str) -> float:
        return pmf_sd(self._pmf[competency_id], self.grid)

    def entropy(self, competency_id: str) -> float:
        return pmf_entropy(self._pmf[competency_id])

    def credible_interval(self, competency_id: str, mass: float = 0.8) -> tuple[float, float]:
        return pmf_interval(self._pmf[competency_id], mass, self.grid)

    # ----------------------------------------------------------- write side

    def update(self, question: Question, score: int) -> None:
        cid = question.competency_id
        if cid not in self._log_post:
            # A follow-up may target a competency outside the rubric if the
            # bank drifts. Ignore rather than raise: losing one observation is
            # recoverable, losing the run is not.
            return
        self._log_post[cid] = self._log_post[cid] + log_likelihood(
            score, self.grid, question.grm.a, question.grm.b
        )
        self._pmf[cid] = normalise(self._log_post[cid])
        self.n_observations[cid] += 1

    def set_pmf(self, competency_id: str, pmf: np.ndarray) -> None:
        """Overwrite a posterior directly.

        Used by the correlation arm, which computes a coupled update outside
        this class and writes the result back. Kept explicit rather than
        letting that arm reach into ``_pmf``.
        """
        pmf = np.asarray(pmf, dtype=float)
        self._pmf[competency_id] = pmf / pmf.sum()
        self._log_post[competency_id] = np.log(np.maximum(self._pmf[competency_id], 1e-300))
