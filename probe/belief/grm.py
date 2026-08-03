"""Samejima's graded response model.

Answers are scored on a 5-point rubric, so the response model has to be
ordinal. Binarising to pass/fail and using a 2PL throws away most of the
information the policy needs — the difference between a 2 and a 4 is precisely
what tells you where somebody sits.

.. math::

    P(\\text{score} \\ge k \\mid \\theta) = \\sigma(a (\\theta - b_k)),
        \\quad k = 2 \\ldots 5

    P(\\text{score} = k \\mid \\theta)
        = P(\\ge k \\mid \\theta) - P(\\ge k{+}1 \\mid \\theta)

with :math:`P(\\ge 1) = 1` and :math:`P(\\ge 6) = 0` by definition.

This module is pure NumPy and has no notion of a candidate, an interview or a
trace. It is used from both planes — the simulator draws responses from it,
the belief state takes its likelihood from it — which is deliberate: the
recovery question is "can the estimator find theta given responses generated
this way", and using two different response models would be answering a
different question without saying so.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

N_CATEGORIES = 5


def sigmoid(x: NDArray[np.float64] | float) -> NDArray[np.float64]:
    """Numerically stable logistic.

    The naive form overflows for |x| beyond ~700 and produces warnings well
    before that; a grid over theta crossed with extreme thresholds hits this
    routinely.
    """
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def p_at_least(theta: NDArray[np.float64] | float, a: float, b: list[float] | NDArray[np.float64]):
    """``P(score >= k)`` for k = 2..5.

    Returns shape ``(..., 4)`` where the leading axes follow ``theta``.
    """
    theta_arr = np.atleast_1d(np.asarray(theta, dtype=float))
    b_arr = np.asarray(b, dtype=float)
    if b_arr.shape != (N_CATEGORIES - 1,):
        raise ValueError(f"expected 4 thresholds, got shape {b_arr.shape}")
    return sigmoid(a * (theta_arr[..., None] - b_arr[None, ...]))


def category_probs(
    theta: NDArray[np.float64] | float, a: float, b: list[float] | NDArray[np.float64]
) -> NDArray[np.float64]:
    """``P(score = k)`` for k = 1..5. Shape ``(n_theta, 5)``.

    Rows sum to exactly 1 by construction (a telescoping difference of
    boundary probabilities), which is why the property test asserting that can
    use a tight tolerance rather than a forgiving one.
    """
    ge = p_at_least(theta, a, b)  # (n, 4) -> P(>=2) .. P(>=5)
    n = ge.shape[0]
    boundaries = np.empty((n, N_CATEGORIES + 1), dtype=float)
    boundaries[:, 0] = 1.0  # P(>= 1)
    boundaries[:, 1:N_CATEGORIES] = ge
    boundaries[:, N_CATEGORIES] = 0.0  # P(>= 6)
    probs = boundaries[:, :-1] - boundaries[:, 1:]
    # Thresholds are validated strictly increasing at the model boundary, so
    # negatives here would mean a caller bypassed GRMParams. Clip rather than
    # raise: a degenerate item should not take down a sweep.
    return np.clip(probs, 0.0, 1.0)


def expected_score(
    theta: NDArray[np.float64] | float, a: float, b: list[float] | NDArray[np.float64]
) -> NDArray[np.float64]:
    """E[score | theta]. Monotone non-decreasing in theta for any valid item."""
    probs = category_probs(theta, a, b)
    levels = np.arange(1, N_CATEGORIES + 1, dtype=float)
    return probs @ levels


def log_likelihood(
    score: int, theta: NDArray[np.float64], a: float, b: list[float] | NDArray[np.float64]
) -> NDArray[np.float64]:
    """log P(score | theta) over a theta grid, floored to keep -inf out of the
    posterior when an item makes an observed category vanishingly unlikely."""
    probs = category_probs(theta, a, b)[:, score - 1]
    return np.log(np.maximum(probs, 1e-300))


def sample_score(theta: float, a: float, b: list[float], rng) -> int:
    """Draw one graded response. ``rng`` is a ``random.Random`` or anything
    exposing ``random()``, so the simulator stays seeded end to end."""
    probs = category_probs(theta, a, b)[0]
    u = rng.random()
    cumulative = 0.0
    for k, p in enumerate(probs, start=1):
        cumulative += float(p)
        if u <= cumulative:
            return k
    return N_CATEGORIES


def information(
    theta: NDArray[np.float64] | float, a: float, b: list[float] | NDArray[np.float64]
) -> NDArray[np.float64]:
    """Fisher information for the item at theta.

    Not used by the EIG policy — that works with entropies directly — but it is
    the classical adaptive-testing selection criterion, so having it here makes
    the "is EIG doing anything a maximum-information rule would not" question
    answerable rather than rhetorical.
    """
    ge = p_at_least(theta, a, b)
    padded = np.concatenate(
        [np.ones((ge.shape[0], 1)), ge, np.zeros((ge.shape[0], 1))], axis=1
    )
    probs = padded[:, :-1] - padded[:, 1:]
    d_ge = a * ge * (1.0 - ge)
    d_padded = np.concatenate(
        [np.zeros((ge.shape[0], 1)), d_ge, np.zeros((ge.shape[0], 1))], axis=1
    )
    d_probs = d_padded[:, :-1] - d_padded[:, 1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(probs > 1e-12, d_probs**2 / np.maximum(probs, 1e-12), 0.0)
    return terms.sum(axis=1)
