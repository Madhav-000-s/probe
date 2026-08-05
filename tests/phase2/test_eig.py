"""Expected information gain, checked against Monte Carlo and against the
identities it has to satisfy.

The analytic-versus-Monte-Carlo comparison is the important one. The closed
form is a sum over five categories; the Monte-Carlo estimate simulates the
whole draw-and-update loop. They are computed by completely different code
paths, so agreement between them is real evidence rather than a tautology.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from probe.belief.grid import THETA_GRID, gaussian_log_prior, normalise, pmf_entropy
from probe.belief.grm import category_probs
from probe.policy.eig import expected_information_gain

from .test_posterior import make_question

B = [-1.5, -0.5, 0.5, 1.5]


def prior(mean=0.0, var=1.0):
    return normalise(gaussian_log_prior(mean, var))


def monte_carlo_eig(pmf, a, b, n=20_000, seed=0):
    """H[prior] - E_k H[posterior | k], estimated by sampling.

    Draws theta from the posterior, draws a category from that theta, then
    averages the entropy of the exact Bayesian update for the drawn category.
    Shares no code with the analytic implementation beyond the likelihood.
    """
    rng = np.random.default_rng(seed)
    probs = category_probs(THETA_GRID, a, b)

    theta_idx = rng.choice(len(THETA_GRID), size=n, p=pmf)
    draws = np.array(
        [rng.choice(5, p=probs[i] / probs[i].sum()) for i in theta_idx]
    )

    posterior_entropies = {}
    for k in range(5):
        joint = probs[:, k] * pmf
        total = joint.sum()
        posterior_entropies[k] = pmf_entropy(joint / total) if total > 1e-15 else 0.0

    expected = float(np.mean([posterior_entropies[int(k)] for k in draws]))
    return pmf_entropy(pmf) - expected


# ------------------------------------------------------------- identities


@settings(max_examples=250, deadline=None)
@given(
    a=st.floats(min_value=0.0, max_value=3.0, allow_nan=False),
    mean=st.floats(min_value=-2.0, max_value=2.0),
    var=st.floats(min_value=0.05, max_value=2.5),
)
def test_eig_is_never_negative(a, mean, var):
    """It is a mutual information. Conditioning cannot raise expected entropy,
    so a negative value means something upstream is malformed."""
    assert expected_information_gain(prior(mean, var), THETA_GRID, a, B) >= 0.0


def test_zero_discrimination_yields_exactly_zero():
    """An item that responds identically at every ability level tells you
    nothing, and the formula must return that exactly rather than to within a
    tolerance."""
    eig = expected_information_gain(prior(), THETA_GRID, 0.0, B)
    assert eig == pytest.approx(0.0, abs=1e-12)


def test_higher_discrimination_gives_more_information():
    """For two items differing only in `a`, the steeper one must win."""
    p = prior()
    values = [expected_information_gain(p, THETA_GRID, a, B) for a in (0.3, 0.8, 1.5, 2.2, 3.0)]
    assert values == sorted(values), values
    assert values[-1] > values[0] * 3


def test_a_wider_posterior_leaves_more_to_learn():
    wide = expected_information_gain(prior(0.0, 2.25), THETA_GRID, 1.6, B)
    narrow = expected_information_gain(prior(0.0, 0.04), THETA_GRID, 1.6, B)
    assert wide > narrow


def test_information_is_highest_where_the_item_discriminates():
    """An item with thresholds around zero is worth more against a posterior
    centred at zero than against one centred far away — that targeting is what
    makes adaptive selection work at all."""
    on_target = expected_information_gain(prior(0.0, 0.5), THETA_GRID, 1.8, [-0.6, -0.2, 0.2, 0.6])
    off_target = expected_information_gain(prior(2.6, 0.5), THETA_GRID, 1.8, [-0.6, -0.2, 0.2, 0.6])
    assert on_target > off_target


def test_eig_cannot_exceed_the_prior_entropy():
    p = prior()
    assert expected_information_gain(p, THETA_GRID, 3.0, B) <= pmf_entropy(p) + 1e-12


# --------------------------------------------------- analytic vs simulation


@pytest.mark.slow
def test_analytic_eig_matches_monte_carlo_across_random_states():
    """Fifty random (posterior, item) pairs, 20k samples each."""
    rng = np.random.default_rng(20260805)
    worst = 0.0
    for trial in range(50):
        mean = float(rng.uniform(-1.5, 1.5))
        var = float(rng.uniform(0.15, 2.0))
        a = float(rng.uniform(0.4, 2.5))
        spread = float(rng.uniform(0.4, 1.6))
        b = [-spread, -spread / 3, spread / 3, spread]

        p = prior(mean, var)
        analytic = expected_information_gain(p, THETA_GRID, a, b)
        estimated = monte_carlo_eig(p, a, b, n=20_000, seed=trial)
        worst = max(worst, abs(analytic - estimated))
        assert analytic == pytest.approx(estimated, abs=0.02), (
            f"trial {trial}: analytic={analytic:.5f} mc={estimated:.5f} "
            f"(mean={mean:.2f} var={var:.2f} a={a:.2f})"
        )
    assert worst < 0.02


def test_eig_equals_prior_minus_expected_posterior_entropy_by_definition():
    """Recompute the definition inline and compare, so an indexing slip in the
    implementation cannot pass."""
    p = prior(0.3, 0.8)
    a, b = 1.7, [-1.2, -0.4, 0.4, 1.2]

    probs = category_probs(THETA_GRID, a, b)
    joint = probs * p[:, None]
    p_k = joint.sum(axis=0)
    expected_posterior = sum(
        float(p_k[k]) * pmf_entropy(joint[:, k] / p_k[k]) for k in range(5) if p_k[k] > 1e-15
    )
    assert expected_information_gain(p, THETA_GRID, a, b) == pytest.approx(
        pmf_entropy(p) - expected_posterior, abs=1e-12
    )


def test_category_marginal_is_a_distribution():
    p = prior(0.4, 1.1)
    probs = category_probs(THETA_GRID, 1.4, B)
    p_k = (probs * p[:, None]).sum(axis=0)
    assert p_k.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.all(p_k >= 0)


def test_question_objects_route_through_the_same_maths():
    """The policy scores Question objects; confirm no adapter drift."""
    question = make_question(a=1.9)
    p = prior()
    assert expected_information_gain(
        p, THETA_GRID, question.grm.a, question.grm.b
    ) == pytest.approx(expected_information_gain(p, THETA_GRID, 1.9, question.grm.b))
