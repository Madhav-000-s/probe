"""Graded response model correctness.

Checked against hand-computed fixtures and algebraic identities, not against
"the plot looks right". Every later number in this project is downstream of
these five functions, so an error here would propagate silently into recovery,
efficiency and calibration alike.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from probe.belief.grm import (
    N_CATEGORIES,
    category_probs,
    expected_score,
    information,
    log_likelihood,
    p_at_least,
    sample_score,
    sigmoid,
)

thetas = st.floats(min_value=-4.0, max_value=4.0, allow_nan=False, allow_infinity=False)
discriminations = st.floats(min_value=0.05, max_value=3.0, allow_nan=False)


@st.composite
def valid_thresholds(draw):
    """Four strictly increasing thresholds, as ``GRMParams`` guarantees."""
    base = draw(st.floats(min_value=-3.0, max_value=0.0))
    gaps = [draw(st.floats(min_value=0.05, max_value=1.5)) for _ in range(3)]
    out = [base]
    for gap in gaps:
        out.append(out[-1] + gap)
    return out


# ------------------------------------------------------------------ sigmoid


def test_sigmoid_is_stable_at_extremes():
    """The naive form overflows well before |x| = 700, and a theta grid crossed
    with extreme thresholds hits that routinely."""
    extreme = np.array([-1e4, -800.0, -50.0, 0.0, 50.0, 800.0, 1e4])
    out = sigmoid(extreme)
    assert np.all(np.isfinite(out))
    assert np.all((out >= 0.0) & (out <= 1.0))
    assert out[3] == pytest.approx(0.5)
    assert out[0] == pytest.approx(0.0, abs=1e-12)
    assert out[-1] == pytest.approx(1.0, abs=1e-12)


def test_sigmoid_matches_the_closed_form_where_it_is_safe():
    x = np.linspace(-30, 30, 121)
    assert np.allclose(sigmoid(x), 1.0 / (1.0 + np.exp(-x)), atol=1e-12)


# ------------------------------------------------------- category structure


@settings(max_examples=400, deadline=None)
@given(theta=thetas, a=discriminations, b=valid_thresholds())
def test_category_probabilities_sum_to_one(theta, a, b):
    """Property over 400 random (a, b, theta) draws.

    Tight tolerance on purpose: the categories are a telescoping difference of
    boundary probabilities, so they sum to 1 exactly up to floating point. A
    loose tolerance here would hide a real indexing error.
    """
    probs = category_probs(theta, a, b)
    assert probs.shape == (1, N_CATEGORIES)
    assert probs.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.all(probs >= 0.0)


@settings(max_examples=200, deadline=None)
@given(theta=thetas, a=discriminations, b=valid_thresholds())
def test_boundary_probabilities_are_decreasing(theta, a, b):
    """P(>=2) >= P(>=3) >= P(>=4) >= P(>=5). If this fails the category
    differences go negative."""
    ge = p_at_least(theta, a, b)[0]
    assert np.all(np.diff(ge) <= 1e-12)


def test_category_probabilities_match_hand_computed_fixtures():
    """Reference values computed by hand from the definition, to 1e-9.

    a = 1.0, b = (-1, 0, 1, 2), theta = 0.
      P(>=2) = sigmoid(1)  = 0.7310585786300049
      P(>=3) = sigmoid(0)  = 0.5
      P(>=4) = sigmoid(-1) = 0.2689414213699951
      P(>=5) = sigmoid(-2) = 0.11920292202211755
    Categories are the successive differences, with P(>=1)=1 and P(>=6)=0.
    """
    a, b = 1.0, [-1.0, 0.0, 1.0, 2.0]
    s = lambda x: 1.0 / (1.0 + math.exp(-x))  # noqa: E731
    expected = [
        1.0 - s(1.0),
        s(1.0) - s(0.0),
        s(0.0) - s(-1.0),
        s(-1.0) - s(-2.0),
        s(-2.0),
    ]
    got = category_probs(0.0, a, b)[0]
    assert got == pytest.approx(expected, abs=1e-9)
    assert sum(expected) == pytest.approx(1.0, abs=1e-12)


def test_symmetric_item_is_symmetric_about_its_centre():
    """With thresholds symmetric around zero, theta = 0 must give a
    distribution symmetric in the categories."""
    probs = category_probs(0.0, 1.4, [-1.5, -0.5, 0.5, 1.5])[0]
    assert probs[0] == pytest.approx(probs[4], abs=1e-12)
    assert probs[1] == pytest.approx(probs[3], abs=1e-12)


def test_vectorises_over_a_theta_grid():
    grid = np.linspace(-3, 3, 61)
    probs = category_probs(grid, 1.2, [-1.5, -0.5, 0.5, 1.5])
    assert probs.shape == (61, 5)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-12)
    for i, theta in enumerate(grid):
        assert probs[i] == pytest.approx(category_probs(float(theta), 1.2, [-1.5, -0.5, 0.5, 1.5])[0])


def test_rejects_wrong_number_of_thresholds():
    with pytest.raises(ValueError, match="expected 4 thresholds"):
        category_probs(0.0, 1.0, [0.0, 1.0])


# ------------------------------------------------------------- monotonicity


@settings(max_examples=150, deadline=None)
@given(a=discriminations, b=valid_thresholds())
def test_expected_score_is_non_decreasing_in_ability(a, b):
    grid = np.linspace(-3.5, 3.5, 71)
    e = expected_score(grid, a, b)
    assert np.all(np.diff(e) >= -1e-12)
    assert np.all((e >= 1.0 - 1e-9) & (e <= N_CATEGORIES + 1e-9))


def test_higher_discrimination_sharpens_the_response_curve():
    """The defining property of discrimination: a steeper item separates
    ability better."""
    grid = np.linspace(-3, 3, 61)
    b = [-1.5, -0.5, 0.5, 1.5]
    flat = expected_score(grid, 0.4, b)
    steep = expected_score(grid, 2.5, b)
    assert (steep.max() - steep.min()) > (flat.max() - flat.min())


def test_shifting_thresholds_shifts_the_curve():
    grid = np.linspace(-3, 3, 61)
    easy = expected_score(grid, 1.5, [-2.0, -1.0, 0.0, 1.0])
    hard = expected_score(grid, 1.5, [0.0, 1.0, 2.0, 3.0])
    assert np.all(easy >= hard - 1e-12)


# -------------------------------------------------------------- likelihood


@settings(max_examples=100, deadline=None)
@given(score=st.integers(min_value=1, max_value=5), a=discriminations, b=valid_thresholds())
def test_log_likelihood_is_finite_and_matches_the_categories(score, a, b):
    grid = np.linspace(-3, 3, 61)
    ll = log_likelihood(score, grid, a, b)
    assert ll.shape == (61,)
    assert np.all(np.isfinite(ll))
    direct = np.log(np.maximum(category_probs(grid, a, b)[:, score - 1], 1e-300))
    assert np.allclose(ll, direct, atol=1e-12)


def test_log_likelihood_floors_rather_than_returning_negative_infinity():
    """An impossible-looking category must not poison the whole posterior with
    a -inf that propagates to every grid point after normalising."""
    ll = log_likelihood(5, np.array([-3.0]), 3.0, [2.0, 2.5, 3.0, 3.5])
    assert np.all(np.isfinite(ll))


# ---------------------------------------------------------------- sampling


def test_sampled_scores_match_the_category_distribution():
    """Law of large numbers against the closed form."""
    a, b, theta = 1.3, [-1.2, -0.3, 0.4, 1.4], 0.35
    rng = random.Random(11)
    draws = [sample_score(theta, a, b, rng) for _ in range(40_000)]
    empirical = np.array([draws.count(k) for k in range(1, 6)], dtype=float) / len(draws)
    assert empirical == pytest.approx(category_probs(theta, a, b)[0], abs=0.01)


def test_sampling_is_bounded_and_seeded():
    a, b = 1.5, [-1.5, -0.5, 0.5, 1.5]
    first = [sample_score(0.5, a, b, random.Random(3)) for _ in range(20)]
    second = [sample_score(0.5, a, b, random.Random(3)) for _ in range(20)]
    assert first == second
    assert all(1 <= s <= 5 for s in first)


# ------------------------------------------------------------- information


@settings(max_examples=120, deadline=None)
@given(a=discriminations, b=valid_thresholds())
def test_fisher_information_is_non_negative(a, b):
    assume(a > 0.05)
    info = information(np.linspace(-3, 3, 61), a, b)
    assert np.all(info >= -1e-12)
    assert np.all(np.isfinite(info))


def test_information_peaks_near_the_thresholds():
    """An item is most informative about ability near where it discriminates,
    not out in the tails."""
    grid = np.linspace(-3, 3, 121)
    info = information(grid, 1.8, [-0.6, -0.2, 0.2, 0.6])
    peak = grid[int(np.argmax(info))]
    assert abs(peak) < 0.75
    assert info[0] < info[int(np.argmax(info))]
