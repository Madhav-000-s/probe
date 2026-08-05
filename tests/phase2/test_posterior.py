"""The grid posterior, verified against simulation ground truth.

The coverage test is the one that matters. It is the calibration machinery
testing itself before it is ever pointed at a persona: if an 80% credible
interval does not contain the truth about 80% of the time, then every interval
in every report is a lie, and the ECE number in the results table would be
measuring a broken estimator rather than a working one.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from probe.belief.grid import (
    GRID_STEP,
    THETA_GRID,
    GridBelief,
    gaussian_log_prior,
    normalise,
    pmf_entropy,
    pmf_interval,
    pmf_mean,
    pmf_sd,
)
from probe.belief.grm import sample_score
from probe.models import (
    Competency,
    GRMParams,
    ProbeFamily,
    Question,
    Rubric,
    RubricAnchor,
)


def make_question(cid="c0", a=1.6, b=(-1.5, -0.5, 0.5, 1.5), qid="q0"):
    return Question(
        id=qid,
        competency_id=cid,
        probe_family=ProbeFamily.SCENARIO,
        text="q",
        anchors=[RubricAnchor(level=i, descriptor=f"d{i}") for i in range(1, 6)],
        grm=GRMParams(a=a, b=list(b)),
        expected_seconds=60.0,
    )


def make_rubric(prior_mean=0.0, prior_var=1.0, ids=("c0",)):
    return Rubric(
        candidate_id="c",
        role_title="r",
        competencies=[
            Competency(
                id=cid,
                label=cid,
                required_level=4,
                evidence_in_resume=0.0,
                prior_mean=prior_mean,
                prior_var=prior_var,
                probe_families=[ProbeFamily.SCENARIO],
            )
            for cid in ids
        ],
        taxonomy_version="v1",
    )


# ----------------------------------------------------------------- the grid


def test_grid_shape_matches_the_architecture():
    assert len(THETA_GRID) == 61
    assert THETA_GRID[0] == pytest.approx(-3.0)
    assert THETA_GRID[-1] == pytest.approx(3.0)
    assert GRID_STEP == pytest.approx(0.1)


def test_normalise_survives_extreme_log_weights():
    """After a dozen updates the raw log weights sit near -300 and exp() of
    that is zero in float64. Max-subtraction is what keeps this alive."""
    pmf = normalise(np.full(61, -5000.0) + gaussian_log_prior(0.0, 1.0))
    assert pmf.sum() == pytest.approx(1.0)
    assert np.all(np.isfinite(pmf))
    assert pmf.argmax() == 30


def test_prior_matches_the_analytic_truncated_normal():
    """The grid prior is a *truncated* Gaussian, and that is correct rather
    than a defect: theta genuinely cannot leave [-3, 3] in this model, so a
    prior that put mass outside would be asserting something the parameter
    space does not allow.

    Comparing against ``scipy.stats.truncnorm`` separates the two error
    sources cleanly. Truncation is intended and shows up in the reference;
    discretisation is the grid's own error and is what this test bounds. A
    naive comparison against the *untruncated* moments would conflate them and
    make a correct implementation look 4% wrong at the widest prior.
    """
    from scipy import stats

    # (mean, var, tolerance). The last case is deliberately pathological: a
    # wide prior centred half a step from the boundary puts most of its mass
    # into a handful of bins where the density changes fast, and the
    # bin-centre (rectangle) approximation is correspondingly worse. 0.026 of
    # residual error there is the grid's honest discretisation cost, measured
    # rather than assumed. No compiled prior goes anywhere near it.
    cases = (
        (0.0, 1.0, 0.01),
        (0.65, 0.36, 0.01),
        (-0.20, 1.44, 0.01),
        (0.3, 0.9, 0.01),
        (-2.5, 1.44, 0.03),
    )
    for mean, var, tol in cases:
        sd = np.sqrt(var)
        reference = stats.truncnorm(
            (THETA_GRID[0] - mean) / sd, (THETA_GRID[-1] - mean) / sd, loc=mean, scale=sd
        )
        pmf = normalise(gaussian_log_prior(mean, var))
        assert pmf_mean(pmf) == pytest.approx(reference.mean(), abs=tol), f"mean at {mean=}"
        assert pmf_sd(pmf) == pytest.approx(reference.std(), abs=tol), f"sd at {mean=}"


def test_priors_well_inside_the_grid_lose_nothing():
    """The priors the compiler actually emits — means in [-0.20, +0.65],
    variances in [0.36, 1.44] — sit far enough inside that truncation is
    negligible for the mean."""
    for mean, var in ((0.0, 1.0), (0.65, 0.36), (0.3, 0.9)):
        pmf = normalise(gaussian_log_prior(mean, var))
        assert pmf_mean(pmf) == pytest.approx(mean, abs=0.02)
        assert pmf_sd(pmf) == pytest.approx(np.sqrt(var), abs=0.07)


def test_entropy_is_non_negative_and_ordered_by_width():
    wide = normalise(gaussian_log_prior(0.0, 1.44))
    narrow = normalise(gaussian_log_prior(0.0, 0.16))
    assert pmf_entropy(narrow) >= 0.0
    assert pmf_entropy(wide) > pmf_entropy(narrow)


def test_interval_is_centred_and_ordered():
    pmf = normalise(gaussian_log_prior(0.0, 1.0))
    lo80, hi80 = pmf_interval(pmf, 0.8)
    lo50, hi50 = pmf_interval(pmf, 0.5)
    assert lo80 < lo50 < hi50 < hi80
    # Standard normal 80% interval is +-1.2816.
    assert lo80 == pytest.approx(-1.2816, abs=0.05)
    assert hi80 == pytest.approx(1.2816, abs=0.05)


# ------------------------------------------------------------ update sanity


def test_update_moves_the_posterior_toward_the_evidence():
    belief = GridBelief(make_rubric())
    before = belief.mean("c0")
    for _ in range(6):
        belief.update(make_question(), 5)
    assert belief.mean("c0") > before + 0.5

    belief2 = GridBelief(make_rubric())
    for _ in range(6):
        belief2.update(make_question(), 1)
    assert belief2.mean("c0") < before - 0.5


def test_update_narrows_the_posterior():
    belief = GridBelief(make_rubric())
    sds = [belief.sd("c0")]
    for i in range(10):
        belief.update(make_question(qid=f"q{i}"), 3)
        sds.append(belief.sd("c0"))
    assert sds[-1] < sds[0]
    assert all(sds[i + 1] <= sds[i] + 1e-9 for i in range(len(sds) - 1))


def test_update_is_confined_to_the_targeted_competency():
    belief = GridBelief(make_rubric(ids=("c0", "c1")))
    before = belief.mean("c1")
    for _ in range(5):
        belief.update(make_question(cid="c0"), 5)
    assert belief.mean("c1") == pytest.approx(before)
    assert belief.n_observations["c1"] == 0


def test_unknown_competency_is_ignored_rather_than_raising():
    """A drifting follow-up should cost one observation, not the run."""
    belief = GridBelief(make_rubric())
    belief.update(make_question(cid="not.in.rubric"), 4)
    assert belief.n_observations == {"c0": 0}


def test_snapshot_covers_every_competency():
    belief = GridBelief(make_rubric(ids=("c0", "c1", "c2")))
    snap = belief.snapshot()
    assert set(snap.means) == set(snap.sds) == set(snap.entropies) == {"c0", "c1", "c2"}


# ------------------------------------------------- simulation ground truth


@pytest.mark.slow
def test_posterior_mean_converges_to_the_truth():
    """Draw a true ability, simulate 50 graded responses from it, and check the
    posterior lands on it within grid resolution."""
    rng = random.Random(20260805)
    errors = []
    for _trial in range(60):
        theta_true = rng.uniform(-1.8, 1.8)
        belief = GridBelief(make_rubric())
        for i in range(50):
            question = make_question(qid=f"q{i}", a=1.6)
            score = sample_score(theta_true, question.grm.a, question.grm.b, rng)
            belief.update(question, score)
        errors.append(abs(belief.mean("c0") - theta_true))

    assert np.median(errors) < 0.25, f"median |error| = {np.median(errors):.3f}"
    assert np.mean(errors) < 0.35


@pytest.mark.slow
def test_posterior_sd_shrinks_monotonically_in_expectation():
    rng = random.Random(7)
    curves = []
    for _ in range(40):
        theta_true = rng.uniform(-1.5, 1.5)
        belief = GridBelief(make_rubric())
        curve = [belief.sd("c0")]
        for i in range(25):
            question = make_question(qid=f"q{i}")
            belief.update(question, sample_score(theta_true, question.grm.a, question.grm.b, rng))
            curve.append(belief.sd("c0"))
        curves.append(curve)

    mean_curve = np.mean(curves, axis=0)
    assert np.all(np.diff(mean_curve) <= 1e-9), "expected SD is not monotonically shrinking"
    assert mean_curve[-1] < mean_curve[0] * 0.5


@pytest.mark.slow
@pytest.mark.gate
def test_eighty_percent_interval_has_eighty_percent_coverage():
    """Calibration, checked directly over 500 synthetic runs.

    The prior must match the distribution the truth is drawn from, or this
    measures prior misspecification rather than the inference machinery. Both
    are N(0, 1) here deliberately; the persona population's real spread is
    close to that by construction.
    """
    rng = random.Random(20260805)
    nprng = np.random.default_rng(20260805)
    contained = 0
    n_runs = 500

    for _ in range(n_runs):
        theta_true = float(nprng.normal(0.0, 1.0))
        theta_true = float(np.clip(theta_true, -2.9, 2.9))
        belief = GridBelief(make_rubric(prior_mean=0.0, prior_var=1.0))
        for i in range(12):
            question = make_question(qid=f"q{i}", a=1.6)
            belief.update(question, sample_score(theta_true, question.grm.a, question.grm.b, rng))
        lo, hi = belief.credible_interval("c0", 0.8)
        contained += lo <= theta_true <= hi

    coverage = contained / n_runs
    assert 0.78 <= coverage <= 0.82, f"80% interval covered {coverage:.1%} (want 78-82%)"


@pytest.mark.slow
def test_fifty_percent_interval_has_fifty_percent_coverage():
    """A second mass level, because a single one can be right by accident."""
    rng = random.Random(99)
    nprng = np.random.default_rng(99)
    contained = 0
    n_runs = 400
    for _ in range(n_runs):
        theta_true = float(np.clip(nprng.normal(0.0, 1.0), -2.9, 2.9))
        belief = GridBelief(make_rubric())
        for i in range(12):
            question = make_question(qid=f"q{i}")
            belief.update(question, sample_score(theta_true, question.grm.a, question.grm.b, rng))
        lo, hi = belief.credible_interval("c0", 0.5)
        contained += lo <= theta_true <= hi

    coverage = contained / n_runs
    assert 0.45 <= coverage <= 0.55, f"50% interval covered {coverage:.1%}"


def test_set_pmf_round_trips():
    belief = GridBelief(make_rubric())
    target = normalise(gaussian_log_prior(1.0, 0.25))
    belief.set_pmf("c0", target)
    assert belief.mean("c0") == pytest.approx(1.0, abs=0.02)
    assert np.allclose(belief.pmf("c0"), target)
