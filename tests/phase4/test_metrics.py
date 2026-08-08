"""Metric correctness on synthetic traces with known answers.

Every metric ships with a can't-fool-yourself fixture: a case where the right
answer is computable by hand or known by construction, so a metric that is
subtly wrong fails here rather than quietly shifting a number in the results
table by 0.03.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from evals.gold import GoldItem, cohens_kappa, kappa_report, quadratic_weighted_kappa
from evals.metrics import efficiency, recovery
from evals.metrics.bootstrap import bootstrap_ci, paired_difference_ci
from evals.metrics.loader import PersonaRuns, RunView
from probe.models import BeliefSnapshot, Persona, RunRecord, StopReason, StyleProfile, Turn

NEUTRAL = StyleProfile(id="neutral", verbosity=1.0, hedging=0.1, assertiveness=0.1, l1_transfer=0.0)


def make_persona(pid: str, theta: dict[str, float]) -> Persona:
    return Persona(
        id=pid, theta_star=theta, style=NEUTRAL, behavior="honest", resume="r", jd_id="jd"
    )


def make_run(
    persona: Persona,
    arm: str,
    means: dict[str, float],
    sds: dict[str, float] | None = None,
    n_turns: int | None = None,
    seconds: float = 60.0,
    tokens: int = 100,
) -> RunView:
    """A trace built by hand, so the right answer is known."""
    sds = sds or {c: 0.5 for c in means}
    competencies = list(means)
    n_turns = n_turns or len(competencies)
    snapshot = BeliefSnapshot(
        means=dict(means), sds=dict(sds), entropies={c: 1.0 for c in means}
    )
    turns = [
        Turn(
            run_id=f"{arm}.{persona.id}",
            turn_idx=i,
            question_id=f"{competencies[i % len(competencies)]}::0::scenario",
            competency_id=competencies[i % len(competencies)],
            question_text="q",
            answer="a",
            grade=None,
            belief_after=snapshot,
            elapsed_seconds=seconds,
            tokens_used=tokens,
        )
        for i in range(n_turns)
    ]
    run = RunRecord(
        run_id=f"{arm}.{persona.id}",
        arm=arm,
        persona_id=persona.id,
        style_id="neutral",
        bank_version="v2",
        population_version="v2",
        code_commit="test",
        seed=1,
        stop_reason=StopReason.CONFIDENCE,
        n_turns=n_turns,
        completed=True,
    )
    return RunView(run=run, turns=turns, persona=persona)


# ------------------------------------------------------------ recovery rho


def test_rho_matches_a_hand_computable_case():
    """Estimates that are a strictly increasing function of the truth must give
    rho exactly 1."""
    theta = {"a.x": -1.0, "a.y": 0.0, "a.z": 1.0, "b.x": 2.0, "b.y": 0.5}
    persona = make_persona("p1", theta)
    units = [PersonaRuns("p1", [make_run(persona, "fixed", {k: v * 0.5 for k, v in theta.items()})])]
    assert recovery.spearman_rho(units, "fixed") == pytest.approx(1.0)


def test_rho_is_minus_one_when_the_ordering_is_inverted():
    theta = {"a.x": -1.0, "a.y": 0.0, "a.z": 1.0, "b.x": 2.0, "b.y": 0.5}
    persona = make_persona("p1", theta)
    units = [PersonaRuns("p1", [make_run(persona, "fixed", {k: -v for k, v in theta.items()})])]
    assert recovery.spearman_rho(units, "fixed") == pytest.approx(-1.0)


def test_rho_agrees_with_scipy_on_a_known_sample():
    theta = {"a.x": -1.3, "a.y": 0.2, "a.z": 0.9, "b.x": 1.8, "b.y": -0.4}
    estimates = {"a.x": -0.9, "a.y": 0.6, "a.z": 0.4, "b.x": 1.5, "b.y": -1.1}
    persona = make_persona("p1", theta)
    units = [PersonaRuns("p1", [make_run(persona, "fixed", estimates)])]

    expected, _p = stats.spearmanr(
        [theta[k] for k in theta], [estimates[k] for k in theta]
    )
    assert recovery.spearman_rho(units, "fixed") == pytest.approx(float(expected))


# ------------------------------------------------------------------- ECE


def test_ece_of_a_perfectly_calibrated_posterior_is_near_zero():
    """Truth drawn from exactly the posterior each run reports. A metric that
    returned a large number here would be measuring its own arithmetic."""
    rng = np.random.default_rng(11)
    units = []
    for i in range(220):
        means = {f"c.{j}": float(rng.normal(0, 1)) for j in range(4)}
        sds = dict.fromkeys(means, 0.6)
        theta = {c: float(rng.normal(m, sds[c])) for c, m in means.items()}
        persona = make_persona(f"p{i}", theta)
        units.append(PersonaRuns(persona.id, [make_run(persona, "fixed", means, sds)]))

    assert recovery.expected_calibration_error(units, "fixed") < 0.05
    assert recovery.coverage_at(units, "fixed", 0.8) == pytest.approx(0.8, abs=0.05)


def test_ece_detects_an_overconfident_posterior():
    """Intervals half as wide as they should be must show up as miscalibrated —
    this is the failure the real system exhibits, so the detector has to work."""
    rng = np.random.default_rng(12)
    units = []
    for i in range(220):
        means = {f"c.{j}": float(rng.normal(0, 1)) for j in range(4)}
        true_sd = 0.6
        reported = dict.fromkeys(means, true_sd / 2.0)
        theta = {c: float(rng.normal(m, true_sd)) for c, m in means.items()}
        persona = make_persona(f"p{i}", theta)
        units.append(PersonaRuns(persona.id, [make_run(persona, "fixed", means, reported)]))

    assert recovery.expected_calibration_error(units, "fixed") > 0.15
    assert recovery.coverage_at(units, "fixed", 0.8) < 0.7


# ------------------------------------------------------------- efficiency


def test_questions_to_confidence_is_the_first_crossing():
    persona = make_persona("p1", {"c.a": 0.0, "c.b": 0.0})
    run = make_run(persona, "eig", {"c.a": 0.0, "c.b": 0.0}, n_turns=4)
    # Widen the first two turns so the crossing happens exactly at turn 3.
    for i, sd in enumerate([0.9, 0.9, 0.4, 0.4]):
        run.turns[i].belief_after = BeliefSnapshot(
            means={"c.a": 0.0, "c.b": 0.0},
            sds={"c.a": sd, "c.b": sd},
            entropies={"c.a": 1.0, "c.b": 1.0},
        )
    assert run.questions_to_confidence(tau=0.5) == 3


def test_censored_runs_are_excluded_not_replaced_by_the_budget():
    """Substituting the budget for a run that never converged would drag every
    arm toward the same number and hide the difference the metric exists to
    show."""
    persona_a = make_persona("p1", {"c.a": 0.0})
    persona_b = make_persona("p2", {"c.a": 0.0})
    reached = make_run(persona_a, "eig", {"c.a": 0.0}, {"c.a": 0.2}, n_turns=2)
    censored = make_run(persona_b, "eig", {"c.a": 0.0}, {"c.a": 0.9}, n_turns=12)

    units = [PersonaRuns("p1", [reached]), PersonaRuns("p2", [censored])]
    mean_q, fraction = efficiency.questions_to_confidence(units, "eig", tau=0.5)
    assert mean_q == pytest.approx(1.0)
    assert fraction == pytest.approx(0.5)


def test_resolved_fraction_counts_competencies_not_runs():
    persona = make_persona("p1", {"c.a": 0.0, "c.b": 0.0, "c.c": 0.0, "c.d": 0.0})
    run = make_run(
        persona,
        "fixed",
        dict.fromkeys(["c.a", "c.b", "c.c", "c.d"], 0.0),
        {"c.a": 0.2, "c.b": 0.3, "c.c": 0.9, "c.d": 0.9},
    )
    assert efficiency.resolved_fraction([PersonaRuns("p1", [run])], "fixed", tau=0.5) == 0.5


def test_stop_reason_distribution_sums_to_one():
    persona = make_persona("p1", {"c.a": 0.0})
    units = [PersonaRuns("p1", [make_run(persona, "eig", {"c.a": 0.0})])]
    dist = efficiency.stop_reason_distribution(units, "eig")
    assert sum(dist.values()) == pytest.approx(1.0)


# -------------------------------------------------------------- bootstrap


def test_bootstrap_of_a_constant_metric_has_zero_width():
    """The plan's can't-fool-yourself check for the CI machinery itself."""
    units = list(range(30))
    interval = bootstrap_ci(units, lambda _sample: 0.42, resamples=200)
    assert interval.point == pytest.approx(0.42)
    assert interval.width == pytest.approx(0.0)
    assert not interval.excludes(0.42)


def test_bootstrap_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(3)
    units = [float(x) for x in rng.normal(0.5, 1.0, 80)]
    interval = bootstrap_ci(units, lambda s: float(np.mean(s)), resamples=800)
    assert interval.lo <= interval.point <= interval.hi
    assert interval.width > 0


def test_bootstrap_is_deterministic_under_a_fixed_seed():
    """`make eval` twice must produce byte-identical output, which requires
    this."""
    units = [float(x) for x in np.random.default_rng(5).normal(0, 1, 40)]
    a = bootstrap_ci(units, lambda s: float(np.mean(s)), resamples=300, seed=7)
    b = bootstrap_ci(units, lambda s: float(np.mean(s)), resamples=300, seed=7)
    assert a == b


def test_paired_difference_of_identical_statistics_is_zero():
    units = list(range(25))
    interval = paired_difference_ci(
        units, lambda s: float(np.mean(s)), lambda s: float(np.mean(s)), resamples=200
    )
    assert interval.point == pytest.approx(0.0)
    assert interval.width == pytest.approx(0.0)


def test_paired_difference_detects_a_real_gap():
    units = [float(x) for x in np.random.default_rng(9).normal(0, 1, 120)]
    interval = paired_difference_ci(
        units,
        lambda s: float(np.mean(s)) + 1.0,
        lambda s: float(np.mean(s)),
        resamples=500,
    )
    assert interval.point == pytest.approx(1.0)
    assert interval.excludes(0.0)


# ------------------------------------------------------------------ kappa


def test_kappa_of_perfect_agreement_is_one():
    scores = [1, 2, 3, 4, 5, 3, 2, 4, 1, 5]
    assert cohens_kappa(scores, scores) == pytest.approx(1.0)
    assert quadratic_weighted_kappa(scores, scores) == pytest.approx(1.0)


def test_kappa_of_chance_agreement_is_near_zero():
    rng = np.random.default_rng(4)
    a = [int(x) for x in rng.integers(1, 6, 4000)]
    b = [int(x) for x in rng.integers(1, 6, 4000)]
    assert abs(cohens_kappa(a, b)) < 0.06


def test_weighted_kappa_is_gentler_on_near_misses():
    """A 4-vs-5 disagreement is milder than 1-vs-5 on an ordinal rubric, and
    the weighted form has to say so."""
    truth = [3] * 50 + [4] * 50
    near = [3] * 50 + [5] * 50
    far = [3] * 50 + [1] * 50
    assert quadratic_weighted_kappa(truth, near) > quadratic_weighted_kappa(truth, far)


def test_kappa_report_shape():
    items = [
        GoldItem(
            answer_id=f"a{i}",
            question_id="q",
            competency_id="c",
            question_text="q",
            answer="a",
            anchors=["1", "2", "3", "4", "5"],
            reference_score=(i % 5) + 1,
            llm_score=(i % 5) + 1,
            llm_confidence=0.8,
        )
        for i in range(20)
    ]
    report = kappa_report(items)
    assert report["n"] == 20
    assert report["cohens_kappa"] == pytest.approx(1.0)
    assert report["exact_agreement"] == pytest.approx(1.0)
