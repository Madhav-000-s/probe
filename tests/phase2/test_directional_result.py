"""The first directional result.

No confidence intervals yet, and deliberately so: the point of this phase gate
is to confirm the effect exists and points the right way *before* spending on
the full sweep. If adaptive selection were not beating a fixed script on ten
personas, scaling to eight hundred runs would only buy a more precise estimate
of nothing.

The number this produces is logged in results-log.md. If it had come out the
other way it would have been logged too.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from probe.config import Budgets, ExperimentConfig
from probe.policy.registry import ARMS
from probe.runtime.session import InterviewSpec, build_interview
from probe.runtime.tracing import TracedClient, TraceStore

BUDGET = 12
SEED = 20260805


def run_sweep(arm, personas, jds, bank, sim, store, budget=BUDGET):
    """One interview per persona under one arm. Everything except the arm is
    held identical — same bank, same grader, same stop rule, same seed."""
    config = ExperimentConfig(
        tau=0.55,
        budgets=Budgets(
            max_questions=budget, max_tokens=10**9, max_wallclock_seconds=1e9
        ),
        seed_set=[SEED],
        bank_version=bank.version,
        population_version="v1-phase2",
    )
    out = []
    for persona in personas:
        traced = TracedClient(sim, store=store)
        spec = InterviewSpec(persona=persona, jd=jds[persona.jd_id], arm=arm, seed=SEED)
        loop = build_interview(spec, bank=bank, config=config, client=traced, store=store)
        result = loop.run(resume=False)
        out.append((persona, result, loop))
    return out


def resolved_fraction(loop, tau=0.55):
    flags = loop.belief.resolved(tau)
    return sum(flags.values()) / max(1, len(flags))


def mean_required_sd(loop):
    return float(np.mean([loop.belief.sd(c.id) for c in loop.rubric.required]))


def recovery_rho(runs):
    """Spearman between posterior mean and true ability, over every
    (persona, required competency) pair that was actually probed."""
    truth, estimate = [], []
    for persona, result, loop in runs:
        probed = {t.competency_id for t in result.transcript.turns}
        for comp in loop.rubric.required:
            if comp.id in probed:
                truth.append(persona.ability(comp.id))
                estimate.append(loop.belief.mean(comp.id))
    if len(truth) < 10:
        return float("nan"), len(truth)
    rho, _p = stats.spearmanr(truth, estimate)
    return float(rho), len(truth)


@pytest.fixture(scope="module")
def sweeps(personas, jds, starter, request):
    """Run every arm once, shared across the assertions below."""
    from probe.runtime.llm import get_client

    sim = get_client("sim", seed=SEED)
    store = TraceStore(":memory:")
    request.addfinalizer(store.close)
    return {arm: run_sweep(arm, personas, jds, starter, sim, store) for arm in ARMS}


@pytest.mark.slow
@pytest.mark.gate
def test_eig_leaves_less_uncertainty_than_the_fixed_script(sweeps):
    """The Phase 2 directional gate.

    Measured as mean posterior SD over required competencies at a fixed
    question budget — the continuous form of the accuracy-vs-budget curve that
    is the project's centrepiece figure. Deliberately *not* questions-to-
    confidence: tau is still a placeholder at this phase and no arm reaches it
    within budget, so that metric is fully censored and cannot discriminate.
    Phase 3 sets tau empirically and it becomes measurable then.
    """
    fixed = np.mean([mean_required_sd(loop) for _p, _r, loop in sweeps["fixed"]])
    eig = np.mean([mean_required_sd(loop) for _p, _r, loop in sweeps["eig"]])

    assert eig < fixed, f"eig mean SD {eig:.3f} vs fixed {fixed:.3f}"


@pytest.mark.slow
def test_greedy_eig_loses_on_the_threshold_count(sweeps):
    """A negative result, pinned so it cannot quietly disappear.

    The eig arm leaves *less total uncertainty* than the fixed script and
    still resolves *fewer competencies below tau*. Both are true and the
    tension is real: entropy-greedy selection maximises nats, while the
    confidence stop rule counts threshold crossings, and those are not the
    same objective.

    The mechanism is arithmetic. Against tau = 0.55 with a = 1.9 items:

      - a resume-evidenced competency starts at SD 0.60 and crosses after
        **one** question;
      - a resume-silent one starts at SD 1.15 and needs **three**.

    So a breadth-first script buys roughly three threshold crossings for every
    one a widest-first policy buys, even though the widest-first policy
    extracts more information overall. This is exactly the "eig lost, what
    broke?" case the plan says to report in the main text rather than bury,
    and it is the reason tau is set empirically in Phase 3 instead of guessed.

    Asserted rather than merely noted, so that if a later change reverses it
    the suite says so instead of letting the README go stale.
    """
    fixed_resolved = np.mean([resolved_fraction(loop) for _p, _r, loop in sweeps["fixed"]])
    eig_resolved = np.mean([resolved_fraction(loop) for _p, _r, loop in sweeps["eig"]])
    fixed_sd = np.mean([mean_required_sd(loop) for _p, _r, loop in sweeps["fixed"]])
    eig_sd = np.mean([mean_required_sd(loop) for _p, _r, loop in sweeps["eig"]])

    assert eig_sd < fixed_sd, "the premise of this finding no longer holds"
    assert eig_resolved < fixed_resolved, (
        "the threshold-count inversion has reversed — good news, but the "
        "results-log entry and the README's 'where eig loses' section now "
        "describe something that is no longer true"
    )


@pytest.mark.slow
def test_every_arm_recovers_ability_above_chance(sweeps):
    """A policy that resolved fast but recovered nothing would be worse than
    useless, so efficiency is only meaningful alongside accuracy."""
    for arm, runs in sweeps.items():
        rho, n = recovery_rho(runs)
        assert n >= 20, f"{arm}: only {n} probed competencies"
        assert rho > 0.2, f"{arm}: recovery rho={rho:.3f} over n={n}"


@pytest.mark.slow
def test_the_heuristic_arm_is_a_real_competitor(sweeps):
    """If the heuristic arm were weak, beating it would prove nothing.

    Compared on the same continuous metric as the gate. It has to beat the
    fixed script it is a smarter version of, and it should land between the
    script and the belief-driven arm — which is precisely what makes it a
    credible stand-in for "what everyone else builds".
    """
    fixed = np.mean([mean_required_sd(loop) for _p, _r, loop in sweeps["fixed"]])
    heuristic = np.mean([mean_required_sd(loop) for _p, _r, loop in sweeps["heuristic"]])
    eig = np.mean([mean_required_sd(loop) for _p, _r, loop in sweeps["eig"]])

    assert heuristic < fixed, (
        f"the heuristic arm ({heuristic:.3f}) did not beat the fixed script "
        f"({fixed:.3f}); a strawman competitor would make the headline result "
        f"meaningless"
    )
    assert eig <= heuristic, (
        f"eig ({eig:.3f}) did not beat the heuristic arm ({heuristic:.3f}) — "
        f"report this rather than tuning until it does"
    )


@pytest.mark.slow
def test_all_arms_spent_the_same_budget(sweeps):
    """An arm that quietly asked fewer questions would win on efficiency for
    the wrong reason."""
    for arm, runs in sweeps.items():
        lengths = {len(result.transcript) for _p, result, _loop in runs}
        assert lengths == {BUDGET}, f"{arm} ran interviews of length {sorted(lengths)}"


@pytest.mark.slow
def test_directional_summary(sweeps, capsys):
    """Prints the table that goes into results-log.md. Not an assertion —
    numbers are computed and recorded, never typed by hand."""
    rows = []
    for arm in ARMS:
        runs = sweeps[arm]
        rho, n = recovery_rho(runs)
        rows.append(
            (
                arm,
                np.mean([resolved_fraction(loop) for _p, _r, loop in runs]),
                np.mean([mean_required_sd(loop) for _p, _r, loop in runs]),
                rho,
                n,
            )
        )

    with capsys.disabled():
        print(f"\n  arm         resolved   mean SD   recovery rho   n (budget {BUDGET})")
        for arm, resolved, sd, rho, n in rows:
            print(f"  {arm:<11} {resolved:7.1%} {sd:9.3f} {rho:14.3f} {n:5d}")

    assert all(np.isfinite(r[3]) for r in rows)
