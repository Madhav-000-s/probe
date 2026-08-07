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
from probe.runtime.session import InterviewSpec, build_interview
from probe.runtime.tracing import TracedClient, TraceStore

#: The three arms that existed when this directional result was recorded.
#: `eig+corr` lands in Phase 3 and is compared with confidence intervals in
#: Phase 4; adding it here would silently change a logged number.
ARMS = ("fixed", "heuristic", "eig")

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
        spec = InterviewSpec(
            persona=persona,
            jd=jds[persona.jd_id],
            arm=arm,
            seed=SEED,
            # Follow-ups are a Phase 3 feature and a Phase 6 ablation. Pinning
            # them off here keeps this phase's recorded finding reproducible
            # and keeps the comparison about question *selection*, which is the
            # only thing that differs between arms.
            followups_enabled=False,
        )
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
def test_eig_resolves_more_competencies_than_the_fixed_script(sweeps):
    """The Phase 2 directional gate, as re-measured under the frozen constants.

    **This result was retracted and re-run once.** The first version was
    measured against a rubric of fourteen competencies on a twelve-question
    budget, which cannot even ask one question per competency: several were
    never probed, sat at their prior interval for the whole interview, and
    counted against every arm identically. Under that configuration the eig arm
    resolved *fewer* competencies than the fixed script, and the log recorded
    that as a genuine tension between entropy-greedy selection and a threshold
    stop rule.

    Phase 3 established the rubric size was the mis-specification, set it to
    six, and set tau empirically. Re-run under the corrected design the
    ordering flips and eig wins. The mechanism named in the old entry was real
    arithmetic — a resume-evidenced competency crosses tau in one question, a
    resume-silent one in three — but it only dominated because the budget was
    too thin to reach most of the rubric at all.

    The retraction is in results-log.md under 2026-08-07. This is what the
    plan's "frozen means frozen: re-run or retract" rule is for, and it earned
    its place the first time it was invoked.
    """
    fixed = np.mean([resolved_fraction(loop) for _p, _r, loop in sweeps["fixed"]])
    eig = np.mean([resolved_fraction(loop) for _p, _r, loop in sweeps["eig"]])

    assert eig > fixed, f"eig resolved {eig:.1%} vs fixed {fixed:.1%} at budget {BUDGET}"


@pytest.mark.slow
def test_total_uncertainty_is_a_wash_between_arms(sweeps):
    """The honest counterpart, pinned so it does not get quietly overstated.

    On *mean posterior SD* the three arms are indistinguishable at this sample
    size — roughly 0.58 for all of them, differences of a few thousandths on
    ten personas with no confidence interval. The eig arm's advantage is in
    *where* it spends questions, not in extracting more total information, and
    the write-up should not claim otherwise.
    """
    values = {
        arm: np.mean([mean_required_sd(loop) for _p, _r, loop in runs])
        for arm, runs in sweeps.items()
    }
    spread = max(values.values()) - min(values.values())
    assert spread < 0.05, (
        f"arms have separated on mean posterior SD ({values}); this test "
        f"asserted they were a wash, so the claim needs revisiting"
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
    fixed = np.mean([resolved_fraction(loop) for _p, _r, loop in sweeps["fixed"]])
    heuristic = np.mean([resolved_fraction(loop) for _p, _r, loop in sweeps["heuristic"]])
    eig = np.mean([resolved_fraction(loop) for _p, _r, loop in sweeps["eig"]])

    assert heuristic > fixed, (
        f"the heuristic arm ({heuristic:.1%}) did not beat the fixed script "
        f"({fixed:.1%}); a strawman competitor would make the headline result "
        f"meaningless"
    )
    assert eig >= heuristic, (
        f"eig ({eig:.1%}) did not beat the heuristic arm ({heuristic:.1%}) — "
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
