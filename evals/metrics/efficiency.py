"""Efficiency: how much does the interview cost to get there?

The centrepiece figure lives here — recovery as a function of question count,
one curve per arm. Every point on it is computed from a persisted belief
snapshot, never by re-running an interview at a shorter budget. That is not an
optimisation: re-running would change the interview, because the policy's
choices depend on the budget it thinks it has, and the curve would then be
comparing different experiments at each x.

Two variants of every efficiency number, because they can disagree:

* **per question** — the headline. Fewer questions is the claim.
* **per second** — the honest counterpart. An arm that asks three long
  questions instead of five short ones has not obviously won, and a candidate's
  time is what an interview actually spends.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import stats

from evals.metrics.loader import PersonaRuns, RunView, flatten

#: Minimum paired observations before a rank correlation is reported.
#: Below this the estimate is noise with a number attached.
MIN_PAIRS_FOR_RHO = 5


def questions_to_confidence(
    units: Sequence[PersonaRuns], arm: str, tau: float
) -> tuple[float, float]:
    """``(mean questions among those that got there, fraction that did)``.

    Censored runs are excluded from the mean and reported separately rather
    than folded in at the budget. Substituting the budget for a censored run
    would drag every arm toward the same number and hide exactly the
    difference the metric exists to show.
    """
    reached, total = [], 0
    for run in flatten(list(units), arm):
        total += 1
        n = run.questions_to_confidence(tau)
        if n is not None:
            reached.append(n)
    if total == 0:
        return float("nan"), float("nan")
    return (float(np.mean(reached)) if reached else float("nan"), len(reached) / total)


def seconds_to_confidence(units: Sequence[PersonaRuns], arm: str, tau: float) -> float:
    """The time-normalised variant."""
    totals = []
    for run in flatten(list(units), arm):
        n = run.questions_to_confidence(tau)
        if n is None:
            continue
        totals.append(sum(t.elapsed_seconds for t in run.turns[:n]))
    return float(np.mean(totals)) if totals else float("nan")


def resolved_fraction(
    units: Sequence[PersonaRuns], arm: str, tau: float, budget: int | None = None
) -> float:
    """Share of competencies under tau at ``budget`` questions."""
    resolved = total = 0
    for run in flatten(list(units), arm):
        got, n = run.resolved_count(tau, budget=budget)
        resolved += got
        total += n
    return resolved / total if total else float("nan")


def mean_posterior_sd(
    units: Sequence[PersonaRuns], arm: str, budget: int | None = None
) -> float:
    values = []
    for run in flatten(list(units), arm):
        snapshot = run.belief_at(budget) if budget else run.final_belief()
        if snapshot is not None:
            values.extend(snapshot.sds.values())
    return float(np.mean(values)) if values else float("nan")


def accuracy_vs_budget(
    units: Sequence[PersonaRuns], arm: str, budgets: Sequence[int] = tuple(range(1, 16))
) -> dict[int, float]:
    """The centrepiece: Spearman rho as a function of question count.

    Computed entirely from stored snapshots. The Phase 4 curve-integrity test
    replays one run turn by turn and diffs its curve against this to prove it.
    """
    runs = flatten(list(units), arm)
    curve: dict[int, float] = {}
    for budget in budgets:
        truth: list[float] = []
        estimate: list[float] = []
        for run in runs:
            if len(run.turns) < budget:
                continue
            # Over every competency in the rubric, probed or not -- the
            # estimate for an unprobed one is its prior, which is genuinely
            # what the report would say about it.
            #
            # Restricting to probed competencies made the curve *fall* with
            # more questions (eig: 0.83 at budget 4, 0.71 at budget 12), which
            # looks like the interview getting worse and is actually a
            # composition effect: later questions add newly-probed
            # competencies carrying a single observation each, diluting a pool
            # that previously held only well-measured ones. A curve whose x
            # axis changes what it is averaging over is not a curve about
            # budget.
            t, e = run.truth_and_estimate(budget=budget, probed_only=False)
            truth.extend(t)
            estimate.extend(e)
        if len(truth) >= MIN_PAIRS_FOR_RHO and np.std(estimate) > 1e-12:
            rho, _p = stats.spearmanr(truth, estimate)
            curve[budget] = float(rho) if np.isfinite(rho) else float("nan")
        else:
            curve[budget] = float("nan")
    return curve


def stop_reason_distribution(units: Sequence[PersonaRuns], arm: str) -> dict[str, float]:
    """Which condition ended each interview.

    A reported result in its own right: an arm that always stops on budget is
    not converging, whatever its accuracy looks like.
    """
    counts: dict[str, int] = {}
    runs = flatten(list(units), arm)
    for run in runs:
        reason = run.run.stop_reason.value if run.run.stop_reason else "unknown"
        counts[reason] = counts.get(reason, 0) + 1
    total = max(1, len(runs))
    return {k: v / total for k, v in sorted(counts.items())}


def mean_questions(units: Sequence[PersonaRuns], arm: str) -> float:
    runs = flatten(list(units), arm)
    return float(np.mean([r.n_questions for r in runs])) if runs else float("nan")


def followup_rate(units: Sequence[PersonaRuns], arm: str) -> float:
    """Share of turns that were generated follow-ups rather than bank items."""
    total = followups = 0
    for run in flatten(list(units), arm):
        for turn in run.turns:
            total += 1
            followups += "followup" in turn.question_id
    return followups / total if total else 0.0


def replay_curve(run: RunView, budgets: Sequence[int]) -> dict[int, float]:
    """Recompute one run's contribution by walking its turns.

    Deliberately a second implementation. The curve-integrity test diffs this
    against :func:`accuracy_vs_budget`; if the stored snapshots had drifted
    from what the interview actually believed, the two would disagree.
    """
    curve: dict[int, float] = {}
    for budget in budgets:
        if len(run.turns) < budget:
            curve[budget] = float("nan")
            continue
        snapshot = run.turns[budget - 1].belief_after
        known = set(run.persona.theta_star)
        truth = [run.persona.ability(c) for c in snapshot.means if c in known]
        estimate = [v for c, v in snapshot.means.items() if c in known]
        # Same inclusion threshold as accuracy_vs_budget. The two must differ in
        # *computation* to be a real cross-check, and agree on *which* pairs
        # they compute over — otherwise the diff reports a disagreement that is
        # only a difference of opinion about what counts as enough data.
        if len(truth) >= MIN_PAIRS_FOR_RHO and np.std(estimate) > 1e-12:
            rho, _p = stats.spearmanr(truth, estimate)
            curve[budget] = float(rho)
        else:
            curve[budget] = float("nan")
    return curve
