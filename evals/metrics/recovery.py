"""Recovery: does the interview find out what the candidate actually knows?

Three things, because a single correlation hides too much:

* **rho** — rank agreement between posterior mean and true ability. Says the
  ordering is right.
* **ECE** — calibration of the credible intervals. Says the *uncertainty* is
  right, which is what makes a report actionable rather than a guess with a
  number attached.
* **decision precision/recall** — whether the hire/no-hire call the report
  supports is the correct one. The only metric a hiring manager would
  recognise.

An arm can win the first and lose the others. A policy that concentrates on a
few competencies gets sharp estimates where it looked and none where it did
not; rho over probed competencies flatters it, and the decision metric does
not.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import stats

from evals.metrics.efficiency import MIN_PAIRS_FOR_RHO
from evals.metrics.loader import PersonaRuns, RunView, flatten

#: Ability at or above which a competency counts as "meets the bar". Expressed
#: on the theta scale, where 0 is the population mean.
HIRE_THRESHOLD = 0.0


def _pairs(runs: Sequence[RunView], budget: int | None, probed_only: bool = True):
    truth: list[float] = []
    estimate: list[float] = []
    for run in runs:
        t, e = run.truth_and_estimate(budget=budget, probed_only=probed_only)
        truth.extend(t)
        estimate.extend(e)
    return np.asarray(truth), np.asarray(estimate)


def spearman_rho(
    units: Sequence[PersonaRuns], arm: str, budget: int | None = None, probed_only: bool = True
) -> float:
    truth, estimate = _pairs(flatten(list(units), arm), budget, probed_only)
    if truth.size < MIN_PAIRS_FOR_RHO or np.std(estimate) < 1e-12:
        return float("nan")
    rho, _p = stats.spearmanr(truth, estimate)
    return float(rho)


def mean_absolute_error(
    units: Sequence[PersonaRuns], arm: str, budget: int | None = None
) -> float:
    truth, estimate = _pairs(flatten(list(units), arm), budget)
    if truth.size == 0:
        return float("nan")
    return float(np.mean(np.abs(truth - estimate)))


def expected_calibration_error(
    units: Sequence[PersonaRuns],
    arm: str,
    masses: Sequence[float] = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95),
    budget: int | None = None,
) -> float:
    """Mean gap between a credible interval's nominal and actual coverage.

    Computed from the posterior SD via the Gaussian quantile rather than from
    the stored grid, because the snapshot persists mean and SD rather than the
    full pmf. That is an approximation and it is the reason ECE here is a
    summary rather than an exact figure; the Phase 2 coverage test checks the
    grid intervals directly and exactly.
    """
    runs = flatten(list(units), arm)
    if not runs:
        return float("nan")

    gaps = []
    for mass in masses:
        z = float(stats.norm.ppf(0.5 + mass / 2.0))
        hits = total = 0
        for run in runs:
            snapshot = run.belief_at(budget) if budget else run.final_belief()
            if snapshot is None:
                continue
            for cid, mean in snapshot.means.items():
                if cid not in run.persona.theta_star or cid not in run.probed():
                    continue
                sd = snapshot.sds[cid]
                truth = run.persona.ability(cid)
                hits += abs(truth - mean) <= z * sd
                total += 1
        if total:
            gaps.append(abs(hits / total - mass))
    return float(np.mean(gaps)) if gaps else float("nan")


def decision_precision_recall(
    units: Sequence[PersonaRuns], arm: str, budget: int | None = None
) -> tuple[float, float, float]:
    """Precision, recall and F1 of the "meets the bar" call per competency."""
    runs = flatten(list(units), arm)
    tp = fp = fn = 0
    for run in runs:
        truth, estimate = run.truth_and_estimate(budget=budget)
        for t, e in zip(truth, estimate, strict=True):
            predicted = e >= HIRE_THRESHOLD
            actual = t >= HIRE_THRESHOLD
            tp += predicted and actual
            fp += predicted and not actual
            fn += (not predicted) and actual

    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    if not np.isfinite(precision) or not np.isfinite(recall) or (precision + recall) == 0:
        return precision, recall, float("nan")
    return precision, recall, 2 * precision * recall / (precision + recall)


def coverage_at(
    units: Sequence[PersonaRuns], arm: str, mass: float = 0.8, budget: int | None = None
) -> float:
    """Observed coverage of the nominal ``mass`` interval. Should equal mass."""
    z = float(stats.norm.ppf(0.5 + mass / 2.0))
    hits = total = 0
    for run in flatten(list(units), arm):
        snapshot = run.belief_at(budget) if budget else run.final_belief()
        if snapshot is None:
            continue
        for cid, mean in snapshot.means.items():
            if cid not in run.persona.theta_star or cid not in run.probed():
                continue
            hits += abs(run.persona.ability(cid) - mean) <= z * snapshot.sds[cid]
            total += 1
    return hits / total if total else float("nan")


def per_competency_rho(units: Sequence[PersonaRuns], arm: str) -> dict[str, float]:
    """Where the arm does well and badly, competency by competency.

    Reported because the cases an arm *loses* are first-class results and the
    raw material for an honest write-up, not something to average away.
    """
    buckets: dict[str, tuple[list[float], list[float]]] = {}
    for run in flatten(list(units), arm):
        snapshot = run.final_belief()
        if snapshot is None:
            continue
        for cid, mean in snapshot.means.items():
            if cid not in run.persona.theta_star or cid not in run.probed():
                continue
            truth, estimate = buckets.setdefault(cid, ([], []))
            truth.append(run.persona.ability(cid))
            estimate.append(mean)

    out: dict[str, float] = {}
    for cid, (truth, estimate) in buckets.items():
        if len(truth) < 6 or np.std(estimate) < 1e-12:
            continue
        rho, _p = stats.spearmanr(truth, estimate)
        if np.isfinite(rho):
            out[cid] = float(rho)
    return dict(sorted(out.items(), key=lambda kv: kv[1]))
