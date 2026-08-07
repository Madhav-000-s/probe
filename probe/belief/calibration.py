"""Fitting item parameters from data.

Marginal maximum likelihood by EM, which is the standard way to calibrate an
item bank and — crucially — never touches ``theta_star``. Using the true
abilities to fit the items would produce a bank calibrated against ground
truth, which no real calibration can do and which would quietly inflate every
recovery number downstream. The E-step integrates ability out against a
population prior instead.

    E-step: for each respondent, posterior over theta on the grid given the
            current item parameters.
    M-step: for each item, maximise the expected complete-data log-likelihood
            over (a, b) with ability treated as the posterior weights.

Parameters are optimised unconstrained and mapped back:

    a    = exp(la)                       (positive)
    b_1  = beta
    b_k  = b_{k-1} + exp(delta_k)        (strictly increasing)

which means the optimiser can never propose thresholds that violate the model,
rather than having to be told not to.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import optimize

from probe.belief.grid import THETA_GRID, gaussian_log_prior, normalise
from probe.belief.grm import category_probs
from probe.models import GRMParams, QuestionBank

#: Discrimination below which an item is uninformative and gets quarantined.
MIN_DISCRIMINATION = 0.30
#: Above this, the fit has almost certainly run away on thin data.
MAX_DISCRIMINATION = 6.0
#: Responses an item needs before its fit is trusted at all.
MIN_RESPONSES = 20
#: EM stopping rule.
MAX_EM_ITERATIONS = 60
TOLERANCE = 1e-4


@dataclass
class ItemFit:
    question_id: str
    a: float
    b: list[float]
    n_responses: int
    converged: bool
    log_likelihood: float
    quarantined: bool = False
    reason: str | None = None


@dataclass
class CalibrationReport:
    fits: dict[str, ItemFit] = field(default_factory=dict)
    n_em_iterations: int = 0
    converged: bool = False

    @property
    def quarantined(self) -> list[str]:
        return sorted(qid for qid, fit in self.fits.items() if fit.quarantined)

    @property
    def quarantine_fraction(self) -> float:
        return len(self.quarantined) / max(1, len(self.fits))

    def summary(self) -> dict[str, object]:
        live = [f for f in self.fits.values() if not f.quarantined]
        return {
            "n_items": len(self.fits),
            "n_quarantined": len(self.quarantined),
            "quarantine_fraction": round(self.quarantine_fraction, 4),
            "em_iterations": self.n_em_iterations,
            "converged": self.converged,
            "mean_a": round(float(np.mean([f.a for f in live])), 4) if live else None,
            "median_a": round(float(np.median([f.a for f in live])), 4) if live else None,
            "mean_responses": round(
                float(np.mean([f.n_responses for f in self.fits.values()])), 1
            )
            if self.fits
            else 0.0,
        }


def _pack(a: float, b: list[float]) -> np.ndarray:
    """(a, b) -> unconstrained vector."""
    deltas = np.diff(np.asarray(b, dtype=float))
    return np.concatenate([[np.log(max(a, 1e-6)), b[0]], np.log(np.maximum(deltas, 1e-6))])


def _unpack(x: np.ndarray) -> tuple[float, list[float]]:
    """Unconstrained vector -> (a, b), monotone by construction."""
    a = float(np.exp(np.clip(x[0], -8, 3)))
    b = [float(x[1])]
    for delta in x[2:]:
        b.append(b[-1] + float(np.exp(np.clip(delta, -8, 3))))
    return a, b


def _item_neg_loglik(x: np.ndarray, weights: np.ndarray, counts: np.ndarray) -> float:
    """Expected complete-data negative log-likelihood for one item.

    ``weights`` is ``(n_respondents, n_grid)`` — the E-step posteriors — and
    ``counts`` is ``(n_respondents, 5)`` one-hot over observed categories.
    """
    a, b = _unpack(x)
    probs = np.clip(category_probs(THETA_GRID, a, b), 1e-12, 1.0)  # (n_grid, 5)
    # log P(observed category | theta) for each respondent at each grid point
    per_grid = counts @ np.log(probs).T  # (n_respondents, n_grid)
    return -float((weights * per_grid).sum())


def calibrate_competency(
    responses: dict[str, dict[str, int]],
    initial: dict[str, GRMParams],
    prior_mean: float = 0.0,
    prior_var: float = 1.0,
    max_iterations: int = MAX_EM_ITERATIONS,
) -> tuple[dict[str, ItemFit], int, bool]:
    """Fit every item of one competency jointly.

    ``responses`` maps respondent id -> {question_id: score}. Items are fitted
    together because they share the respondents whose abilities are being
    integrated out; fitting them one at a time would re-estimate ability from a
    single item and recover almost nothing.
    """
    respondents = sorted(responses)
    item_ids = sorted(initial)
    if not respondents or not item_ids:
        return {}, 0, True

    # counts[i][r] is a one-hot over the 5 categories, or None if unanswered.
    observed: dict[str, np.ndarray] = {}
    answered: dict[str, np.ndarray] = {}
    for qid in item_ids:
        rows = np.zeros((len(respondents), 5))
        mask = np.zeros(len(respondents), dtype=bool)
        for r_idx, rid in enumerate(respondents):
            score = responses[rid].get(qid)
            if score is not None:
                rows[r_idx, int(score) - 1] = 1.0
                mask[r_idx] = True
        observed[qid] = rows
        answered[qid] = mask

    params = {qid: (initial[qid].a, list(initial[qid].b)) for qid in item_ids}
    log_prior = gaussian_log_prior(prior_mean, prior_var)
    previous = None
    iterations = 0
    converged = False

    for iterations in range(1, max_iterations + 1):  # noqa: B007 - value used after loop
        # ---- E-step: posterior over theta for every respondent
        log_post = np.tile(log_prior, (len(respondents), 1))
        for qid in item_ids:
            a, b = params[qid]
            log_probs = np.log(np.clip(category_probs(THETA_GRID, a, b), 1e-12, 1.0))
            contribution = observed[qid] @ log_probs.T  # (n_respondents, n_grid)
            log_post += np.where(answered[qid][:, None], contribution, 0.0)
        weights = np.vstack([normalise(row) for row in log_post])

        # ---- M-step: one independent optimisation per item
        total_ll = 0.0
        for qid in item_ids:
            mask = answered[qid]
            if mask.sum() < 2:
                continue
            a, b = params[qid]
            # L-BFGS-B rather than Nelder-Mead: the objective is smooth in the
            # unconstrained parameterisation, so a quasi-Newton method is the
            # right tool and a five-dimensional simplex search is not.
            #
            # It was swapped in while chasing what looked like an upward bias
            # on discrimination, and changed the estimates not at all — the
            # apparent bias was sampling error at 800 respondents, which flips
            # sign with the seed and shrinks as n grows. Recorded because the
            # wrong diagnosis is worth remembering: the M-step was never the
            # problem.
            result = optimize.minimize(
                _item_neg_loglik,
                _pack(a, b),
                args=(weights[mask], observed[qid][mask]),
                method="L-BFGS-B",
                options={"maxiter": 400, "ftol": 1e-9, "gtol": 1e-7},
            )
            params[qid] = _unpack(result.x)
            total_ll -= float(result.fun)

        # Relative tolerance. The log-likelihood scales with the number of
        # respondents, so an absolute threshold silently becomes stricter as
        # the sample grows — at 4000 respondents an absolute 1e-4 never trips
        # and every fit is reported as unconverged despite being fine.
        scale = max(1.0, abs(total_ll))
        if previous is not None and abs(total_ll - previous) < TOLERANCE * scale:
            converged = True
            break
        previous = total_ll

    fits: dict[str, ItemFit] = {}
    for qid in item_ids:
        a, b = params[qid]
        n = int(answered[qid].sum())
        fit = ItemFit(
            question_id=qid,
            a=round(a, 4),
            b=[round(v, 4) for v in b],
            n_responses=n,
            converged=converged,
            log_likelihood=0.0,
        )
        _apply_quarantine(fit)
        fits[qid] = fit
    return fits, iterations, converged


def _apply_quarantine(fit: ItemFit) -> None:
    """Flag items whose fitted parameters make them unusable.

    Quarantined items stay in the bank — deleting them would hide the fact
    that they were ever authored — but eval never draws them, and the
    quarantined fraction is reported alongside every results table.
    """
    if fit.n_responses < MIN_RESPONSES:
        fit.quarantined, fit.reason = True, f"only {fit.n_responses} responses"
    elif fit.a < MIN_DISCRIMINATION:
        fit.quarantined, fit.reason = True, f"a={fit.a:.3f} below {MIN_DISCRIMINATION}"
    elif fit.a > MAX_DISCRIMINATION:
        fit.quarantined, fit.reason = True, f"a={fit.a:.3f} above {MAX_DISCRIMINATION}"
    elif any(fit.b[i] >= fit.b[i + 1] for i in range(len(fit.b) - 1)):
        fit.quarantined, fit.reason = True, "non-monotone thresholds"
    elif abs(fit.b[0]) > 6.0 or abs(fit.b[-1]) > 6.0:
        fit.quarantined, fit.reason = True, "thresholds outside the measurable range"


def calibrate_bank(
    bank: QuestionBank,
    responses: dict[str, dict[str, int]],
    new_version: str,
) -> tuple[QuestionBank, CalibrationReport]:
    """Fit the whole bank, competency by competency, and emit a new version.

    Competencies are independent here because ability is modelled
    per-competency. The cross-competency structure lives in the correlation
    matrix used by the ``eig+corr`` arm, not in the item parameters.
    """
    report = CalibrationReport()
    by_competency: dict[str, list] = {}
    for question in bank.questions:
        by_competency.setdefault(question.competency_id, []).append(question)

    total_iterations = 0
    all_converged = True
    for _cid, questions in sorted(by_competency.items()):
        initial = {q.id: q.grm for q in questions}
        subset = {
            rid: {qid: score for qid, score in row.items() if qid in initial}
            for rid, row in responses.items()
        }
        subset = {rid: row for rid, row in subset.items() if row}
        fits, iterations, converged = calibrate_competency(subset, initial)
        report.fits.update(fits)
        total_iterations = max(total_iterations, iterations)
        all_converged &= converged

    report.n_em_iterations = total_iterations
    report.converged = all_converged

    calibrated = bank.model_copy(deep=True)
    calibrated.version = new_version
    for question in calibrated.questions:
        fit = report.fits.get(question.id)
        if fit is None:
            question.grm.quarantined = True
            question.grm.quarantine_reason = "never administered during calibration"
            continue
        if fit.quarantined:
            # Keep the authoring defaults; a bad fit is not better than no fit.
            question.grm.quarantined = True
            question.grm.quarantine_reason = fit.reason
            continue
        question.grm = GRMParams(a=fit.a, b=fit.b, calibrated=True)
    return calibrated, report
