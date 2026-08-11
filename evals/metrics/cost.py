"""Cost and latency.

The metric that can overturn the headline. If the adaptive arm reaches
confidence in fewer questions but spends more tokens doing it — because it
evaluates the whole bank at every turn, or because it generates follow-ups —
then "more efficient" is a claim about the candidate's time and not about the
bill. Both get reported.

Token counts come from the ``llm_calls`` table rather than from the in-loop
approximation, so changing the pricing constants re-costs history correctly
instead of only affecting future runs.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from evals.metrics.loader import PersonaRuns, flatten
from probe.runtime.tracing import TraceStore


def token_totals(traces: str | Path, run_ids: Sequence[str] | None = None) -> dict[str, float]:
    """Exact token and latency totals straight from the call log.

    Scoped to an explicit set of run ids rather than to an arm. Filtering by
    arm summed every sweep that arm ever appeared in — the main sweep plus the
    fairness and ablation sweeps — and then divided by the main sweep's run
    count, which inflated the eig arm's cost per interview roughly threefold.
    The run set the caller is reporting on is the run set the cost has to come
    from.
    """
    store = TraceStore(traces, read_only=True)
    try:
        sql = """
            SELECT
                count(*)                    AS n_calls,
                sum(c.prompt_tokens)        AS prompt_tokens,
                sum(c.completion_tokens)    AS completion_tokens,
                avg(c.latency_ms)           AS mean_latency_ms,
                sum(CASE WHEN NOT c.parsed_ok THEN 1 ELSE 0 END) AS parse_failures,
                sum(CASE WHEN c.repair_attempt > 0 THEN 1 ELSE 0 END) AS repair_calls
            FROM llm_calls c
        """
        params: list = []
        if run_ids is not None:
            placeholders = ",".join("?" for _ in run_ids) or "NULL"
            sql += f" WHERE c.run_id IN ({placeholders})"
            params.extend(run_ids)
        row = store.df(sql, params).iloc[0].to_dict()
    finally:
        store.close()
    return {k: (0.0 if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)) for k, v in row.items()}


def usd(prompt_tokens: float, completion_tokens: float, per_mtok_in: float, per_mtok_out: float) -> float:
    return (prompt_tokens * per_mtok_in + completion_tokens * per_mtok_out) / 1_000_000.0


def cost_per_interview(
    traces: str | Path,
    units: Sequence[PersonaRuns],
    arm: str,
    per_mtok_in: float,
    per_mtok_out: float,
) -> dict[str, float]:
    run_ids = [r.run.run_id for r in flatten(list(units), arm)]
    totals = token_totals(traces, run_ids)
    total_usd = usd(totals["prompt_tokens"], totals["completion_tokens"], per_mtok_in, per_mtok_out)
    n = max(1, len(run_ids))
    return {
        "calls_per_interview": totals["n_calls"] / n,
        "tokens_per_interview": (totals["prompt_tokens"] + totals["completion_tokens"]) / n,
        "usd_per_interview": total_usd / n,
        "total_usd": total_usd,
        "parse_failure_rate": totals["parse_failures"] / max(1.0, totals["n_calls"]),
        "repair_call_rate": totals["repair_calls"] / max(1.0, totals["n_calls"]),
    }


def cost_to_confidence(
    units: Sequence[PersonaRuns],
    arm: str,
    tau: float,
    per_mtok_in: float,
    per_mtok_out: float,
    out_fraction: float = 0.25,
) -> float:
    """USD spent on the turns needed to reach confidence.

    The number that decides whether "fewer questions" survives contact with an
    invoice. Only runs that actually reached confidence contribute; censored
    ones would otherwise make an arm that never converges look cheap.
    """
    costs = []
    for run in flatten(list(units), arm):
        n = run.questions_to_confidence(tau)
        if n is None:
            continue
        tokens = sum(t.tokens_used for t in run.turns[:n])
        costs.append(
            usd(tokens * (1 - out_fraction), tokens * out_fraction, per_mtok_in, per_mtok_out)
        )
    return float(np.mean(costs)) if costs else float("nan")


def wallclock(units: Sequence[PersonaRuns], arm: str) -> float:
    """Mean simulated candidate-facing time per interview, in seconds."""
    runs = flatten(list(units), arm)
    if not runs:
        return float("nan")
    return float(np.mean([sum(t.elapsed_seconds for t in r.turns) for r in runs]))


def schema_health(traces: str | Path) -> dict[str, float]:
    """Schema-violation and repair-success rates.

    Measured rather than asserted: the offline backend emits structurally
    invalid output at a configured rate precisely so these numbers come from
    the repair ladder actually being walked.
    """
    store = TraceStore(traces, read_only=True)
    try:
        rows = store.df(
            """
            SELECT repair_attempt, parsed_ok, count(*) AS n
            FROM llm_calls GROUP BY repair_attempt, parsed_ok
            """
        )
    finally:
        store.close()
    if rows.empty:
        return {"violation_rate": float("nan"), "repair_success_rate": float("nan")}

    total = float(rows["n"].sum())
    first_attempt_failures = float(
        rows[(rows["repair_attempt"] == 0) & (~rows["parsed_ok"])]["n"].sum()
    )
    repairs_attempted = float(rows[rows["repair_attempt"] > 0]["n"].sum())
    repairs_ok = float(rows[(rows["repair_attempt"] > 0) & (rows["parsed_ok"])]["n"].sum())
    return {
        "violation_rate": first_attempt_failures / total if total else float("nan"),
        "repair_success_rate": repairs_ok / repairs_attempted if repairs_attempted else 1.0,
        "n_calls": total,
    }
