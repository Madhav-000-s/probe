"""``make eval`` — every number in the report, computed from committed traces.

The reproducibility contract: run this twice on the same traces and the results
table is byte-identical. That is why nothing here samples without a fixed seed,
nothing calls a model, and every float is rounded at the point of writing
rather than at the point of display.

Outputs:
  analysis/results/main-table.json      the policy x metric table with CIs
  analysis/results/curves.json          accuracy-vs-budget, one series per arm
  analysis/results/*.parquet            tidy frames for notebooks
  analysis/figures/accuracy-vs-budget.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from evals.metrics import cost, efficiency, recovery
from evals.metrics.bootstrap import bootstrap_ci, paired_difference_ci
from evals.metrics.loader import PersonaRuns, load_views
from probe.config import FIGURE_DIR, RESULTS_DIR, ExperimentConfig
from probe.policy.registry import ARMS

BUDGETS = tuple(range(1, 16))
#: Fixed so two runs of `make eval` bootstrap the same resamples.
BOOTSTRAP_SEED = 20260808


def _round(value: Any, places: int = 4) -> Any:
    if isinstance(value, float):
        return None if not np.isfinite(value) else round(value, places)
    if isinstance(value, dict):
        return {k: _round(v, places) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_round(v, places) for v in value]
    return value


def arm_row(units: list[PersonaRuns], arm: str, config: ExperimentConfig, traces: str) -> dict[str, Any]:
    """One row of the main table: every headline metric with a bootstrap CI."""
    tau = config.tau
    n_runs = sum(len(u.by_arm(arm)) for u in units)

    rho = bootstrap_ci(units, lambda s: recovery.spearman_rho(s, arm), seed=BOOTSTRAP_SEED)
    ece = bootstrap_ci(
        units, lambda s: recovery.expected_calibration_error(s, arm), seed=BOOTSTRAP_SEED
    )
    resolved = bootstrap_ci(
        units, lambda s: efficiency.resolved_fraction(s, arm, tau), seed=BOOTSTRAP_SEED
    )
    qtc = bootstrap_ci(
        units, lambda s: efficiency.questions_to_confidence(s, arm, tau)[0], seed=BOOTSTRAP_SEED
    )
    sd = bootstrap_ci(units, lambda s: efficiency.mean_posterior_sd(s, arm), seed=BOOTSTRAP_SEED)

    precision, recall, f1 = recovery.decision_precision_recall(units, arm)
    _mean_q, reached = efficiency.questions_to_confidence(units, arm, tau)

    return {
        "arm": arm,
        "n_runs": n_runs,
        "recovery_rho": rho.to_dict(),
        "ece": ece.to_dict(),
        "resolved_fraction": resolved.to_dict(),
        "questions_to_confidence": qtc.to_dict(),
        "reached_confidence_fraction": reached,
        "mean_posterior_sd": sd.to_dict(),
        "coverage_80": recovery.coverage_at(units, arm, 0.8),
        "decision_precision": precision,
        "decision_recall": recall,
        "decision_f1": f1,
        "mean_questions": efficiency.mean_questions(units, arm),
        "seconds_to_confidence": efficiency.seconds_to_confidence(units, arm, tau),
        "followup_rate": efficiency.followup_rate(units, arm),
        "stop_reasons": efficiency.stop_reason_distribution(units, arm),
        "cost": cost.cost_per_interview(
            traces, units, arm, config.usd_per_mtok_in, config.usd_per_mtok_out
        ),
        "cost_to_confidence_usd": cost.cost_to_confidence(
            units, arm, tau, config.usd_per_mtok_in, config.usd_per_mtok_out
        ),
        "wallclock_seconds": cost.wallclock(units, arm),
    }


def contrasts(units: list[PersonaRuns], arms: tuple[str, ...], config: ExperimentConfig) -> list[dict]:
    """Paired differences against the two baselines.

    Paired on personas, because every arm interviews the same population and
    treating the arms as independent samples throws that away.
    """
    out = []
    for baseline in ("fixed", "heuristic"):
        if baseline not in arms:
            continue
        for arm in arms:
            if arm == baseline:
                continue
            for name, fn in (
                ("recovery_rho", recovery.spearman_rho),
                ("resolved_fraction", lambda s, a: efficiency.resolved_fraction(s, a, config.tau)),
                ("mean_posterior_sd", efficiency.mean_posterior_sd),
            ):
                interval = paired_difference_ci(
                    units,
                    lambda s, a=arm, f=fn: f(s, a),
                    lambda s, b=baseline, f=fn: f(s, b),
                    seed=BOOTSTRAP_SEED,
                )
                out.append(
                    {
                        "metric": name,
                        "arm": arm,
                        "baseline": baseline,
                        "difference": interval.to_dict(),
                        "excludes_zero": interval.excludes(0.0),
                    }
                )
    return out


def build_curves(units: list[PersonaRuns], arms: tuple[str, ...]) -> dict[str, dict[int, float]]:
    return {arm: efficiency.accuracy_vs_budget(units, arm, BUDGETS) for arm in arms}


def plot_curves(curves: dict[str, dict[int, float]], path: Path) -> Path:
    """The centrepiece figure. Generated, never hand-drawn."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=160)
    markers = {"fixed": "o", "heuristic": "s", "eig": "^", "eig+corr": "D"}

    for arm, curve in curves.items():
        xs = [b for b, v in sorted(curve.items()) if v is not None and np.isfinite(v)]
        ys = [curve[b] for b in xs]
        if xs:
            ax.plot(xs, ys, marker=markers.get(arm, "o"), markersize=4, linewidth=1.8, label=arm)

    ax.set_xlabel("questions asked")
    ax.set_ylabel(r"recovery  $\rho(\hat\mu,\ \theta^*)$")
    ax.set_title("Skill recovery as a function of question budget")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def tidy_frames(units: list[PersonaRuns]) -> dict[str, pd.DataFrame]:
    """Long-format frames for notebooks, so analysis never re-parses JSON."""
    rows = []
    for unit in units:
        for run in unit.runs:
            rows.append(
                {
                    "run_id": run.run.run_id,
                    "arm": run.arm,
                    "persona_id": run.run.persona_id,
                    "style_id": run.run.style_id,
                    "behavior": run.persona.behavior.value,
                    "split": run.persona.split,
                    "n_questions": run.n_questions,
                    "stop_reason": run.run.stop_reason.value if run.run.stop_reason else None,
                    "total_tokens": run.run.total_tokens,
                    "style_separation": run.run.style_separation,
                    "followups_enabled": run.run.followups_enabled,
                }
            )
    return {"runs": pd.DataFrame(rows)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute every metric from committed traces.")
    parser.add_argument("--traces", default="traces/probe.duckdb")
    parser.add_argument("--suites", default="evals/suites")
    parser.add_argument("--population", default="")
    parser.add_argument("--figures-only", action="store_true")
    parser.add_argument("--out", default=str(RESULTS_DIR))
    args = parser.parse_args(argv)

    config = ExperimentConfig.load()
    population = args.population or config.population_version
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    units = load_views(
        args.traces,
        population,
        arms=ARMS,
        style_separation=True,
        followups_enabled=True,
        suffix="",
    )
    if not units:
        print("no completed runs found; run `make experiment` first")
        return 1

    present = tuple(a for a in ARMS if any(u.by_arm(a) for u in units))
    curves = build_curves(units, present)
    figure = plot_curves(curves, FIGURE_DIR / "accuracy-vs-budget.png")

    if args.figures_only:
        print(f"wrote {figure}")
        return 0

    table = {
        "provenance": config.provenance,
        "tau": config.tau,
        "budget": config.budgets.max_questions,
        "n_personas": len(units),
        "arms": [_round(arm_row(units, arm, config, args.traces)) for arm in present],
        "contrasts": _round(contrasts(units, present, config)),
        "schema_health": _round(cost.schema_health(args.traces)),
        "per_competency_rho": {
            arm: _round(recovery.per_competency_rho(units, arm)) for arm in present
        },
    }

    (out_dir / "main-table.json").write_text(
        json.dumps(table, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / "curves.json").write_text(
        json.dumps(_round(curves), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for name, frame in tidy_frames(units).items():
        frame.to_parquet(out_dir / f"{name}.parquet", index=False)

    print(f"wrote {out_dir / 'main-table.json'}")
    print(f"wrote {figure}")
    print(render_table(table))
    return 0


def render_table(table: dict[str, Any]) -> str:
    """The results table as plain text, for the terminal and the README."""
    header = (
        f"{'arm':<11} {'recovery rho':>22} {'resolved':>20} "
        f"{'q-to-conf':>18} {'ECE':>18} {'$/interview':>12}"
    )
    lines = [
        "",
        f"tau={table['tau']}  budget={table['budget']}  personas={table['n_personas']}",
        header,
        "-" * len(header),
    ]
    def ci(row: dict, key: str, places: int = 3) -> str:
        d = row[key]
        if d["point"] is None:
            return "n/a"
        lo = "n/a" if d["lo"] is None else f"{d['lo']:.{places}f}"
        hi = "n/a" if d["hi"] is None else f"{d['hi']:.{places}f}"
        return f"{d['point']:.{places}f} [{lo},{hi}]"

    for row in table["arms"]:
        lines.append(
            f"{row['arm']:<11} {ci(row, 'recovery_rho'):>22} "
            f"{ci(row, 'resolved_fraction'):>20} "
            f"{ci(row, 'questions_to_confidence', 1):>18} {ci(row, 'ece'):>18} "
            f"{row['cost']['usd_per_interview']:>12.5f}"
        )
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
