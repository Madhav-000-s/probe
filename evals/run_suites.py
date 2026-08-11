"""Robustness and fairness suites.

Kept separate from ``run_eval`` because they read different sweeps: the main
table comes from the four-arm sweep, while fairness needs the eight-slice
sweep run twice — once with the content–style separation intervention on and
once off. Pooling those would report the average of two different experiments.

Writes:
  analysis/results/robustness.json
  analysis/results/fairness.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evals.metrics import fairness, reliability, robustness
from evals.metrics.loader import load_views
from probe.config import RESULTS_DIR, ExperimentConfig, load_dotenv
from probe.grader.base import LLMGrader
from probe.policy.registry import ARMS
from probe.runtime.llm import get_client
from probe.sim.style import FAIRNESS_PAIRS

FAIRNESS_STYLES = tuple(sorted({s for pair in FAIRNESS_PAIRS for s in pair}))


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Robustness and fairness suites.")
    parser.add_argument("--traces", default="traces/probe.duckdb")
    parser.add_argument("--arm", default="eig", help="Arm the fairness suite is run on.")
    parser.add_argument("--out", default=str(RESULTS_DIR))
    parser.add_argument(
        "--backend",
        default="sim",
        help="Grader backend for the re-grading suites. The interviews' own grader "
        "was configured this way; using a different one measures two things at once.",
    )
    parser.add_argument("--model", default="", help="Live backend only.")
    parser.add_argument(
        "--only",
        default="",
        help="Comma-separated subset of robustness,reliability,fairness. Fairness "
        "needs the sep-on/sep-off sweeps, which a partial run may not have.",
    )
    parser.add_argument(
        "--reliability-samples",
        type=int,
        default=120,
        help="Answers re-graded. Each costs ten grader calls, so this is the "
        "knob that decides what a live reliability run costs.",
    )
    args = parser.parse_args(argv)
    only = {s.strip() for s in args.only.split(",") if s.strip()} or {
        "robustness",
        "reliability",
        "fairness",
    }

    config = ExperimentConfig.load()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- robustness, from the main four-arm sweep
    main_units = load_views(
        args.traces,
        config.population_version,
        arms=ARMS,
        style_separation=True,
        followups_enabled=True,
        suffix="",
    )
    # Resistance is a counterfactual, so this one suite needs a grader
    # configured exactly as the interviews' was.
    client_kwargs = {"model": args.model} if args.model and args.backend == "anthropic" else {}
    grader = LLMGrader(get_client(args.backend, seed=config.seed_set[0], **client_kwargs))

    provenance = {**config.provenance, "grader_backend": args.backend, "grader_model": grader.client.model}

    robustness_payload = None
    if "robustness" in only:
        report = robustness.measure(main_units, grader)
        per_arm = {
            arm: robustness.measure([u for u in main_units if u.by_arm(arm)], grader).to_dict()
            for arm in ARMS
        }
        robustness_payload = {
            "provenance": provenance,
            "pooled": report.to_dict(),
            "per_arm": per_arm,
            "mean_score_by_behaviour": robustness.behaviour_score_profile(main_units),
        }
        (out_dir / "robustness.json").write_text(
            json.dumps(robustness_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    # ---- reliability: re-grade real answers under varied conditions
    #
    # Needs a grader rather than only the trace store, because test-retest and
    # position bias are questions about how the grader behaves when the same
    # answer is put to it again, and that second grading never happened during
    # the interview.
    reliability_report = None
    if "reliability" in only:
        from probe.bank.loader import load_bank

        bank = {q.id: q for q in load_bank(config.bank_version).questions}
        samples = []
        for unit in main_units:
            for run_view in unit.by_arm("eig"):
                for turn in run_view.turns:
                    question = bank.get(turn.question_id)
                    if question is not None and len(turn.answer) > 40:
                        samples.append((question, turn.answer))
        n = args.reliability_samples
        samples = samples[:: max(1, len(samples) // n)][:n]
        reliability_report = reliability.measure(grader, samples)
        (out_dir / "reliability.json").write_text(
            json.dumps(
                {"provenance": provenance, **reliability_report.to_dict()},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    if "fairness" not in only:
        if robustness_payload is not None:
            print(render_robustness(robustness_payload))
        if reliability_report is not None:
            print_reliability(reliability_report)
        return 0

    # ---- fairness, from the eight-slice sweeps
    on = load_views(
        args.traces,
        config.population_version,
        arms=(args.arm,),
        styles=FAIRNESS_STYLES,
        style_separation=True,
        suffix="sep-on",
    )
    off = load_views(
        args.traces,
        config.population_version,
        arms=(args.arm,),
        styles=FAIRNESS_STYLES,
        style_separation=False,
        suffix="sep-off",
    )
    if not on or not off:
        print("fairness sweeps missing; run `probe experiment --suffix sep-on|sep-off`")
        return 1

    before = fairness.measure(off, args.arm, style_separation=False)
    after = fairness.measure(on, args.arm, style_separation=True)
    exact, n_pairs, max_diff = fairness.name_swap_is_exact(on, args.arm)

    fairness_payload = {
        "provenance": provenance,
        "arm": args.arm,
        "intervention_off": before.to_dict(),
        "intervention_on": after.to_dict(),
        "delta": fairness.intervention_delta(before, after),
        "name_swap": {"exact": exact, "n_pairs": n_pairs, "max_difference": max_diff},
    }
    (out_dir / "fairness.json").write_text(
        json.dumps(fairness_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    if robustness_payload is not None:
        print(render_robustness(robustness_payload))
    print(render_fairness(fairness_payload))
    if reliability_report is not None:
        print_reliability(reliability_report)
    print(f"\nwrote {out_dir / 'fairness.json'}")
    return 0


def print_reliability(report) -> None:
    print("\nreliability (Q2)")
    for key, value in report.to_dict().items():
        print(f"  {key:<28}{value}")


def render_robustness(robustness_payload: dict) -> str:
    r = robustness_payload["pooled"]
    return "\n".join(
        [
            "",
            "robustness",
            f"  injection resistance      {r['injection_resistance']:.3f} "
            f"({r['injection_attempts']} attempts, {r['flagged_injection_rate']:.0%} flagged, "
            f"mean score inflation {r['mean_score_inflation']:+.3f})",
            f"  bluff detection AUC       {r['bluff_auc']:.3f}",
            f"  overclaim recall          {r['contradiction_recall']:.3f}",
            f"  non-answer recall         {r['non_answer_recall']:.3f}",
            "  mean score by behaviour   "
            + ", ".join(
                f"{k}={v}" for k, v in robustness_payload["mean_score_by_behaviour"].items()
            ),
        ]
    )


def render_fairness(fairness_payload: dict) -> str:
    lines = [
        "",
        "fairness — style drift, intervention off vs on",
        f"  {'slice':<28}{'off':>8}{'on':>8}{'reduction':>12}",
    ]
    for row in fairness_payload["delta"]["slices"]:
        lines.append(
            f"  {row['slice']:<28}{row['abs_drift_off']:>8.3f}"
            f"{row['abs_drift_on']:>8.3f}{row['reduction']:>12.3f}"
        )
    delta = fairness_payload["delta"]
    lines.append(
        f"  {'MEAN':<28}{delta['mean_abs_drift_off']:>8.3f}"
        f"{delta['mean_abs_drift_on']:>8.3f}{delta['reduction']:>12.3f}"
    )
    worst = fairness_payload["intervention_on"]["worst_slice"]
    if worst:
        lines.append(
            f"\n  residual worst slice: {worst['slice_a']} vs {worst['slice_b']} "
            f"(abs drift {worst['abs_drift']:.3f}), "
            f"max-drift competency: {worst['max_drift_competency']}"
        )
    swap = fairness_payload["name_swap"]
    lines.append(
        f"  name swap exact: {swap['exact']} "
        f"({swap['n_pairs']} pairs, max diff {swap['max_difference']:.2e})"
    )
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
