"""Build and release the anchor set, and compute agreement.

See :mod:`evals.gold` for what this set is and — more importantly — what it is
not. No human graded it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evals.gold import build_gold_set, kappa_report, release
from probe.bank.loader import load_bank
from probe.config import ExperimentConfig
from probe.grader.base import LLMGrader
from probe.runtime.llm import get_client
from probe.runtime.tracing import TraceStore

SAMPLE_SIZE = 100


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the released anchor set.")
    parser.add_argument("--traces", default="traces/probe.duckdb")
    parser.add_argument("--n", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=20260808)
    args = parser.parse_args(argv)

    config = ExperimentConfig.load()
    bank = load_bank(config.bank_version)
    by_id = {q.id: q for q in bank.questions}

    # Sample real answers out of the committed traces, ordered deterministically
    # so the released set is reproducible from the same traces.
    store = TraceStore(args.traces, read_only=True)
    try:
        rows = store.df(
            """
            SELECT run_id, turn_idx, question_id, answer
            FROM turns
            WHERE length(answer) > 40
            ORDER BY run_id, turn_idx
            """
        )
    finally:
        store.close()

    stride = max(1, len(rows) // args.n)
    samples = []
    for i in range(0, len(rows), stride):
        row = rows.iloc[i]
        question = by_id.get(row["question_id"])
        if question is None:  # generated follow-ups are not in the bank
            continue
        samples.append((f"{row['run_id']}#{row['turn_idx']}", question, row["answer"]))
        if len(samples) >= args.n:
            break

    grader = LLMGrader(get_client("sim", seed=args.seed))
    reference = get_client("sim", seed=args.seed + 991)

    items = build_gold_set(samples, grader, reference, seed=args.seed)
    csv_path, meta_path = release(items)
    report = kappa_report(items)

    print(f"wrote {csv_path}")
    print(f"wrote {meta_path}")
    print(json.dumps(report, indent=2))

    Path("analysis/results").mkdir(parents=True, exist_ok=True)
    Path("analysis/results/gold-agreement.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
