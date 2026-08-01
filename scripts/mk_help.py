"""Print the Makefile's self-documenting targets.

A target is documented by putting ``## description`` after its colon. Keeping
the help text next to the recipe means it cannot drift out of date the way a
separately-maintained block would.
"""

from __future__ import annotations

import re
from pathlib import Path

PATTERN = re.compile(r"^([a-zA-Z0-9_-]+):.*?##\s*(.+)$")

SECTIONS = {
    "sync": "setup",
    "test": "setup",
    "lint": "setup",
    "taxonomy": "pipeline",
    "bank": "pipeline",
    "population": "pipeline",
    "calibrate": "pipeline",
    "experiment": "pipeline",
    "eval": "results",
    "figures": "results",
    "report": "results",
    "demo": "results",
    "viewer": "results",
}


def main() -> int:
    makefile = Path(__file__).resolve().parent.parent / "Makefile"
    rows = [
        (m.group(1), m.group(2))
        for line in makefile.read_text(encoding="utf-8").splitlines()
        if (m := PATTERN.match(line))
    ]
    print("probe — targets\n")
    for name, doc in rows:
        print(f"  make {name:<14} {doc}")
    print("\n  make gate-N     cumulative exit gate for phase N (0..6)")
    print("  Variables: BACKEND=sim|fake|anthropic  SEED=<int>  TRACES=<path>  RUN=<run_id>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
