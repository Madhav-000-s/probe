"""Paths and the frozen experimental constants.

After the Phase 3 gate, everything in :class:`ExperimentConfig` is frozen:
``tau``, ``epsilon``, budgets, bank version, population version and seeds
change only with a dated entry in ``results-log.md``, and any already-published
number computed under the old constants is re-run or retracted.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
TAXONOMY_PATH = DATA_DIR / "taxonomy.yaml"
BANK_DIR = DATA_DIR / "bank"
PERSONA_DIR = DATA_DIR / "personas"
GOLD_DIR = DATA_DIR / "gold"
JD_DIR = DATA_DIR / "jd"
TRACE_DIR = ROOT / "traces"
RESULTS_DIR = ROOT / "analysis" / "results"
FIGURE_DIR = ROOT / "analysis" / "figures"
SUITE_DIR = ROOT / "evals" / "suites"
EXPERIMENT_CONFIG_PATH = ROOT / "experiment-config.yaml"

DEFAULT_TRACE_DB = TRACE_DIR / "probe.duckdb"

#: Posterior grid. 61 points over [-3, 3] gives 0.1 resolution in theta, which
#: is finer than the standard error any realistic number of items achieves.
THETA_MIN = -3.0
THETA_MAX = 3.0
THETA_POINTS = 61


def ensure_dirs() -> None:
    for d in (DATA_DIR, BANK_DIR, PERSONA_DIR, GOLD_DIR, JD_DIR, TRACE_DIR, RESULTS_DIR, FIGURE_DIR):
        d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def code_commit() -> str:
    """Short commit hash, stamped onto every run for provenance.

    Falls back to ``"uncommitted"`` rather than raising: an eval run in a
    tarball with no ``.git`` is still a valid eval run, it just carries a
    weaker provenance tuple.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - env dependent
        pass
    return "uncommitted"


@dataclass(frozen=True)
class Budgets:
    """Ceilings enforced per interview. Exceeding one terminates the run with a
    flagged partial report — never an exception."""

    max_questions: int = 12
    max_tokens: int = 60_000
    max_wallclock_seconds: float = 900.0
    max_followups: int = 3


@dataclass(frozen=True)
class ExperimentConfig:
    """The frozen constants. Loaded from ``experiment-config.yaml`` when it
    exists; the defaults here are the Phase 0/1 development values."""

    #: Posterior SD below which a required competency counts as resolved.
    tau: float = 0.55
    #: EIG floor. When the best available question is worth less than this,
    #: there is nothing left worth asking.
    epsilon: float = 0.01
    #: Weight on the repeat-family penalty in the policy objective.
    repeat_family_lambda: float = 0.08
    budgets: Budgets = field(default_factory=Budgets)
    #: How many competencies the compiler puts in a rubric.
    #:
    #: This is an experimental design parameter, not a detail. At 14 against a
    #: 12-question budget the interview cannot even ask one question per
    #: competency: several are never probed, sit at their prior interval
    #: forever, and no value of tau can ever mark them resolved — which is
    #: exactly what made the first tau sweep return 0% at every candidate
    #: threshold. Six is what a focused senior technical interview actually
    #: covers, and it leaves roughly two questions per competency.
    max_competencies: int = 6
    bank_version: str = "v0"
    population_version: str = "v0"
    taxonomy_version: str = "v1"
    seed_set: list[int] = field(default_factory=lambda: [20260801])
    calibration_fraction: float = 0.6
    #: USD per million tokens, used for the cost-to-confidence metric.
    usd_per_mtok_in: float = 0.80
    usd_per_mtok_out: float = 4.00
    frozen: bool = False

    @property
    def provenance(self) -> dict[str, Any]:
        """The tuple every reported number has to carry."""
        return {
            "population_version": self.population_version,
            "bank_version": self.bank_version,
            "taxonomy_version": self.taxonomy_version,
            "code_commit": code_commit(),
            "seed_set": list(self.seed_set),
        }

    @classmethod
    def load(cls, path: Path | None = None) -> ExperimentConfig:
        path = path or EXPERIMENT_CONFIG_PATH
        if not path.exists():
            return cls()
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        budgets = Budgets(**raw.pop("budgets", {}))
        raw.pop("change_log", None)
        return cls(budgets=budgets, **raw)

    def dump(self, path: Path | None = None, change_log: list[dict[str, str]] | None = None) -> None:
        path = path or EXPERIMENT_CONFIG_PATH
        payload: dict[str, Any] = {
            "tau": self.tau,
            "epsilon": self.epsilon,
            "repeat_family_lambda": self.repeat_family_lambda,
            "budgets": {
                "max_questions": self.budgets.max_questions,
                "max_tokens": self.budgets.max_tokens,
                "max_wallclock_seconds": self.budgets.max_wallclock_seconds,
                "max_followups": self.budgets.max_followups,
            },
            "max_competencies": self.max_competencies,
            "bank_version": self.bank_version,
            "population_version": self.population_version,
            "taxonomy_version": self.taxonomy_version,
            "seed_set": list(self.seed_set),
            "calibration_fraction": self.calibration_fraction,
            "usd_per_mtok_in": self.usd_per_mtok_in,
            "usd_per_mtok_out": self.usd_per_mtok_out,
            "frozen": self.frozen,
            "change_log": change_log or [],
        }
        path.write_text(
            "# Frozen after the Phase 3 gate. Changes require a dated entry in\n"
            "# results-log.md and a re-run of every number computed under the old values.\n"
            + yaml.safe_dump(payload, sort_keys=False),
            encoding="utf-8",
        )


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
