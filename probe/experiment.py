"""Sweep orchestration.

One place that decides which interviews get run, so a results table can never
be an accidental mixture of configurations. Everything that varies is named in
:class:`SweepPlan`; everything else is held identical across arms by
construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from probe.bank.loader import load_bank
from probe.belief.correlation import CompetencyCorrelation
from probe.config import DATA_DIR, ExperimentConfig
from probe.jd import load_jd
from probe.models import Persona, QuestionBank
from probe.policy.registry import ARMS
from probe.runtime.concurrency import SweepOutcome, run_concurrently
from probe.runtime.llm import get_client
from probe.runtime.session import InterviewSpec, build_interview
from probe.runtime.tracing import TracedClient, TraceStore
from probe.sim.persona import load_population, restyle
from probe.sim.style import MAIN_SWEEP_STYLES, SWAP_NAMES

CORRELATION_PATH = DATA_DIR / "correlation.json"


@dataclass
class SweepPlan:
    arms: tuple[str, ...] = ARMS
    styles: tuple[str, ...] = MAIN_SWEEP_STYLES
    split: str = "eval"
    backend: str = "sim"
    seed: int = 20260807
    concurrency: int = 8
    style_separation: bool = True
    followups_enabled: bool = True
    #: Extra label appended to run ids, so an ablation sweep does not collide
    #: with the main one in the trace store.
    suffix: str = ""
    limit: int | None = None

    @property
    def n_runs_per_arm(self) -> int:
        return 0  # filled in by the runner, which knows the population size


@dataclass
class SweepResult:
    outcome: SweepOutcome
    n_personas: int
    n_runs: int
    plan: SweepPlan
    notes: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, object]:
        return {
            "arms": list(self.plan.arms),
            "styles": list(self.plan.styles),
            "split": self.plan.split,
            "personas": self.n_personas,
            "runs_requested": self.n_runs,
            "runs_completed": len(self.outcome.completed),
            "failures": len(self.outcome.failures),
            "wallclock_seconds": round(self.outcome.wallclock_seconds, 2),
            "interviews_per_minute": round(self.outcome.throughput_per_minute, 1),
        }


def load_correlation() -> CompetencyCorrelation | None:
    if CORRELATION_PATH.exists():
        return CompetencyCorrelation.load(CORRELATION_PATH)
    return None


def style_variants(persona: Persona, styles: tuple[str, ...]) -> list[Persona]:
    """One persona per style, with ability carried over untouched.

    The name-swap slices additionally rewrite the name on the resume. Holding
    ``theta_star`` fixed across variants is the entire experimental design of
    the fairness suite.
    """
    out = []
    for style_id in styles:
        name = SWAP_NAMES.get(style_id)
        out.append(restyle(persona, style_id, name=name))
    return out


def run_sweep(
    plan: SweepPlan,
    *,
    population_version: str,
    bank_version: str,
    config: ExperimentConfig,
    traces: str | Path,
) -> SweepResult:
    personas, meta = load_population(population_version)
    bank: QuestionBank = load_bank(bank_version)
    correlation = load_correlation()

    subjects = [p for p in personas if plan.split in ("all", p.split)]
    if plan.limit:
        subjects = subjects[: plan.limit]

    store = TraceStore(traces)
    jd_cache = {}
    jobs = []

    for persona in subjects:
        if persona.jd_id not in jd_cache:
            jd_cache[persona.jd_id] = load_jd(persona.jd_id)
        for variant in style_variants(persona, plan.styles):
            for arm in plan.arms:
                spec = InterviewSpec(
                    persona=variant,
                    jd=jd_cache[persona.jd_id],
                    arm=arm,
                    seed=plan.seed,
                    style_separation=plan.style_separation,
                    followups_enabled=plan.followups_enabled,
                    include_name=variant.style.id in SWAP_NAMES,
                    suffix=plan.suffix,
                )
                jobs.append((spec.run_id, _make_job(spec, bank, config, store, plan, correlation)))

    outcome = run_concurrently(jobs, concurrency=plan.concurrency)
    store.close()
    return SweepResult(
        outcome=outcome,
        n_personas=len(subjects),
        n_runs=len(jobs),
        plan=plan,
        notes=[f"population {meta.version} seed {meta.seed}", f"bank {bank.version}"],
    )


def _make_job(spec, bank, config, store, plan, correlation):
    def job():
        # A fresh client per interview: shared token counters across concurrent
        # runs would attribute one run's spend to another.
        traced = TracedClient(get_client(plan.backend, seed=plan.seed), store=store)
        loop = build_interview(
            spec,
            bank=bank,
            config=config,
            client=traced,
            store=store,
            correlation=correlation,
        )
        result = loop.run(resume=True)
        return result.run.run_id

    return job
