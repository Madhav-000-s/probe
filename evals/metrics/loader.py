"""Reading traces back out for analysis.

The one place ground truth is joined onto traces. Everything above this reads
``PersonaRuns`` and never sees a raw Persona, which keeps the join in a single
auditable spot rather than scattered through six metric modules.

Nothing here calls a model. If a metric needed an LLM to compute, it would be
a second thing to validate rather than a measurement of the first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from probe.models import Persona, RunRecord, Turn
from probe.runtime.tracing import TraceStore
from probe.sim.persona import load_population


@dataclass
class RunView:
    """One interview, with the ground truth it can now be scored against."""

    run: RunRecord
    turns: list[Turn]
    persona: Persona

    @property
    def arm(self) -> str:
        return self.run.arm

    @property
    def n_questions(self) -> int:
        return len(self.turns)

    def probed(self, budget: int | None = None) -> set[str]:
        """Competencies asked about within the first ``budget`` turns.

        The budget argument is load-bearing. Without it, a curve point at
        budget 2 counts competencies the interview had not yet reached, reading
        their untouched prior as though it were an estimate. That drags every
        arm toward the same value at low budgets and flattens exactly the part
        of the accuracy-vs-budget curve the comparison lives in. The
        curve-integrity cross-check is what caught it.
        """
        turns = self.turns if budget is None else self.turns[:budget]
        return {t.competency_id for t in turns}

    def final_belief(self):
        return self.turns[-1].belief_after if self.turns else None

    def belief_at(self, budget: int):
        """Posterior after ``budget`` questions.

        Read straight out of the persisted snapshots, which is what makes the
        accuracy-vs-budget curve computable post hoc instead of by re-running
        every interview once per budget.
        """
        if not self.turns:
            return None
        idx = min(budget, len(self.turns)) - 1
        return self.turns[max(idx, 0)].belief_after

    def truth_and_estimate(self, budget: int | None = None, probed_only: bool = True):
        """Paired ``(theta_star, posterior_mean)`` over competencies."""
        snapshot = self.belief_at(budget) if budget else self.final_belief()
        if snapshot is None:
            return [], []
        probed = self.probed(budget) if probed_only else set(snapshot.means)
        truth, estimate = [], []
        for cid, mean in snapshot.means.items():
            if cid in probed and cid in self.persona.theta_star:
                truth.append(self.persona.ability(cid))
                estimate.append(mean)
        return truth, estimate

    def resolved_count(self, tau: float, budget: int | None = None) -> tuple[int, int]:
        snapshot = self.belief_at(budget) if budget else self.final_belief()
        if snapshot is None:
            return 0, 0
        sds = list(snapshot.sds.values())
        return sum(1 for sd in sds if sd < tau), len(sds)

    def questions_to_confidence(self, tau: float) -> int | None:
        """First turn index (1-based) at which every competency is under tau.

        None when the interview never got there — censored, and treated as
        censored rather than silently replaced by the budget, which would
        understate the difference between arms.
        """
        for i, turn in enumerate(self.turns, start=1):
            if all(sd < tau for sd in turn.belief_after.sds.values()):
                return i
        return None


@dataclass
class PersonaRuns:
    """Every run belonging to one persona — the bootstrap's unit of analysis."""

    persona_id: str
    runs: list[RunView] = field(default_factory=list)

    def by_arm(self, arm: str) -> list[RunView]:
        return [r for r in self.runs if r.arm == arm]

    def by_style(self, style_id: str) -> list[RunView]:
        return [r for r in self.runs if r.run.style_id == style_id]


def load_views(
    traces: str | Path,
    population_version: str,
    *,
    arms: tuple[str, ...] | None = None,
    styles: tuple[str, ...] | None = None,
    style_separation: bool | None = None,
    followups_enabled: bool | None = None,
    suffix: str | None = None,
) -> list[PersonaRuns]:
    """Join committed traces to persona ground truth.

    Filters are explicit rather than inferred: a results table that silently
    pooled the style-separation-on and -off sweeps would be reporting an
    average of two different experiments.
    """
    personas, _meta = load_population(population_version)
    by_id = {p.id: p for p in personas}

    store = TraceStore(traces, read_only=True)
    try:
        grouped: dict[str, PersonaRuns] = {}
        for run_id in store.run_ids(completed_only=True):
            run = store.load_run(run_id)
            if run is None or run.persona_id not in by_id:
                continue
            if arms is not None and run.arm not in arms:
                continue
            if styles is not None and run.style_id not in styles:
                continue
            if style_separation is not None and run.style_separation != style_separation:
                continue
            if followups_enabled is not None and run.followups_enabled != followups_enabled:
                continue
            if suffix is not None and not run_id.endswith(suffix):
                continue

            view = RunView(run=run, turns=store.load_turns(run_id), persona=by_id[run.persona_id])
            grouped.setdefault(run.persona_id, PersonaRuns(run.persona_id)).runs.append(view)
        return [grouped[k] for k in sorted(grouped)]
    finally:
        store.close()


def flatten(units: list[PersonaRuns], arm: str | None = None) -> list[RunView]:
    out: list[RunView] = []
    for unit in units:
        out.extend(unit.by_arm(arm) if arm else unit.runs)
    return out


def safe_mean(values) -> float:
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else float("nan")
