"""Setting the experimental constants empirically.

``tau`` is the posterior SD below which a competency counts as resolved. Picked
badly it makes every metric degenerate: too tight and no arm ever reaches
confidence so questions-to-confidence is censored for everyone; too loose and
every arm resolves everything at turn one.

PLAN.md fixes the recipe: choose tau so the *fixed* arm at the frozen budget
reaches confidence on roughly 70% of honest personas. Anchoring on the weakest
arm means the metric has headroom to show a difference in either direction.

Measured on the **calibration split only**. Choosing a constant by looking at
the data you will later report against is exactly the circularity the held-out
design exists to prevent.
"""

from __future__ import annotations

import numpy as np

from probe.config import Budgets, ExperimentConfig
from probe.jd import load_jd
from probe.models import Persona, QuestionBank
from probe.runtime.session import InterviewSpec, build_interview
from probe.runtime.tracing import TracedClient, TraceStore

#: Candidate values, coarse enough to be honest about the resolution of a
#: 36-persona calibration split.
TAU_CANDIDATES = (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 1.00)


def sweep_tau(
    personas: list[Persona],
    bank: QuestionBank,
    client,
    *,
    budget: int = 12,
    target: float = 0.70,
    seed: int = 20260807,
    max_competencies: int = 6,
) -> tuple[list[tuple[float, float, float]], float]:
    """Return ``([(tau, fully_resolved_fraction, mean_sd)], chosen_tau)``.

    The interviews are run once and every candidate tau is evaluated against
    the same final posteriors. Re-running the sweep per tau would be wasteful
    and would also change the interviews themselves, since tau feeds the stop
    rule — and comparing runs of different lengths would confound the choice.
    """
    config = ExperimentConfig(
        # Deliberately unreachable during the run, so no interview stops early
        # on confidence and every one spends the full budget. tau is then
        # applied to the resulting posteriors post hoc.
        tau=0.0,
        budgets=Budgets(max_questions=budget, max_tokens=10**9, max_wallclock_seconds=1e9),
        seed_set=[seed],
        bank_version=bank.version,
        max_competencies=max_competencies,
    )

    store = TraceStore(":memory:")
    finals: list[list[float]] = []
    jd_cache: dict[str, object] = {}

    for persona in personas:
        if persona.jd_id not in jd_cache:
            jd_cache[persona.jd_id] = load_jd(persona.jd_id)
        spec = InterviewSpec(
            persona=persona,
            jd=jd_cache[persona.jd_id],
            arm="fixed",
            seed=seed,
            followups_enabled=False,
        )
        loop = build_interview(
            spec,
            bank=bank,
            config=config,
            client=TracedClient(client, store=store),
            store=store,
        )
        loop.run(resume=False)
        finals.append([loop.belief.sd(c.id) for c in loop.rubric.required])
    store.close()

    table: list[tuple[float, float, float]] = []
    for tau in TAU_CANDIDATES:
        fully = float(np.mean([all(sd < tau for sd in row) for row in finals if row]))
        mean_sd = float(np.mean([np.mean(row) for row in finals if row]))
        table.append((tau, fully, mean_sd))

    chosen = min(table, key=lambda row: abs(row[1] - target))[0]
    return table, chosen
