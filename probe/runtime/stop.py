"""The stop rule.

Three conditions, checked in a fixed order so the recorded reason is
unambiguous when more than one holds:

1. **confidence** — every required competency has posterior SD below ``tau``;
2. **budget** — a question, token or wall-clock ceiling is reached;
3. **no informative question left** — the best available EIG is below
   ``epsilon``.

Which one fired is logged per run, and the distribution over stop reasons is
itself a reported result: an arm that always stops on budget is not converging,
whatever its accuracy looks like.
"""

from __future__ import annotations

from dataclasses import dataclass

from probe.belief.state import BeliefState
from probe.config import ExperimentConfig
from probe.models import StopReason
from probe.runtime.budgets import BudgetTracker


@dataclass
class StopRule:
    config: ExperimentConfig

    def check(
        self,
        belief: BeliefState,
        budget: BudgetTracker,
        best_eig: float | None = None,
        n_asked: int = 0,
    ) -> StopReason | None:
        """Return the reason to stop, or None to continue.

        ``n_asked`` guards the confidence check: a rubric whose priors happen
        to start tighter than ``tau`` would otherwise stop at turn zero having
        asked nothing, which is a configuration bug masquerading as
        efficiency.
        """
        if n_asked > 0 and belief.all_resolved(self.config.tau):
            return StopReason.CONFIDENCE

        exceeded = budget.exceeded()
        if exceeded is not None:
            return exceeded

        if best_eig is not None and best_eig < self.config.epsilon:
            return StopReason.NO_INFORMATIVE_QUESTION

        return None
