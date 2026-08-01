"""Per-interview ceilings.

A budget breach is a normal outcome, not an error. The loop asks the tracker
before each turn whether it may continue; if not, it stops, emits a report
flagged ``partial``, and records which ceiling fired. Nothing raises. An
exception here would abort a sweep partway and lose the runs already in
flight.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from probe.config import Budgets
from probe.models import StopReason


@dataclass
class BudgetTracker:
    budgets: Budgets
    questions_asked: int = 0
    tokens_used: int = 0
    followups_used: int = 0
    #: Virtual clock in seconds. The sim backend advances it by each question's
    #: expected answer time, so time-normalised efficiency is measurable
    #: without waiting in real time. Under a live provider this tracks the
    #: real elapsed wall clock instead.
    virtual_seconds: float = 0.0
    _started: float = field(default_factory=time.perf_counter)
    use_virtual_clock: bool = True

    @property
    def elapsed_seconds(self) -> float:
        if self.use_virtual_clock:
            return self.virtual_seconds
        return time.perf_counter() - self._started

    def charge_question(self, seconds: float, tokens: int = 0) -> None:
        self.questions_asked += 1
        self.virtual_seconds += seconds
        self.tokens_used += tokens

    def charge_tokens(self, tokens: int) -> None:
        self.tokens_used += tokens

    def charge_followup(self) -> None:
        self.followups_used += 1

    def exceeded(self) -> StopReason | None:
        """Which ceiling, if any, has been reached. Checked before selecting
        the next question, so the returned reason is the one that stops the
        interview."""
        if self.questions_asked >= self.budgets.max_questions:
            return StopReason.BUDGET_QUESTIONS
        if self.tokens_used >= self.budgets.max_tokens:
            return StopReason.BUDGET_TOKENS
        if self.elapsed_seconds >= self.budgets.max_wallclock_seconds:
            return StopReason.BUDGET_WALLCLOCK
        return None

    def followups_available(self) -> bool:
        return self.followups_used < self.budgets.max_followups

    def usd_cost(self, per_mtok_in: float, per_mtok_out: float, out_fraction: float = 0.25) -> float:
        """Blended cost estimate. ``out_fraction`` is the share of tokens that
        were completions; the trace store keeps the exact split, so the
        cost metric recomputes this properly at eval time. This method is the
        cheap in-loop approximation used for budget display only."""
        tin = self.tokens_used * (1.0 - out_fraction)
        tout = self.tokens_used * out_fraction
        return (tin * per_mtok_in + tout * per_mtok_out) / 1_000_000.0

    def snapshot(self) -> dict[str, float]:
        return {
            "questions_asked": float(self.questions_asked),
            "tokens_used": float(self.tokens_used),
            "elapsed_seconds": self.elapsed_seconds,
            "followups_used": float(self.followups_used),
        }
