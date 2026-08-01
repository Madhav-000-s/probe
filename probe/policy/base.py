"""The policy contract.

A policy returns one of two things: a question to ask, or a decision to stop.
It never grades, never mutates the belief state, and never sees ground truth.
Keeping it that narrow is what makes the arm comparison clean — the only thing
that varies between `fixed`, `heuristic`, `eig` and `eig+corr` is which
question comes back from this one method.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from probe.belief.state import BeliefState
from probe.models import Question, QuestionBank, StopReason, Transcript


@dataclass(frozen=True)
class Ask:
    """Ask ``question`` next."""

    question: Question
    #: Expected information gain in nats, when the arm computes one. The
    #: belief-free arms leave this None, and the traces make that visible
    #: rather than filling in a fake zero.
    eig: float | None = None
    reason: str = ""
    #: Per-candidate scores at selection time, for the trace viewer.
    diagnostics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Stop:
    reason: StopReason
    detail: str = ""


Decision = Ask | Stop


class Policy(ABC):
    """Base class for all four arms."""

    #: Arm name as it appears in traces and the results table.
    name: str = "base"

    @abstractmethod
    def next_question(
        self,
        belief: BeliefState,
        transcript: Transcript,
        bank: QuestionBank,
    ) -> Decision: ...

    def reset(self) -> None:
        """Called once per interview. Stateless arms need not override."""

    # ------------------------------------------------------------- helpers

    @staticmethod
    def available(
        bank: QuestionBank, transcript: Transcript, competency_ids: list[str] | None = None
    ) -> list[Question]:
        """Live, unasked items, optionally restricted to a competency set.

        Quarantined items are excluded here rather than at each call site, so
        no arm can accidentally draw an item the calibration pass rejected.
        """
        asked = transcript.asked_question_ids()
        pool = [q for q in bank.live() if q.id not in asked]
        if competency_ids is not None:
            allowed = set(competency_ids)
            pool = [q for q in pool if q.competency_id in allowed]
        return pool
