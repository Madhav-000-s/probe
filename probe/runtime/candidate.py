"""The candidate side of the loop.

The interview plane knows a candidate only as "something that returns text when
asked a question". Whether that text comes from a simulated persona
conditioned on hidden ability, a canned fixture, or a human typing into a
terminal is invisible from here — which is precisely the boundary that keeps
``theta_star`` out of the interview plane.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from probe.models import Question, Transcript


@dataclass
class AnswerResult:
    text: str
    #: Simulated answering time, charged against the wall-clock budget and used
    #: by the cost-normalised efficiency metric.
    seconds: float = 60.0
    tokens: int = 0


class AnswerSource(ABC):
    """Anything that can answer an interview question."""

    #: Stable identifier used in run ids and traces.
    id: str = "unknown"
    style_id: str = "neutral"

    @abstractmethod
    def answer(self, question: Question, transcript: Transcript) -> AnswerResult: ...


class StubCandidate(AnswerSource):
    """Fixed answers, cycled by turn index.

    Indexed on ``len(transcript.turns)`` rather than an internal counter, so
    answering is a pure function of the interview state. That is not a
    stylistic preference: a candidate that remembers how many times it has been
    called cannot be resumed, because the process that resumes it starts the
    counter over and produces a different interview from turn N onward. Real
    personas obey the same rule — they are a function of (question, persona,
    seed) and nothing else.
    """

    def __init__(
        self,
        answers: list[str] | None = None,
        candidate_id: str = "stub",
        seconds: float = 60.0,
    ) -> None:
        self.id = candidate_id
        self.style_id = "neutral"
        self._answers = answers or [
            "I would start by reproducing the failure, then bisect the change set "
            "and check whether the deploy correlates with the regression.",
            "The tradeoff is between read latency and consistency; a read replica "
            "gives you throughput but introduces replication lag.",
            "I would add an idempotency key so an at-least-once retry does not "
            "double-apply the side effect.",
        ]
        self.seconds = seconds

    def answer(self, question: Question, transcript: Transcript) -> AnswerResult:
        text = self._answers[len(transcript.turns) % len(self._answers)]
        return AnswerResult(text=text, seconds=self.seconds, tokens=len(text) // 4)
