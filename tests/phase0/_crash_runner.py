"""Runs an interview in a child process and hard-kills it mid-way.

Invoked by ``test_resumability.py``. ``os._exit`` skips every finaliser,
``atexit`` hook and ``finally`` block — the same courtesy a SIGKILL extends —
so anything present in the trace file afterwards is genuinely durable rather
than flushed on the way out.

Usage: ``python _crash_runner.py <trace_db> <run_id> <crash_after_turns>``
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from probe.bank.loader import stub_bank  # noqa: E402
from probe.belief.state import PriorOnlyBelief  # noqa: E402
from probe.config import Budgets, ExperimentConfig  # noqa: E402
from probe.grader.base import LLMGrader  # noqa: E402
from probe.grader.fixtures import constant_grade  # noqa: E402
from probe.models import LLMRole, Question, Transcript  # noqa: E402
from probe.policy.fixed import FixedPolicy  # noqa: E402
from probe.rubric.taxonomy import load_taxonomy  # noqa: E402
from probe.runtime.candidate import AnswerResult, StubCandidate  # noqa: E402
from probe.runtime.llm import FakeLLM  # noqa: E402
from probe.runtime.loop import InterviewLoop  # noqa: E402
from probe.runtime.tracing import TracedClient, TraceStore  # noqa: E402

CRASH_CODE = 137  # what a SIGKILL looks like from the outside


class SuicidalCandidate(StubCandidate):
    """Answers normally until ``crash_after`` turns are on disk, then dies."""

    def __init__(self, crash_after: int) -> None:
        super().__init__(candidate_id="p-resume")
        self.crash_after = crash_after

    def answer(self, question: Question, transcript: Transcript) -> AnswerResult:
        if len(transcript.turns) >= self.crash_after:
            sys.stdout.flush()
            os._exit(CRASH_CODE)
        return super().answer(question, transcript)


def build_loop(trace_db: str, run_id: str, candidate, max_questions: int = 5) -> InterviewLoop:
    """The single definition of this interview, shared by the child process and
    the resuming parent. Both sides must agree exactly or 'resume' would be
    testing two different runs."""
    tax = load_taxonomy()
    rubric = tax.stub_rubric(candidate_id="p-resume", n=6)
    bank = stub_bank(tax, competency_ids=rubric.ids, per_competency=2)
    config = ExperimentConfig(
        budgets=Budgets(
            max_questions=max_questions, max_tokens=10**9, max_wallclock_seconds=1e9
        ),
        seed_set=[7],
    )
    client = FakeLLM(by_role={LLMRole.GRADE: constant_grade(score=3)}, strict=False)
    store = TraceStore(trace_db)
    traced = TracedClient(client, store=store, run_id=run_id)
    return InterviewLoop(
        rubric=rubric,
        bank=bank,
        policy=FixedPolicy(rubric),
        belief=PriorOnlyBelief(rubric),
        grader=LLMGrader(traced),
        candidate=candidate,
        config=config,
        store=store,
        run_id=run_id,
        seed=7,
        persona_id="p-resume",
    )


def main() -> int:
    trace_db, run_id, crash_after = sys.argv[1], sys.argv[2], int(sys.argv[3])
    loop = build_loop(trace_db, run_id, SuicidalCandidate(crash_after))
    loop.run(resume=True)
    return 0  # only reached if the crash point was never hit


if __name__ == "__main__":
    raise SystemExit(main())
