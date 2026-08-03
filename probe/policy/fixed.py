"""The floor baseline: a static script.

Order is derived once from the rubric — required competencies first, highest
required level first, ties broken by taxonomy order — and then followed
regardless of what the candidate says. This is what an interview loop looks
like when nobody has thought about question selection, and beating it is not
evidence of anything on its own. That is what the `heuristic` arm is for.
"""

from __future__ import annotations

from probe.belief.state import BeliefState
from probe.models import Question, QuestionBank, Rubric, StopReason, Transcript
from probe.policy.base import Ask, Decision, Policy, Stop


class FixedPolicy(Policy):
    name = "fixed"

    #: Depth of the script per competency. Three, not one: a one-deep script
    #: exhausts itself before a twelve-question budget does, and an arm that
    #: stops early because it ran out of script has not "reached confidence
    #: sooner" — it has been handed a shorter interview. Matching the bank's
    #: depth keeps the budget the binding constraint for every arm.
    DEFAULT_DEPTH = 3

    def __init__(self, rubric: Rubric, questions_per_competency: int = DEFAULT_DEPTH) -> None:
        self.rubric = rubric
        self.questions_per_competency = questions_per_competency
        self._script: list[str] | None = None

    def reset(self) -> None:
        self._script = None

    def _order(self) -> list[str]:
        """Competency ids in script order. Deterministic given the rubric."""
        comps = sorted(
            self.rubric.competencies,
            key=lambda c: (-c.required_level, self.rubric.ids.index(c.id)),
        )
        return [c.id for c in comps]

    def _build_script(self, bank: QuestionBank) -> list[str]:
        """Round-robin over competencies so the script covers breadth before
        depth. Built once per interview and never revisited."""
        per_comp: dict[str, list[Question]] = {}
        for cid in self._order():
            items = sorted(bank.for_competency(cid), key=lambda q: q.id)
            if items:
                per_comp[cid] = items

        script: list[str] = []
        for depth in range(self.questions_per_competency):
            for cid in self._order():
                items = per_comp.get(cid, [])
                if depth < len(items):
                    script.append(items[depth].id)
        return script

    def next_question(
        self, belief: BeliefState, transcript: Transcript, bank: QuestionBank
    ) -> Decision:
        if self._script is None:
            self._script = self._build_script(bank)

        asked = transcript.asked_question_ids()
        for qid in self._script:
            if qid not in asked:
                return Ask(
                    question=bank.get(qid),
                    eig=None,
                    reason=f"script position {self._script.index(qid)}",
                )
        detail = (
            "fixed script complete"
            if self._script
            else f"bank {bank.version} has no items for any rubric competency"
        )
        return Stop(StopReason.BANK_EXHAUSTED, detail)
