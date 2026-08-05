"""The competitor arm: a well-prompted model choosing the next question.

This is the arm that makes the result defensible. Beating a fixed script proves
nothing — nobody ships a fixed script and calls it adaptive. What a reasonable
engineer actually builds is this: hand the model the rubric, the transcript so
far and the remaining questions, and ask it to pick the most informative one.
No belief state, no information theory, just a good prompt.

So real effort goes here, not into making it easy to beat. The prompt names the
things a thoughtful interviewer would weigh, the offline implementation behind
it is a genuinely sensible heuristic rather than a coin flip, and the
degraded path is competent too. If the EIG arm cannot beat this, that is the
result and it gets reported as the result.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from probe.belief.state import BeliefState
from probe.models import LLMRole, QuestionBank, Rubric, StopReason, Transcript
from probe.policy.base import Ask, Decision, Policy, Stop
from probe.runtime.llm import LLMRequest
from probe.runtime.retry import structured_call

#: How many candidate items to show the model. The full bank would blow the
#: context and bury the good options; a shortlist is what a human would build
#: too. Selection into the shortlist is deliberately naive (round-robin over
#: competencies) so the shortlisting step is not secretly doing the policy's
#: work.
SHORTLIST = 24


class PolicyChoice(BaseModel):
    question_id: str
    reason: str = Field(default="", max_length=400)


class HeuristicPolicy(Policy):
    name = "heuristic"

    def __init__(self, rubric: Rubric, client, *, seed: int = 0) -> None:
        self.rubric = rubric
        self.client = client
        self.seed = seed
        self.fallbacks = 0

    # ------------------------------------------------------------ shortlist

    def _shortlist(self, bank: QuestionBank, transcript: Transcript):
        """Round-robin over required competencies, so every one is represented
        before any is represented twice."""
        pool = self.available(bank, transcript, [c.id for c in self.rubric.required])
        if not pool:
            pool = self.available(bank, transcript)

        by_competency: dict[str, list] = {}
        for question in pool:
            by_competency.setdefault(question.competency_id, []).append(question)
        for items in by_competency.values():
            items.sort(key=lambda q: q.id)

        out = []
        depth = 0
        while len(out) < SHORTLIST and any(len(v) > depth for v in by_competency.values()):
            for cid in sorted(by_competency):
                if len(by_competency[cid]) > depth and len(out) < SHORTLIST:
                    out.append(by_competency[cid][depth])
            depth += 1
        return out

    # --------------------------------------------------------------- prompt

    def _prompt(self, transcript: Transcript, candidates) -> str:
        rubric_lines = "\n".join(
            f"  {c.id} — required level {c.required_level}/5, "
            f"resume evidence {c.evidence_in_resume:.2f}"
            + (" (resume is silent here)" if c.evidence_in_resume == 0 else "")
            for c in self.rubric.competencies
        )
        if transcript.turns:
            history = "\n".join(
                f"  turn {t.turn_idx}: [{t.competency_id}] "
                f"({t.question_id.split('::')[-1]}) scored "
                + (str(t.grade.score) if t.grade else "ungraded")
                + (f", flags: {','.join(f.value for f in t.grade.flags)}" if t.grade and t.grade.flags else "")
                for t in transcript.turns
            )
        else:
            history = "  (nothing asked yet)"

        options = "\n".join(
            f"  {q.id} | {q.competency_id} | {q.probe_family.value} | {q.text[:110]}"
            for q in candidates
        )

        return (
            "You are running a technical interview and choosing what to ask next. "
            "You have a limited number of questions, so each one has to earn its "
            "place.\n\n"
            "Weigh these:\n"
            "  - competencies the role requires most and you know least about;\n"
            "  - competencies the resume is silent on — silence is not evidence "
            "of absence, and it is not evidence of competence either;\n"
            "  - answers so far that were weak, evasive or flagged, where one "
            "more question would actually change your mind;\n"
            "  - avoid asking the same *kind* of question repeatedly; vary the "
            "probe family;\n"
            "  - do not keep digging into something already well established, in "
            "either direction.\n\n"
            f"Rubric:\n{rubric_lines}\n\n"
            f"Interview so far:\n{history}\n\n"
            f"Available questions (id | competency | family | text):\n{options}\n\n"
            'Return JSON: {"question_id": "<exact id from the list>", '
            '"reason": "<one sentence>"}\n'
        )

    # -------------------------------------------------------------- fallback

    def _fallback(self, transcript: Transcript, candidates):
        """Competent, deterministic, no model.

        Used when the model returns an unusable choice. It has to be decent:
        a fallback that picks at random would quietly convert every parse
        failure into a handicap for the competitor arm and flatter the result.
        """
        asked_counts: dict[str, int] = {}
        last_family = (
            transcript.turns[-1].question_id.split("::")[-1] if transcript.turns else None
        )
        for turn in transcript.turns:
            asked_counts[turn.competency_id] = asked_counts.get(turn.competency_id, 0) + 1

        levels = {c.id: c.required_level for c in self.rubric.competencies}
        silent = {c.id for c in self.rubric.competencies if c.evidence_in_resume == 0}

        def key(question):
            return (
                asked_counts.get(question.competency_id, 0),
                question.probe_family.value == last_family,
                -levels.get(question.competency_id, 3),
                question.competency_id not in silent,
                question.id,
            )

        return min(candidates, key=key)

    # ----------------------------------------------------------------- main

    def next_question(
        self, belief: BeliefState, transcript: Transcript, bank: QuestionBank
    ) -> Decision:
        candidates = self._shortlist(bank, transcript)
        if not candidates:
            return Stop(StopReason.BANK_EXHAUSTED, "no unasked items remain")

        by_id = {q.id: q for q in candidates}
        request = LLMRequest(
            role=LLMRole.POLICY_CHOOSE,
            prompt=self._prompt(transcript, candidates),
            seed=self.seed + len(transcript.turns),
            temperature=0.0,
            context={
                "candidates": [
                    {
                        "id": q.id,
                        "competency_id": q.competency_id,
                        "family": q.probe_family.value,
                        "required_level": next(
                            (
                                c.required_level
                                for c in self.rubric.competencies
                                if c.id == q.competency_id
                            ),
                            3,
                        ),
                        "resume_silent": next(
                            (
                                c.evidence_in_resume == 0
                                for c in self.rubric.competencies
                                if c.id == q.competency_id
                            ),
                            True,
                        ),
                    }
                    for q in candidates
                ],
                "history": [
                    {
                        "competency_id": t.competency_id,
                        "family": t.question_id.split("::")[-1],
                        "score": t.grade.score if t.grade else None,
                        "flags": [f.value for f in t.grade.flags] if t.grade else [],
                    }
                    for t in transcript.turns
                ],
            },
            run_id=transcript.run_id or None,
        )

        result = structured_call(
            self.client,
            request,
            PolicyChoice,
            postcheck=lambda c: (
                None if c.question_id in by_id else f"question_id: {c.question_id!r} is not on the list"
            ),
            degraded=lambda: PolicyChoice(
                question_id=self._fallback(transcript, candidates).id,
                reason="deterministic fallback: model choice unusable",
            ),
            max_repairs=1,
        )

        choice = result.value
        if choice is None or choice.question_id not in by_id:
            self.fallbacks += 1
            question = self._fallback(transcript, candidates)
            reason = "deterministic fallback"
        else:
            question = by_id[choice.question_id]
            reason = choice.reason or "model choice"
            if result.outcome.value == "degraded":
                self.fallbacks += 1

        # No belief state, so no EIG to report. Left as None rather than filled
        # with a zero, so the traces show honestly which arms computed one.
        return Ask(question=question, eig=None, reason=reason)
