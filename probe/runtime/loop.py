"""The interview loop.

This is the ~300 lines that had to stay hand-written. No agent framework, no
callback graph, no hidden retry policy: the control flow *is* the thing being
measured, and every branch in it has to be explainable to someone reading the
results table.

One turn is: consult the stop rule, ask the policy for a question, get an
answer, grade it, fold the grade into the belief, snapshot, persist. The
persist happens before the next turn begins, which is what makes a killed run
resumable and what makes the trace a complete record rather than a summary.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from probe.belief.state import BeliefState
from probe.config import ExperimentConfig
from probe.grader.base import Grader
from probe.models import (
    CompetencyVerdict,
    InterviewReport,
    QuestionBank,
    Rubric,
    RunRecord,
    StopReason,
    Transcript,
    Turn,
)
from probe.policy.base import Ask, Policy, Stop
from probe.runtime.budgets import BudgetTracker
from probe.runtime.candidate import AnswerSource
from probe.runtime.stop import StopRule
from probe.runtime.tracing import TraceStore


@dataclass
class InterviewResult:
    run: RunRecord
    transcript: Transcript
    report: InterviewReport
    #: True when the loop picked up a partially-persisted run rather than
    #: starting from turn 0.
    resumed_from: int | None = None
    notes: list[str] = field(default_factory=list)


class InterviewLoop:
    def __init__(
        self,
        *,
        rubric: Rubric,
        bank: QuestionBank,
        policy: Policy,
        belief: BeliefState,
        grader: Grader,
        candidate: AnswerSource,
        config: ExperimentConfig,
        store: TraceStore | None = None,
        run_id: str = "run",
        seed: int = 0,
        persona_id: str | None = None,
        style_id: str | None = None,
        style_separation: bool = True,
        followups_enabled: bool = True,
        grader_model: str = "sim-grader",
    ) -> None:
        self.rubric = rubric
        self.bank = bank
        self.policy = policy
        self.belief = belief
        self.grader = grader
        self.candidate = candidate
        self.config = config
        self.store = store
        self.run_id = run_id
        self.seed = seed
        self.persona_id = persona_id or candidate.id
        self.style_id = style_id or candidate.style_id
        self.style_separation = style_separation
        self.followups_enabled = followups_enabled
        self.grader_model = grader_model

        self.budget = BudgetTracker(budgets=config.budgets)
        self.stop_rule = StopRule(config)
        self.transcript = Transcript(
            run_id=run_id, candidate_id=self.persona_id, arm=policy.name
        )
        self.notes: list[str] = []

    # ------------------------------------------------------------------ run

    def run(self, resume: bool = True) -> InterviewResult:
        started = time.perf_counter()
        resumed_from = self._maybe_resume() if resume else None

        run = RunRecord(
            run_id=self.run_id,
            arm=self.policy.name,
            persona_id=self.persona_id,
            style_id=self.style_id,
            bank_version=self.bank.version,
            population_version=self.config.population_version,
            code_commit=self.config.provenance["code_commit"],
            seed=self.seed,
            followups_enabled=self.followups_enabled,
            style_separation=self.style_separation,
            grader_model=self.grader_model,
        )
        if self.store is not None:
            self.store.upsert_run(run)

        if resumed_from is None:
            self.policy.reset()

        stop_reason = self._interview()

        run.stop_reason = stop_reason
        run.n_turns = len(self.transcript.turns)
        run.total_tokens = self.budget.tokens_used
        run.wallclock_seconds = self.budget.elapsed_seconds
        run.usd_cost = self.budget.usd_cost(
            self.config.usd_per_mtok_in, self.config.usd_per_mtok_out
        )
        run.partial = stop_reason in {
            StopReason.BUDGET_QUESTIONS,
            StopReason.BUDGET_TOKENS,
            StopReason.BUDGET_WALLCLOCK,
            StopReason.UNRECOVERABLE,
        }
        run.completed = True
        if self.store is not None:
            self.store.upsert_run(run)

        report = self.build_report(stop_reason, partial=run.partial)
        self.notes.append(f"real elapsed {time.perf_counter() - started:.3f}s")
        return InterviewResult(
            run=run,
            transcript=self.transcript,
            report=report,
            resumed_from=resumed_from,
            notes=list(self.notes),
        )

    # -------------------------------------------------------------- internals

    def _maybe_resume(self) -> int | None:
        """Rebuild in-memory state from committed turns.

        Grades are replayed through the belief update rather than restored from
        a snapshot, so a resumed run's posterior is computed by exactly the
        same code path as a fresh one. Restoring the snapshot instead would
        make resume a second, untested inference implementation.
        """
        if self.store is None or not self.store.run_exists(self.run_id):
            return None
        turns = self.store.load_turns(self.run_id)
        if not turns:
            return None

        for turn in turns:
            self.transcript.turns.append(turn)
            self.budget.charge_question(turn.elapsed_seconds, turn.tokens_used)
            if turn.grade is not None:
                self.belief.update(self.bank.get(turn.question_id), turn.grade.score)
        self.notes.append(f"resumed after turn {turns[-1].turn_idx}")
        return turns[-1].turn_idx

    def _interview(self) -> StopReason:
        while True:
            n_asked = len(self.transcript.turns)

            early = self.stop_rule.check(self.belief, self.budget, n_asked=n_asked)
            if early is not None:
                return early

            decision = self.policy.next_question(self.belief, self.transcript, self.bank)
            if isinstance(decision, Stop):
                return decision.reason
            assert isinstance(decision, Ask)

            if decision.eig is not None and decision.eig < self.config.epsilon:
                return StopReason.NO_INFORMATIVE_QUESTION

            self._run_turn(decision, turn_idx=n_asked)

    def _run_turn(self, decision: Ask, turn_idx: int) -> None:
        question = decision.question

        answered = self.candidate.answer(question, self.transcript)
        outcome = self.grader.grade(
            question, answered.text, self.transcript, seed=self.seed + turn_idx
        )

        if outcome.grade is not None:
            self.belief.update(question, outcome.grade.score)
        else:
            # Never raise. A turn the grader could not produce a usable verdict
            # for is recorded as such and the interview continues; the
            # unrecoverable rate is a reported robustness number.
            self.notes.append(f"turn {turn_idx}: unrecoverable ({'; '.join(outcome.errors)})")

        grader_tokens = _grader_tokens(self.grader)
        self.budget.charge_question(answered.seconds, answered.tokens + grader_tokens)

        turn = Turn(
            run_id=self.run_id,
            turn_idx=turn_idx,
            question_id=question.id,
            competency_id=question.competency_id,
            question_text=question.text,
            answer=answered.text,
            grade=outcome.grade,
            belief_after=self.belief.snapshot(),
            eig_at_selection=decision.eig,
            selection_reason=decision.reason,
            elapsed_seconds=answered.seconds,
            tokens_used=answered.tokens + grader_tokens,
            unrecoverable=outcome.grade is None,
        )
        self.transcript.turns.append(turn)
        if self.store is not None:
            # Persist before the next turn starts. This ordering is the whole
            # resumability guarantee.
            self.store.upsert_turn(turn)

    # ---------------------------------------------------------------- report

    def build_report(self, stop_reason: StopReason, partial: bool) -> InterviewReport:
        verdicts: dict[str, CompetencyVerdict] = {}
        for comp in self.rubric.competencies:
            turns = [t for t in self.transcript.turns if t.competency_id == comp.id]
            evidence = [s for t in turns if t.grade for s in t.grade.evidence_spans]
            flags = sorted({f for t in turns if t.grade for f in t.grade.flags})
            lo, hi = self.belief.credible_interval(comp.id, 0.8)
            verdicts[comp.id] = CompetencyVerdict(
                competency_id=comp.id,
                posterior_mean=self.belief.mean(comp.id),
                posterior_sd=self.belief.sd(comp.id),
                ci80=(lo, hi),
                required_level=comp.required_level,
                n_questions=len(turns),
                evidence=evidence[:5],
                flags=flags,
                confident=self.belief.sd(comp.id) < self.config.tau,
            )
        return InterviewReport(
            run_id=self.run_id,
            candidate_id=self.persona_id,
            arm=self.policy.name,
            per_competency=verdicts,
            stop_reason=stop_reason,
            partial=partial,
            n_questions=len(self.transcript.turns),
            notes=list(self.notes),
        )


def _grader_tokens(grader: Grader) -> int:
    """Tokens the grader consumed since this was last called.

    Only a traced client can answer that, so an untraced one contributes zero
    to the in-loop budget. Nothing is lost: the cost metric recomputes exact
    totals from the ``llm_calls`` table at eval time. This number exists to
    enforce the ceiling *during* the interview, where an approximation that
    errs low is preferable to an attribute error.
    """
    take = getattr(getattr(grader, "client", None), "take_token_delta", None)
    return take() if callable(take) else 0
