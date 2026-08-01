"""Budget ceilings terminate an interview with a flagged partial report and no
exception. An exception here would abort a sweep and lose every run in
flight."""

from __future__ import annotations

from dataclasses import replace

import pytest

from probe.belief.state import PriorOnlyBelief
from probe.config import Budgets
from probe.grader.base import LLMGrader
from probe.grader.fixtures import constant_grade
from probe.models import LLMRole, StopReason
from probe.policy.fixed import FixedPolicy
from probe.runtime.budgets import BudgetTracker
from probe.runtime.candidate import StubCandidate
from probe.runtime.llm import FakeLLM
from probe.runtime.loop import InterviewLoop


def _loop(rubric, bank, config, store=None, seconds=60.0):
    client = FakeLLM(by_role={LLMRole.GRADE: constant_grade(score=3)}, strict=False)
    return InterviewLoop(
        rubric=rubric,
        bank=bank,
        policy=FixedPolicy(rubric),
        belief=PriorOnlyBelief(rubric),
        grader=LLMGrader(client),
        candidate=StubCandidate(seconds=seconds),
        config=config,
        store=store,
        run_id="budget-run",
        seed=1,
    )


def test_question_ceiling_stops_the_interview(rubric, bank, config):
    config = replace(config, budgets=Budgets(max_questions=3, max_tokens=10**9, max_wallclock_seconds=1e9))
    result = _loop(rubric, bank, config).run()

    assert result.run.stop_reason is StopReason.BUDGET_QUESTIONS
    assert len(result.transcript) == 3
    assert result.run.partial is True
    assert result.report.partial is True


def test_token_ceiling_stops_the_interview(rubric, bank, config):
    config = replace(config, budgets=Budgets(max_questions=99, max_tokens=40, max_wallclock_seconds=1e9))
    result = _loop(rubric, bank, config).run()

    assert result.run.stop_reason is StopReason.BUDGET_TOKENS
    assert result.run.partial is True
    assert len(result.transcript) >= 1


def test_wallclock_ceiling_stops_the_interview(rubric, bank, config):
    config = replace(
        config, budgets=Budgets(max_questions=99, max_tokens=10**9, max_wallclock_seconds=150.0)
    )
    result = _loop(rubric, bank, config, seconds=60.0).run()

    assert result.run.stop_reason is StopReason.BUDGET_WALLCLOCK
    assert len(result.transcript) == 3  # 3 x 60s crosses 150s


def test_partial_report_is_still_a_complete_report(rubric, bank, config):
    """A breached budget must not truncate the deliverable — every competency
    still gets a verdict, just with a wider interval and a partial flag."""
    config = replace(config, budgets=Budgets(max_questions=2, max_tokens=10**9, max_wallclock_seconds=1e9))
    result = _loop(rubric, bank, config).run()

    assert set(result.report.per_competency) == set(rubric.ids)
    assert result.report.partial
    unprobed = [v for v in result.report.per_competency.values() if v.n_questions == 0]
    assert unprobed, "with a 2-question budget most competencies stay unprobed"


def test_breaching_a_budget_does_not_raise(rubric, bank, config):
    config = replace(config, budgets=Budgets(max_questions=1, max_tokens=1, max_wallclock_seconds=0.001))
    try:
        result = _loop(rubric, bank, config).run()
    except Exception as exc:  # pragma: no cover - the assertion is the point
        pytest.fail(f"budget breach raised {type(exc).__name__}: {exc}")
    assert result.run.completed is True
    assert result.run.partial is True


# ------------------------------------------------------------------ tracker


def test_tracker_reports_the_first_ceiling_reached():
    t = BudgetTracker(budgets=Budgets(max_questions=2, max_tokens=100, max_wallclock_seconds=100.0))
    assert t.exceeded() is None
    t.charge_question(10.0, 10)
    assert t.exceeded() is None
    t.charge_question(10.0, 10)
    assert t.exceeded() is StopReason.BUDGET_QUESTIONS


def test_tracker_uses_the_virtual_clock():
    t = BudgetTracker(budgets=Budgets(max_wallclock_seconds=90.0))
    t.charge_question(45.0)
    assert t.elapsed_seconds == 45.0
    t.charge_question(45.0)
    assert t.exceeded() is StopReason.BUDGET_WALLCLOCK


def test_followup_allowance():
    t = BudgetTracker(budgets=Budgets(max_followups=2))
    assert t.followups_available()
    t.charge_followup()
    t.charge_followup()
    assert not t.followups_available()
