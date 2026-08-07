"""Concurrency and generated follow-ups.

The concurrency test is looking for one specific disaster: a trace with turns
from two runs mixed together. That failure is invisible afterwards — the rows
all look plausible — and it would poison every metric computed from the store.
"""

from __future__ import annotations

import pytest

from probe.bank.followup import (
    PARENT_SHRINKAGE,
    PRIOR_A,
    FollowUpGenerator,
    FollowUpSpec,
    build_followup,
    should_follow_up,
    shrinkage_bounds,
)
from probe.belief.grid import GridBelief
from probe.config import Budgets, ExperimentConfig
from probe.grader.base import LLMGrader
from probe.grader.fixtures import constant_grade
from probe.models import GradeFlag, LLMRole
from probe.policy.fixed import FixedPolicy
from probe.runtime.candidate import StubCandidate
from probe.runtime.concurrency import (
    FlakyClient,
    RateLimitError,
    RetryingClient,
    run_concurrently,
)
from probe.runtime.llm import FakeLLM, LLMRequest, get_client
from probe.runtime.loop import InterviewLoop
from probe.runtime.tracing import TracedClient, TraceStore


def _interview(run_id, rubric, bank, store, config):
    client = FakeLLM(by_role={LLMRole.GRADE: constant_grade(score=3)}, strict=False)
    loop = InterviewLoop(
        rubric=rubric,
        bank=bank,
        policy=FixedPolicy(rubric),
        belief=GridBelief(rubric),
        grader=LLMGrader(TracedClient(client, store=store, run_id=run_id)),
        candidate=StubCandidate(),
        config=config,
        store=store,
        run_id=run_id,
        seed=3,
    )
    return loop.run(resume=False)


# ------------------------------------------------------------ concurrency


@pytest.mark.slow
def test_fifty_parallel_interviews_produce_fifty_clean_traces(rubric, bank, trace_file):
    config = ExperimentConfig(
        tau=0.0,
        budgets=Budgets(max_questions=5, max_tokens=10**9, max_wallclock_seconds=1e9),
        seed_set=[3],
    )
    store = TraceStore(trace_file)
    jobs = [
        (f"conc-{i:02d}", (lambda i=i: _interview(f"conc-{i:02d}", rubric, bank, store, config)))
        for i in range(50)
    ]
    outcome = run_concurrently(jobs, concurrency=8)

    assert outcome.failures == []
    assert len(outcome.completed) == 50

    counts = store.df(
        "SELECT run_id, count(*) AS n FROM turns GROUP BY run_id ORDER BY run_id"
    )
    assert len(counts) == 50
    assert set(counts["n"]) == {5}, "traces interleaved or turns went missing"

    # Every turn must belong to the run whose id it carries.
    mismatched = store.df(
        "SELECT count(*) AS n FROM turns t JOIN runs r USING (run_id) WHERE r.n_turns <> 5"
    )
    assert int(mismatched["n"].iloc[0]) == 0
    assert outcome.throughput_per_minute > 0
    store.close()


def test_injected_rate_limits_are_retried_with_jittered_backoff():
    slept: list[float] = []
    inner = FlakyClient(get_client("sim", seed=1), failure_rate=0.5, seed=7)
    client = RetryingClient(inner, max_retries=8, base_delay=0.001, sleep=slept.append, seed=11)

    request = LLMRequest(
        role=LLMRole.GRADE,
        prompt="p",
        context={"competency_id": "c", "answer": "quorum reads matter here", "concept_pool": []},
    )
    for _ in range(40):
        assert client.complete(request).text

    assert inner.injected > 0, "no faults were injected; the test is vacuous"
    assert client.retries > 0
    # Jitter matters: synchronised retries from a fan-out are what turn a rate
    # limit into a retry storm.
    assert len({round(d, 6) for d in slept}) > 1


def test_retries_give_up_rather_than_hanging():
    client = RetryingClient(
        FlakyClient(get_client("sim", seed=1), failure_rate=1.0),
        max_retries=3,
        base_delay=0.0,
        sleep=lambda _d: None,
    )
    with pytest.raises(RateLimitError):
        client.complete(LLMRequest(role=LLMRole.GRADE, prompt="p", context={"answer": "x"}))


@pytest.mark.slow
def test_one_failing_interview_does_not_kill_the_sweep(rubric, bank, trace_file):
    config = ExperimentConfig(
        budgets=Budgets(max_questions=3, max_tokens=10**9, max_wallclock_seconds=1e9)
    )
    store = TraceStore(trace_file)

    def boom():
        raise RuntimeError("simulated interview failure")

    jobs = [("ok-1", lambda: _interview("ok-1", rubric, bank, store, config)), ("bad", boom)]
    outcome = run_concurrently(jobs, concurrency=2)

    assert len(outcome.completed) == 1
    assert len(outcome.failures) == 1
    assert outcome.failures[0][0] == "bad"
    store.close()


# ------------------------------------------------------------- follow-ups


def test_shrinkage_stays_between_the_parent_and_the_prior():
    """A shrinkage estimator can never leave the segment between the two things
    it blends, which makes this a property rather than a range somebody
    eyeballed."""
    from probe.bank.followup import shrunk_discrimination

    for parent_a in (0.3, 0.8, 1.0, 1.9, 3.5, 5.0):
        lo, hi = shrinkage_bounds(parent_a)
        estimate = shrunk_discrimination(parent_a)
        assert lo <= estimate <= hi
        # And strictly between, unless the parent already equals the prior.
        if parent_a != PRIOR_A:
            assert lo < estimate < hi


def test_shrinkage_weight_is_conservative():
    """A brand-new question must never be trusted more than the calibrated
    parent it came from."""
    from probe.bank.followup import shrunk_discrimination

    assert 0.0 < PARENT_SHRINKAGE < 1.0
    assert shrunk_discrimination(3.0) < 3.0
    assert shrunk_discrimination(0.5) > 0.5


def test_followup_inherits_competency_and_anchors(bank):
    parent = bank.questions[0]
    followup = build_followup(parent, FollowUpSpec(text="What about quorum reads here?"), 1)

    assert followup.competency_id == parent.competency_id
    assert followup.parent_question_id == parent.id
    assert followup.is_followup
    assert [a.required_concepts for a in followup.anchors] == [
        a.required_concepts for a in parent.anchors
    ], "a follow-up with fresh anchors would be a different instrument"
    assert not followup.grm.calibrated
    assert followup.expected_seconds < parent.expected_seconds


def test_followup_trigger_is_ambiguity_not_a_low_score():
    """A confidently-wrong answer needs no follow-up."""
    assert should_follow_up(0.9, [], followups_available=True)
    assert should_follow_up(0.2, [GradeFlag.NON_ANSWER], followups_available=True)
    assert not should_follow_up(0.2, [], followups_available=True)
    assert not should_follow_up(0.9, [], followups_available=False)


def test_generator_produces_a_usable_question(bank):
    generator = FollowUpGenerator(get_client("sim", seed=5))
    parent = bank.questions[0]
    unnamed = list(parent.anchor(5).required_concepts[:2])
    followup = generator.generate(parent, "A thin answer.", _empty_transcript(), unnamed)

    assert followup is not None
    assert followup.competency_id == parent.competency_id
    assert len(followup.text) >= 15
    assert generator.generated == 1


def test_generator_degrades_rather_than_returning_nothing(bank):
    generator = FollowUpGenerator(FakeLLM(by_role={LLMRole.FOLLOWUP_GEN: "nope"}, strict=False))
    followup = generator.generate(
        bank.questions[0], "thin", _empty_transcript(), ["quorum reads"]
    )
    assert followup is not None
    assert "quorum reads" in followup.text


@pytest.mark.slow
def test_followups_are_recorded_and_resumable(taxonomy, trace_file):
    """A resumed run must rebuild generated follow-ups exactly, or the
    resumed posterior differs from the uninterrupted one."""
    from probe.bank.loader import load_bank
    from probe.jd import load_jd
    from probe.runtime.session import InterviewSpec, build_interview
    from probe.sim.persona import load_population

    personas, _meta = load_population("v2")
    persona = next(p for p in personas if p.behavior.value == "dodger")
    config = ExperimentConfig.load()

    store = TraceStore(trace_file)
    bank = load_bank(config.bank_version)
    spec = InterviewSpec(
        persona=persona, jd=load_jd(persona.jd_id), arm="fixed", seed=config.seed_set[0]
    )
    first = build_interview(
        spec,
        bank=bank,
        config=config,
        client=TracedClient(get_client("sim", seed=1), store=store),
        store=store,
    ).run(resume=False)

    followups = [t for t in first.transcript.turns if "followup" in t.question_id]
    assert followups, "a dodger should have triggered at least one follow-up"

    resumed = build_interview(
        spec,
        bank=bank,
        config=config,
        client=TracedClient(get_client("sim", seed=1), store=store),
        store=store,
    ).run(resume=True)
    assert resumed.transcript.render() == first.transcript.render()
    store.close()


def _empty_transcript():
    from probe.models import Transcript

    return Transcript(run_id="r", candidate_id="c", arm="fixed")
