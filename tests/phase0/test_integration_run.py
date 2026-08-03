"""End-to-end: one fake-LLM interview, fully traced and byte-reconstructable.

The reconstruction check is the load-bearing one. Every metric in this project
is computed from the trace store rather than from live objects, so if a
transcript rebuilt from DuckDB differs from the one that was actually
conducted — by so much as a character — the results table is measuring a
fiction.
"""

from __future__ import annotations

from probe.belief.state import PriorOnlyBelief
from probe.grader.base import LLMGrader
from probe.grader.fixtures import length_proportional_grade
from probe.models import LLMRole, StopReason
from probe.policy.fixed import FixedPolicy
from probe.report.render import render_report
from probe.runtime.candidate import StubCandidate
from probe.runtime.llm import FakeLLM
from probe.runtime.loop import InterviewLoop
from probe.runtime.tracing import TracedClient, TraceStore, new_run_id


def _run(rubric, bank, config, store, run_id="int-1"):
    client = FakeLLM(by_role={LLMRole.GRADE: length_proportional_grade()}, strict=False)
    traced = TracedClient(client, store=store, run_id=run_id)
    loop = InterviewLoop(
        rubric=rubric,
        bank=bank,
        policy=FixedPolicy(rubric),
        belief=PriorOnlyBelief(rubric),
        grader=LLMGrader(traced),
        candidate=StubCandidate(candidate_id="p-int"),
        config=config,
        store=store,
        run_id=run_id,
        seed=7,
        persona_id="p-int",
    )
    return loop.run(), traced


def test_five_question_fixed_interview_completes(rubric, bank, config, store):
    result, _ = _run(rubric, bank, config, store)

    assert len(result.transcript) == 5
    assert result.run.completed is True
    assert result.run.stop_reason is StopReason.BUDGET_QUESTIONS
    assert all(t.grade is not None for t in result.transcript.turns)
    assert all(t.grade.spans_valid_for(t.answer) for t in result.transcript.turns)


def test_run_is_byte_reconstructable_from_the_trace(rubric, bank, config, store):
    result, _ = _run(rubric, bank, config, store)

    rebuilt = store.load_transcript(result.run.run_id)
    assert rebuilt.render() == result.transcript.render()
    assert rebuilt.model_dump() == result.transcript.model_dump()


def test_every_llm_call_is_traced(rubric, bank, config, store):
    result, traced = _run(rubric, bank, config, store)

    counts = store.counts()
    assert counts["runs"] == 1
    assert counts["turns"] == 5
    assert counts["llm_calls"] == len(traced.calls) >= 5

    rows = store.df("SELECT role, parsed_ok, prompt_hash, prompt FROM llm_calls")
    assert set(rows["role"]) == {"grade"}
    assert rows["parsed_ok"].all()
    assert rows["prompt_hash"].map(len).eq(32).all()
    # A call is reconstructable only if the prompt is stored verbatim.
    assert all(rubric.ids[0].split(".")[0] in p or "Competency:" in p for p in rows["prompt"])


def test_belief_snapshot_persisted_every_turn(rubric, bank, config, store):
    result, _ = _run(rubric, bank, config, store)

    for turn in store.load_turns(result.run.run_id):
        assert set(turn.belief_after.means) == set(rubric.ids)
        assert set(turn.belief_after.sds) == set(rubric.ids)
        assert all(sd > 0 for sd in turn.belief_after.sds.values())


def test_fixed_policy_follows_a_deterministic_script(rubric, bank, config, store):
    a, _ = _run(rubric, bank, config, store, run_id="det-a")
    b, _ = _run(rubric, bank, config, store, run_id="det-b")

    assert [t.question_id for t in a.transcript.turns] == [
        t.question_id for t in b.transcript.turns
    ]
    # Breadth before depth: five questions on a six-competency rubric should
    # touch five distinct competencies.
    assert len({t.competency_id for t in a.transcript.turns}) == 5


def test_report_covers_every_competency(rubric, bank, config, store):
    result, _ = _run(rubric, bank, config, store)

    assert set(result.report.per_competency) == set(rubric.ids)
    text = render_report(result.report)
    assert "Interview report" in text
    for cid in rubric.ids:
        assert cid in text


def test_run_ids_are_deterministic():
    assert new_run_id("eig", "p01", "verbose", 7) == "eig.p01.verbose.s7"
    assert new_run_id("eig", "p01", "verbose", 7, "r2") == "eig.p01.verbose.s7.r2"


def test_trace_store_survives_reopen(rubric, bank, config, trace_file):
    with TraceStore(trace_file) as store:
        result, _ = _run(rubric, bank, config, store, run_id="persist-1")
        expected = result.transcript.render()

    with TraceStore(trace_file) as reopened:
        assert reopened.load_transcript("persist-1").render() == expected
        assert reopened.run_ids() == ["persist-1"]
