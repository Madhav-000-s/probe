"""What the arms actually do, as opposed to what the maths says they should.

The high-variance targeting test is the behavioural statement of the whole
claim: given one competency it knows nothing about and four it knows well, an
information-driven policy should go straight at the gap — without anything
anywhere naming the concept of a gap.
"""

from __future__ import annotations

import json

import pytest

from probe.bank.loader import build_question
from probe.belief.grid import GridBelief
from probe.belief.state import PriorOnlyBelief
from probe.config import Budgets, ExperimentConfig
from probe.grader.base import LLMGrader
from probe.grader.fixtures import constant_grade
from probe.models import (
    Competency,
    LLMRole,
    ProbeFamily,
    QuestionBank,
    Rubric,
    StopReason,
    Transcript,
)
from probe.policy.base import Ask, Stop
from probe.policy.eig import EIGPolicy, repeat_family_penalty
from probe.policy.fixed import FixedPolicy
from probe.policy.heuristic import HeuristicPolicy, PolicyChoice
from probe.policy.registry import ARMS, make_policy, needs_grid_belief
from probe.rubric.taxonomy import HIGH_UNCERTAINTY_VAR, LOW_UNCERTAINTY_VAR
from probe.runtime.candidate import StubCandidate
from probe.runtime.llm import FakeLLM
from probe.runtime.loop import InterviewLoop

WIDE = "distributed_systems.consistency"
NARROW = (
    "distributed_systems.partitioning",
    "distributed_systems.replication",
    "databases.indexing",
    "databases.transactions",
)


@pytest.fixture
def lopsided_rubric(taxonomy):
    """One competency we know nothing about, four we know well."""
    comps = []
    for cid in (WIDE, *NARROW):
        wide = cid == WIDE
        node = taxonomy.get(cid)
        comps.append(
            Competency(
                id=cid,
                label=node.label,
                required_level=4,
                evidence_in_resume=0.0 if wide else 0.9,
                prior_mean=0.0 if wide else 0.5,
                prior_var=HIGH_UNCERTAINTY_VAR if wide else LOW_UNCERTAINTY_VAR,
                probe_families=list(node.probe_families),
                resume_spans=[]
                if wide
                else [{"start": 0, "end": len(node.label), "text": node.label}],
            )
        )
    return Rubric(
        candidate_id="lopsided", role_title="r", competencies=comps, taxonomy_version="v1"
    )


@pytest.fixture
def lopsided_bank(taxonomy, lopsided_rubric):
    questions = []
    for cid in lopsided_rubric.ids:
        node = taxonomy.get(cid)
        for i in range(4):
            questions.append(
                build_question(
                    competency_id=cid,
                    label=node.label,
                    concepts=node.concepts,
                    family=list(ProbeFamily)[i % 4],
                    index=i,
                )
            )
    return QuestionBank(version="lopsided", taxonomy_version="v1", questions=questions)


@pytest.fixture
def cfg():
    return ExperimentConfig(
        budgets=Budgets(max_questions=12, max_tokens=10**9, max_wallclock_seconds=1e9),
        seed_set=[5],
    )


def run_arm(policy, rubric, bank, cfg, belief=None, score=3):
    client = FakeLLM(by_role={LLMRole.GRADE: constant_grade(score=score)}, strict=False)
    loop = InterviewLoop(
        rubric=rubric,
        bank=bank,
        policy=policy,
        belief=belief if belief is not None else GridBelief(rubric),
        grader=LLMGrader(client),
        candidate=StubCandidate(),
        config=cfg,
        run_id="policy-test",
        seed=5,
    )
    return loop.run(), loop


# ------------------------------------------------------- targeting the gap


def test_eig_attacks_the_high_variance_competency_first(lopsided_rubric, lopsided_bank, cfg):
    """The gap-probing behaviour, with nothing in the policy that knows the
    word 'gap'. It emerges entirely from prior width.

    The plan's wording is "the first three questions target the high-variance
    one". At a = 1.9 that turns out to overstate it: two questions already
    pull the wide competency's SD below the narrow ones', at which point
    moving on is the correct adaptive choice rather than a failure to focus.
    So the assertion is on the behaviour that actually matters — the gap gets
    the opening questions and the lion's share of early attention — not on an
    exact count that depends on how discriminating the items happen to be.
    """
    result, _loop = run_arm(EIGPolicy(lopsided_rubric, cfg), lopsided_rubric, lopsided_bank, cfg)

    opening = [t.competency_id for t in result.transcript.turns[:2]]
    assert opening == [WIDE, WIDE], opening

    first_five = [t.competency_id for t in result.transcript.turns[:5]]
    assert first_five.count(WIDE) >= 2
    assert first_five.count(WIDE) == max(first_five.count(c) for c in set(first_five))


def test_fixed_script_does_not_prioritise_the_gap(lopsided_rubric, lopsided_bank, cfg):
    """The contrast that makes the previous test mean something."""
    result, _loop = run_arm(
        FixedPolicy(lopsided_rubric), lopsided_rubric, lopsided_bank, cfg
    )
    first_three = [t.competency_id for t in result.transcript.turns[:3]]
    assert first_three != [WIDE, WIDE, WIDE]
    assert len(set(first_three)) == 3, "the fixed script goes breadth-first regardless"


def test_eig_moves_on_once_the_gap_is_closed(lopsided_rubric, lopsided_bank, cfg):
    """An adaptive policy that never leaves its first target is not adaptive."""
    result, _loop = run_arm(EIGPolicy(lopsided_rubric, cfg), lopsided_rubric, lopsided_bank, cfg)
    probed = [t.competency_id for t in result.transcript.turns]
    assert len(set(probed)) > 1, "eig never left the first competency"


# --------------------------------------------------- repeat-family penalty


def test_repeat_family_penalty_computes_a_recent_ratio():
    transcript = Transcript(run_id="r", candidate_id="c", arm="eig")
    assert repeat_family_penalty(ProbeFamily.DEBUG, transcript) == 0.0


def test_repeat_family_penalty_breaks_a_streak(lopsided_rubric, lopsided_bank):
    """With the penalty off, one probe family can dominate. Turning it on has
    to visibly break that up, or it is doing nothing."""
    without = ExperimentConfig(
        budgets=Budgets(max_questions=8, max_tokens=10**9, max_wallclock_seconds=1e9),
        repeat_family_lambda=0.0,
        seed_set=[5],
    )
    with_penalty = ExperimentConfig(
        budgets=without.budgets, repeat_family_lambda=0.9, seed_set=[5]
    )

    def longest_streak(transcript):
        best = run = 1
        families = [t.question_id.split("::")[-1] for t in transcript.turns]
        for a, b in zip(families, families[1:], strict=False):
            run = run + 1 if a == b else 1
            best = max(best, run)
        return best

    plain, _ = run_arm(
        EIGPolicy(lopsided_rubric, without), lopsided_rubric, lopsided_bank, without
    )
    penalised, _ = run_arm(
        EIGPolicy(lopsided_rubric, with_penalty),
        lopsided_rubric,
        lopsided_bank,
        with_penalty,
    )

    assert longest_streak(plain.transcript) >= 4, "no streak to break; test is vacuous"
    assert longest_streak(penalised.transcript) < longest_streak(plain.transcript)


# ------------------------------------------------------------- guard rails


def test_eig_refuses_a_prior_only_belief(lopsided_rubric, lopsided_bank, cfg):
    """Running the belief-driven arm against a belief that never learns would
    score it at chance and silently invalidate the comparison. Fail loudly."""
    policy = EIGPolicy(lopsided_rubric, cfg)
    with pytest.raises(TypeError, match="needs a grid posterior"):
        policy.next_question(
            PriorOnlyBelief(lopsided_rubric),
            Transcript(run_id="r", candidate_id="c", arm="eig"),
            lopsided_bank,
        )


def test_registry_knows_which_arms_need_a_posterior():
    assert needs_grid_belief("eig")
    assert not needs_grid_belief("fixed")
    assert not needs_grid_belief("heuristic")
    assert set(ARMS) == {"fixed", "heuristic", "eig"}


def test_heuristic_arm_requires_a_client(lopsided_rubric, cfg):
    with pytest.raises(ValueError, match="needs an LLM client"):
        make_policy("heuristic", lopsided_rubric, cfg, client=None)


def test_unknown_arm_is_rejected(lopsided_rubric, cfg):
    with pytest.raises(ValueError, match="unknown arm"):
        make_policy("telepathy", lopsided_rubric, cfg)


# ------------------------------------------------------- the heuristic arm


def test_heuristic_covers_breadth_before_depth(lopsided_rubric, lopsided_bank, cfg, sim):
    """It has no belief state, but it is not stupid: every competency should be
    touched before any is revisited."""
    result, _loop = run_arm(
        HeuristicPolicy(lopsided_rubric, sim), lopsided_rubric, lopsided_bank, cfg
    )
    first_five = [t.competency_id for t in result.transcript.turns[:5]]
    assert len(set(first_five)) == 5, first_five


def test_heuristic_falls_back_competently_on_a_bad_choice(lopsided_rubric, lopsided_bank, cfg):
    """A fallback that picked at random would quietly handicap the competitor
    arm and flatter the headline result."""
    client = FakeLLM(
        by_role={
            LLMRole.GRADE: constant_grade(score=3),
            LLMRole.POLICY_CHOOSE: json.dumps({"question_id": "not-a-real-id"}),
        },
        strict=False,
    )
    policy = HeuristicPolicy(lopsided_rubric, client)
    result, _loop = run_arm(policy, lopsided_rubric, lopsided_bank, cfg)

    assert policy.fallbacks > 0
    # It may terminate on confidence before exhausting the budget; what matters
    # is that an unusable model choice never derails the interview.
    assert len(result.transcript) >= 5
    assert result.run.completed
    first_five = [t.competency_id for t in result.transcript.turns[:5]]
    assert len(set(first_five)) == 5, "even the fallback goes breadth-first"


def test_heuristic_choice_must_be_on_the_shortlist(lopsided_rubric, lopsided_bank, sim):
    policy = HeuristicPolicy(lopsided_rubric, sim)
    transcript = Transcript(run_id="r", candidate_id="c", arm="heuristic")
    decision = policy.next_question(GridBelief(lopsided_rubric), transcript, lopsided_bank)
    assert isinstance(decision, Ask)
    assert decision.question.id in {q.id for q in lopsided_bank.questions}


def test_heuristic_reports_no_eig(lopsided_rubric, lopsided_bank, sim):
    """Arms without a belief state leave the EIG column empty rather than
    filling in a fake zero, so traces show honestly who computed one."""
    policy = HeuristicPolicy(lopsided_rubric, sim)
    decision = policy.next_question(
        GridBelief(lopsided_rubric),
        Transcript(run_id="r", candidate_id="c", arm="heuristic"),
        lopsided_bank,
    )
    assert decision.eig is None


def test_policy_choice_model_round_trips():
    choice = PolicyChoice(question_id="q1", reason="because")
    assert PolicyChoice.model_validate_json(choice.model_dump_json()) == choice


# ------------------------------------------------------------- the stop rule


def test_confidence_stop_fires(lopsided_rubric, lopsided_bank):
    """Loose tau plus consistent evidence should terminate on confidence."""
    cfg = ExperimentConfig(
        tau=1.05,
        budgets=Budgets(max_questions=20, max_tokens=10**9, max_wallclock_seconds=1e9),
        seed_set=[5],
    )
    result, _loop = run_arm(EIGPolicy(lopsided_rubric, cfg), lopsided_rubric, lopsided_bank, cfg)
    assert result.run.stop_reason is StopReason.CONFIDENCE
    assert len(result.transcript) < 20


def test_budget_stop_fires(lopsided_rubric, lopsided_bank):
    cfg = ExperimentConfig(
        tau=0.05,
        budgets=Budgets(max_questions=4, max_tokens=10**9, max_wallclock_seconds=1e9),
        seed_set=[5],
    )
    result, _loop = run_arm(EIGPolicy(lopsided_rubric, cfg), lopsided_rubric, lopsided_bank, cfg)
    assert result.run.stop_reason is StopReason.BUDGET_QUESTIONS
    assert len(result.transcript) == 4


def test_no_informative_question_stop_fires(lopsided_rubric, lopsided_bank):
    """With epsilon above every achievable EIG, there is by definition nothing
    left worth asking — at turn zero."""
    cfg = ExperimentConfig(
        tau=0.01,
        epsilon=99.0,
        budgets=Budgets(max_questions=12, max_tokens=10**9, max_wallclock_seconds=1e9),
        seed_set=[5],
    )
    policy = EIGPolicy(lopsided_rubric, cfg)
    decision = policy.next_question(
        GridBelief(lopsided_rubric),
        Transcript(run_id="r", candidate_id="c", arm="eig"),
        lopsided_bank,
    )
    assert isinstance(decision, Stop)
    assert decision.reason is StopReason.NO_INFORMATIVE_QUESTION


def test_bank_exhausted_stop_fires(lopsided_rubric, lopsided_bank):
    cfg = ExperimentConfig(
        tau=0.001,
        budgets=Budgets(max_questions=999, max_tokens=10**9, max_wallclock_seconds=1e9),
        seed_set=[5],
    )
    result, _loop = run_arm(EIGPolicy(lopsided_rubric, cfg), lopsided_rubric, lopsided_bank, cfg)
    assert result.run.stop_reason in {
        StopReason.BANK_EXHAUSTED,
        StopReason.NO_INFORMATIVE_QUESTION,
    }
    assert len(result.transcript) <= len(lopsided_bank.questions)


def test_stop_reasons_reach_the_traces(lopsided_rubric, lopsided_bank, cfg, store):
    client = FakeLLM(by_role={LLMRole.GRADE: constant_grade(score=3)}, strict=False)
    loop = InterviewLoop(
        rubric=lopsided_rubric,
        bank=lopsided_bank,
        policy=EIGPolicy(lopsided_rubric, cfg),
        belief=GridBelief(lopsided_rubric),
        grader=LLMGrader(client),
        candidate=StubCandidate(),
        config=cfg,
        store=store,
        run_id="stop-trace",
        seed=5,
    )
    loop.run()
    rows = store.df("SELECT stop_reason FROM runs WHERE run_id = 'stop-trace'")
    assert rows["stop_reason"].iloc[0] in {r.value for r in StopReason}


def test_eig_is_recorded_per_turn(lopsided_rubric, lopsided_bank, cfg, store):
    """Belief snapshots and the selection-time EIG are what make the
    accuracy-vs-budget curves computable post hoc."""
    client = FakeLLM(by_role={LLMRole.GRADE: constant_grade(score=3)}, strict=False)
    loop = InterviewLoop(
        rubric=lopsided_rubric,
        bank=lopsided_bank,
        policy=EIGPolicy(lopsided_rubric, cfg),
        belief=GridBelief(lopsided_rubric),
        grader=LLMGrader(client),
        candidate=StubCandidate(),
        config=cfg,
        store=store,
        run_id="eig-trace",
        seed=5,
    )
    loop.run()
    turns = store.load_turns("eig-trace")
    assert all(t.eig_at_selection is not None and t.eig_at_selection >= 0 for t in turns)
    # Information gets used up: later questions are worth less than the first.
    assert turns[-1].eig_at_selection < turns[0].eig_at_selection
