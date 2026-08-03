"""End-to-end on the sim backend: a real persona, a real compiled rubric, a
real grader, and the firewall still holding.

The ordering check is the Phase 1 exit criterion the plan words as "by
eyeball". Written as an assertion instead, because an eyeball does not run in
CI and does not fail when somebody breaks the channel six weeks later.
"""

from __future__ import annotations

import json

import pytest
from scipy import stats

from probe.config import Budgets, ExperimentConfig
from probe.models import INTERVIEW_PLANE_ROLES, LLMRole
from probe.runtime.session import InterviewSpec, build_interview
from probe.runtime.tracing import TracedClient


def _run(persona, jds, starter, sim, store, questions=12, seed=20260803):
    config = ExperimentConfig(
        budgets=Budgets(max_questions=questions, max_tokens=10**9, max_wallclock_seconds=1e9),
        seed_set=[seed],
        bank_version=starter.version,
        population_version="v1-test",
    )
    traced = TracedClient(sim, store=store)
    spec = InterviewSpec(persona=persona, jd=jds[persona.jd_id], arm="fixed", seed=seed)
    loop = build_interview(spec, bank=starter, config=config, client=traced, store=store)
    return loop.run(), loop


def test_interview_completes_and_grades_every_turn(personas, jds, starter, sim, store):
    result, _loop = _run(personas[7], jds, starter, sim, store)

    assert len(result.transcript) == 12
    assert result.run.completed
    graded = [t for t in result.transcript.turns if t.grade is not None]
    assert len(graded) >= 11, "at most one turn should be unrecoverable at this violation rate"
    for turn in graded:
        assert turn.grade.spans_valid_for(turn.answer)


def test_starter_bank_covers_both_role_families(personas, jds, starter, sim, store):
    """A bank that does not cover the roles being hired for is an instrument
    that cannot measure its own subjects.

    This started as a silent failure: data/ML personas compiled a rubric the
    all-backend bank had no items for and terminated at turn zero with a
    perfectly valid-looking report.
    """
    covered = {q.competency_id for q in starter.questions}
    for jd in jds.values():
        missing = set(jd.required) - covered
        assert not missing, f"{jd.id} requires uncovered competencies: {sorted(missing)}"

    # Every persona must be able to spend its whole budget. An arm that stops
    # early because the script ran out has not reached confidence sooner; it
    # has been handed a shorter interview, and comparing that to another arm
    # would be comparing interview lengths rather than policies.
    for persona in personas:
        result, _loop = _run(persona, jds, starter, sim, store, questions=12)
        assert len(result.transcript) == 12, (
            f"{persona.id} ({persona.jd_id}) ran short: "
            f"{len(result.transcript)} turns, stop={result.run.stop_reason.value}"
        )


def test_report_orders_competencies_the_way_ability_does(personas, jds, starter, sim, store):
    """The Phase 1 sanity check: within one interview, competencies the
    persona is genuinely stronger at should grade higher.

    Pooled across several personas because a single 12-question interview is a
    handful of noisy observations, and asserting a clean ordering from that
    would be asserting the noise away.
    """
    thetas: list[float] = []
    scores: list[float] = []

    for persona in personas[:6]:
        result, _loop = _run(persona, jds, starter, sim, store)
        by_competency: dict[str, list[int]] = {}
        for turn in result.transcript.turns:
            if turn.grade is not None:
                by_competency.setdefault(turn.competency_id, []).append(turn.grade.score)
        for cid, got in by_competency.items():
            thetas.append(persona.ability(cid))
            scores.append(sum(got) / len(got))

    assert len(thetas) > 25
    rho, p = stats.spearmanr(thetas, scores)
    assert rho > 0.35, f"grades barely track ability end to end: rho={rho:.3f}, p={p:.3g}"


def test_ground_truth_never_reaches_an_interview_plane_prompt(
    personas, jds, starter, sim, store
):
    """The Phase 0 firewall, re-pointed at the real simulator.

    Phase 0 ran this against a stub candidate. Now there is a genuine persona
    with genuine hidden abilities flowing through a genuine LLM call, which is
    the configuration the guarantee actually has to hold in.
    """
    persona = personas[8]
    _run(persona, jds, starter, sim, store)

    prompts = store.all_prompts()
    interview_prompts = [
        (role, text) for role, text in prompts if LLMRole(role) in INTERVIEW_PLANE_ROLES
    ]
    assert interview_prompts, "no interview-plane prompts logged; test is vacuous"

    for role, prompt in interview_prompts:
        lowered = prompt.lower()
        assert "theta" not in lowered
        assert "ground_truth" not in lowered
        for value in persona.theta_star.values():
            for form in (repr(value), f"{value:.6f}", f"{value:.3f}", f"{value:.2f}"):
                assert form not in prompt, f"{form!r} leaked into a {role} prompt"


def test_measurement_plane_prompts_are_logged_separately(personas, jds, starter, sim, store):
    """Persona answering is a measurement-plane role and is allowed ability.
    It must still be traced, and it must still not print the number."""
    _run(personas[8], jds, starter, sim, store)

    roles = {role for role, _ in store.all_prompts()}
    assert LLMRole.PERSONA_ANSWER.value in roles
    assert LLMRole.GRADE.value in roles
    assert LLMRole.RUBRIC_COMPILE.value in roles

    persona_prompts = [t for r, t in store.all_prompts() if r == LLMRole.PERSONA_ANSWER.value]
    for prompt in persona_prompts:
        assert "theta" not in prompt.lower(), (
            "even the persona prompt should express ability in words, so a live "
            "backend never receives the latent value"
        )


def test_run_is_reconstructable_from_the_trace(personas, jds, starter, sim, store):
    result, _loop = _run(personas[3], jds, starter, sim, store, questions=6)
    rebuilt = store.load_transcript(result.run.run_id)
    assert rebuilt.render() == result.transcript.render()


def test_repair_ladder_is_exercised_by_the_real_backend(personas, jds, starter, sim, store):
    """SimLLM emits structurally invalid output at a configured rate so the
    schema-violation and repair-success numbers are measured rather than
    assumed. Confirm the ladder is genuinely being walked."""
    for persona in personas[:5]:
        _run(persona, jds, starter, sim, store, questions=10)

    rows = store.df(
        "SELECT repair_attempt, parsed_ok, count(*) AS n FROM llm_calls "
        "WHERE role = 'grade' GROUP BY repair_attempt, parsed_ok"
    )
    assert not rows.empty
    assert (rows["repair_attempt"] > 0).any(), "no repair attempt was ever needed"


def test_persona_candidate_keeps_its_audit_trail(personas, jds, starter, sim, store):
    """The measurement plane needs to distinguish 'did not say it' from 'grader
    did not see it'. That requires the drawn level, which the loop must never
    receive."""
    _result, loop = _run(personas[7], jds, starter, sim, store, questions=5)
    records = loop.candidate.records

    assert len(records) == 5
    for record in records:
        assert 1 <= record.drawn_level <= 5
        assert record.text

    turn_json = json.dumps(
        [t.model_dump(mode="json") for t in _result.transcript.turns], default=str
    )
    assert "drawn_level" not in turn_json


@pytest.mark.slow
def test_stronger_personas_score_higher_than_weaker_ones(personas, jds, starter, sim, store):
    """A blunt aggregate check that survives per-turn noise.

    Ability is averaged over the competencies the interview actually *probed*,
    not over all fifty. Averaging over the whole taxonomy would score a persona
    on competencies nobody asked about, which is a different question and a
    noisier one.
    """
    means = []
    for persona in personas:
        result, _loop = _run(persona, jds, starter, sim, store, questions=10)
        graded = [t for t in result.transcript.turns if t.grade]
        assert graded, f"{persona.id} was never asked anything"
        probed = {t.competency_id for t in graded}
        ability = sum(persona.ability(c) for c in probed) / len(probed)
        means.append((ability, sum(t.grade.score for t in graded) / len(graded)))

    rho, _p = stats.spearmanr([m[0] for m in means], [m[1] for m in means])
    assert rho > 0.5, f"probed ability does not track mean grade: rho={rho:.3f}"
