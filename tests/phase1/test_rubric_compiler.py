"""The rubric compiler's three commitments, tested directly.

These are not "does it run" tests. Each one corresponds to a claim the
architecture makes, and each would fail loudly if the claim stopped holding.
"""

from __future__ import annotations

import json

import pytest

from probe.jd import seniority_level
from probe.models import LLMRole
from probe.rubric.compiler import (
    HIGH_UNCERTAINTY_THRESHOLD,
    CompiledCompetency,
    CompilerOutput,
    RubricCompiler,
    prior_from_evidence,
)
from probe.rubric.taxonomy import HIGH_UNCERTAINTY_VAR, LOW_UNCERTAINTY_VAR
from probe.runtime.llm import FakeLLM


def compile_for(persona, jds, sim, taxonomy, **kwargs):
    jd = jds[persona.jd_id]
    compiler = RubricCompiler(sim, taxonomy)
    rubric = compiler.compile(
        candidate_id=persona.id,
        jd_text=jd.text,
        resume=persona.resume,
        role_title=jd.title,
        seniority_level=seniority_level(jd.seniority),
        **kwargs,
    )
    return rubric, compiler, jd


# ------------------------------------------------------- commitment 1: ids


def test_every_emitted_id_exists_in_the_taxonomy(personas, jds, sim, taxonomy):
    for persona in personas[:5]:
        rubric, compiler, _ = compile_for(persona, jds, sim, taxonomy)
        assert rubric.competencies, f"{persona.id} compiled to an empty rubric"
        for comp in rubric.competencies:
            assert taxonomy.has(comp.id)
        assert compiler.rejected_ids == []


def test_invented_ids_are_dropped_and_counted(taxonomy):
    """A model that hallucinates an id must not be able to widen the
    vocabulary — stable ids are what make the bank reusable."""
    payload = {
        "competencies": [
            {"id": "distributed_systems.consistency", "required_level": 4},
            {"id": "vibes.general_engineering_excellence", "required_level": 5},
        ]
    }
    client = FakeLLM(by_role={LLMRole.RUBRIC_COMPILE: json.dumps(payload)}, strict=False)
    compiler = RubricCompiler(client, taxonomy)
    rubric = compiler.compile(candidate_id="c", jd_text="consistency", resume="")

    assert [c.id for c in rubric.competencies] == ["distributed_systems.consistency"]
    assert compiler.rejected_ids == ["vibes.general_engineering_excellence"]


def test_taxonomy_version_is_recorded(personas, jds, sim, taxonomy):
    rubric, _, _ = compile_for(personas[0], jds, sim, taxonomy)
    assert rubric.taxonomy_version == taxonomy.version


# -------------------------------------------------- commitment 2: evidence


def test_evidence_implies_a_span_that_really_appears_in_the_resume(
    personas, jds, sim, taxonomy
):
    checked = 0
    for persona in personas:
        rubric, _, _ = compile_for(persona, jds, sim, taxonomy, max_competencies=50)
        for comp in rubric.competencies:
            if comp.evidence_in_resume > 0:
                checked += 1
                assert comp.resume_spans
                for span in comp.resume_spans:
                    assert span.verify_against(persona.resume)
                    assert span.text.strip()
    assert checked > 0, "no persona produced span-backed evidence; test is vacuous"


def test_unverifiable_spans_are_discarded_and_evidence_zeroed(taxonomy):
    """A citation the audit trail cannot check is worse than no citation."""
    payload = {
        "competencies": [
            {
                "id": "databases.indexing",
                "required_level": 4,
                "evidence_in_resume": 0.9,
                "resume_spans": [{"start": 0, "end": 20, "text": "text that is not there"}],
            }
        ]
    }
    client = FakeLLM(by_role={LLMRole.RUBRIC_COMPILE: json.dumps(payload)}, strict=False)
    rubric = RubricCompiler(client, taxonomy).compile(
        candidate_id="c", jd_text="index", resume="A resume about other things entirely."
    )
    comp = rubric.competencies[0]
    assert comp.evidence_in_resume == 0.0
    assert comp.resume_spans == []
    assert comp.prior_var > HIGH_UNCERTAINTY_THRESHOLD


# ---------------------------------------------------- commitment 3: priors


def test_resume_silence_produces_a_wide_prior(personas, jds, sim, taxonomy):
    """The gap-probing mechanism, tested at its source.

    Nothing in the policy says 'ask about gaps'. Gap-probing exists only
    because a JD-required, resume-silent competency starts with a wide prior
    and therefore looks like the most informative thing to ask about.
    """
    found_silent = 0
    for persona in personas:
        rubric, _, _ = compile_for(persona, jds, sim, taxonomy, max_competencies=50)
        for comp in rubric.competencies:
            if comp.required_level >= 3 and comp.evidence_in_resume == 0.0:
                found_silent += 1
                assert comp.prior_var > HIGH_UNCERTAINTY_THRESHOLD, (
                    f"{persona.id}/{comp.id}: required and resume-silent but "
                    f"prior_var={comp.prior_var}"
                )
    assert found_silent > 0


def test_understated_competencies_end_up_with_wide_priors(personas, jds, sim, taxonomy):
    """The deliberately-hidden strengths are exactly the cases the policy has
    to find, so they must reach the rubric looking uncertain."""
    checked = 0
    for persona in personas:
        rubric, _, _ = compile_for(persona, jds, sim, taxonomy, max_competencies=50)
        ids = {c.id: c for c in rubric.competencies}
        for cid in persona.understated:
            if cid in ids:
                checked += 1
                assert ids[cid].evidence_in_resume == 0.0
                assert ids[cid].prior_var > HIGH_UNCERTAINTY_THRESHOLD
    assert checked > 0, "no understated competency reached a rubric"


def test_prior_is_monotone_in_evidence():
    means, variances = zip(
        *[prior_from_evidence(e) for e in (0.0, 0.25, 0.5, 0.75, 1.0)], strict=True
    )
    assert list(means) == sorted(means), "more evidence must not lower the prior mean"
    assert list(variances) == sorted(variances, reverse=True), "evidence must narrow the prior"
    assert variances[0] == pytest.approx(HIGH_UNCERTAINTY_VAR)
    assert variances[-1] == pytest.approx(LOW_UNCERTAINTY_VAR)


def test_rubric_is_not_pre_sorted_by_uncertainty(personas, jds, sim, taxonomy):
    """The compiler must not do the policy's job.

    If the rubric arrived ranked by uncertainty, every arm would look adaptive
    — including the fixed script — and the comparison would be measuring the
    compiler instead of the policy.
    """
    mixed = False
    for persona in personas:
        rubric, _, _ = compile_for(persona, jds, sim, taxonomy)
        variances = [c.prior_var for c in rubric.competencies]
        if len(set(variances)) > 1:
            mixed = True
            assert variances != sorted(variances, reverse=True), (
                f"{persona.id}: rubric arrived pre-ranked by uncertainty"
            )
    assert mixed, "no rubric contained a mix of prior widths; test is vacuous"


# ------------------------------------------------------------- degradation


def test_degraded_path_widens_rather_than_guesses(taxonomy):
    """When the model cannot produce valid output the compiler must fail
    toward *more* uncertainty: ask more questions than needed, never report
    confidence it did not earn."""
    client = FakeLLM(by_role={LLMRole.RUBRIC_COMPILE: "I cannot do that."}, strict=False)
    compiler = RubricCompiler(client, taxonomy)
    rubric = compiler.compile(
        candidate_id="c",
        jd_text="We need someone strong on index design and cache invalidation.",
        resume="Built a thing with index design and cache invalidation.",
        seniority_level=4,
    )
    assert rubric.competencies
    assert all(c.evidence_in_resume == 0.0 for c in rubric.competencies)
    assert all(c.prior_var > HIGH_UNCERTAINTY_THRESHOLD for c in rubric.competencies)


def test_compiler_output_model_round_trips():
    out = CompilerOutput(
        competencies=[CompiledCompetency(id="databases.indexing", required_level=4)]
    )
    assert CompilerOutput.model_validate_json(out.model_dump_json()) == out
