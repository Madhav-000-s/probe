"""Assembling one interview.

Every path that runs an interview — the CLI, the tests, the Phase 4 sweep —
goes through :func:`build_interview`. That is deliberate: the moment two call
sites assemble a run slightly differently, the arm comparison quietly stops
being a controlled experiment.

Note the shape of the signature. A ``Persona`` goes in, and what comes back is
a loop that has only ever seen the persona's *resume*. The compilation step is
where ground truth is dropped, and it is dropped here so that no caller has to
remember to do it.
"""

from __future__ import annotations

from dataclasses import dataclass

from probe.belief.grid import GridBelief
from probe.belief.state import BeliefState
from probe.config import ExperimentConfig
from probe.grader.base import LLMGrader
from probe.jd import JobDescription, seniority_level
from probe.models import Persona, QuestionBank, Rubric
from probe.policy.registry import make_policy
from probe.rubric.compiler import RubricCompiler
from probe.rubric.taxonomy import Taxonomy, load_taxonomy
from probe.runtime.loop import InterviewLoop
from probe.runtime.tracing import TracedClient, TraceStore, new_run_id
from probe.sim.candidate import PersonaCandidate


@dataclass
class InterviewSpec:
    """Everything that identifies one run, and nothing that does not."""

    persona: Persona
    jd: JobDescription
    arm: str
    seed: int
    style_separation: bool = True
    followups_enabled: bool = True
    include_name: bool = False
    suffix: str = ""

    @property
    def run_id(self) -> str:
        return new_run_id(
            self.arm, self.persona.id, self.persona.style.id, self.seed, self.suffix
        )


def resume_claims(rubric: Rubric) -> dict[str, list[str]]:
    """What the resume asserted, per competency.

    Handed to the grader so it can raise ``resume_contradiction``. This is
    resume text — a document the interview plane is entitled to read — not
    ground truth, and the firewall test covers it either way.
    """
    return {
        c.id: [s.text for s in c.resume_spans]
        for c in rubric.competencies
        if c.evidence_in_resume > 0
    }


def build_interview(
    spec: InterviewSpec,
    *,
    bank: QuestionBank,
    config: ExperimentConfig,
    client,
    store: TraceStore | None = None,
    taxonomy: Taxonomy | None = None,
    belief_factory=None,
) -> InterviewLoop:
    taxonomy = taxonomy or load_taxonomy()
    traced = client if isinstance(client, TracedClient) else TracedClient(client, store=store)
    traced.bind_run(spec.run_id)

    # The one place ground truth is dropped: in goes the persona's resume,
    # out comes a rubric, and the loop below never sees the Persona object.
    rubric = RubricCompiler(traced, taxonomy).compile(
        candidate_id=spec.persona.id,
        jd_text=spec.jd.text,
        resume=spec.persona.resume,
        role_title=spec.jd.title,
        seniority_level=seniority_level(spec.jd.seniority),
        seed=spec.seed,
        # Only compile competencies the bank can actually probe. Quarantined
        # items are already excluded from `live()`, so a competency whose only
        # items were quarantined by calibration correctly drops out here too.
        available_competencies={q.competency_id for q in bank.live()},
    )

    # Every arm shares the same estimator. Only *selection* differs between
    # arms; the posterior that produces the report and the recovery number is
    # identical machinery in all four. Giving the belief-free arms a weaker
    # estimator would make them lose on inference rather than on question
    # choice, which is not the experiment.
    belief: BeliefState = (belief_factory or GridBelief)(rubric)

    return InterviewLoop(
        rubric=rubric,
        bank=bank,
        policy=make_policy(spec.arm, rubric, config, traced, seed=spec.seed),
        belief=belief,
        grader=LLMGrader(
            traced,
            style_separation=spec.style_separation,
            resume_claims=resume_claims(rubric),
        ),
        candidate=PersonaCandidate(
            spec.persona,
            traced,
            taxonomy=taxonomy,
            seed=spec.seed,
            include_name=spec.include_name,
        ),
        config=config,
        store=store,
        run_id=spec.run_id,
        seed=spec.seed,
        persona_id=spec.persona.id,
        style_id=spec.persona.style.id,
        style_separation=spec.style_separation,
        followups_enabled=spec.followups_enabled,
        grader_model=getattr(traced, "model", "unknown"),
    )
