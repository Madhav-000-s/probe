"""The ground-truth firewall.

``theta_star`` may appear in ``data/personas/`` and in the eval harness. It may
never appear in anything the interview plane sends to a model. A leak here
does not break a test or crash a run — it silently makes every recovery number
in the project meaningless, which is exactly why the check is automated and
why it never leaves the suite.

This test grows with the project: Phase 0 wires it to the stub candidate, and
later phases point it at the real persona simulator without changing what it
asserts.
"""

from __future__ import annotations

import json

import pytest

from probe.belief.state import PriorOnlyBelief
from probe.grader.base import LLMGrader
from probe.grader.fixtures import constant_grade
from probe.models import (
    INTERVIEW_PLANE_ROLES,
    LLMRole,
    Persona,
    Question,
    StyleProfile,
    Transcript,
)
from probe.policy.fixed import FixedPolicy
from probe.runtime.candidate import AnswerResult, AnswerSource
from probe.runtime.llm import FakeLLM
from probe.runtime.loop import InterviewLoop
from probe.runtime.tracing import TracedClient

#: Deliberately weird values. If any of these substrings shows up in a prompt
#: it is because ground truth leaked, not because a float coincided.
THETA = {
    "distributed_systems.consistency": 1.234567,
    "distributed_systems.partitioning": -0.876543,
    "distributed_systems.replication": 2.109876,
    "distributed_systems.consensus": -1.345678,
    "distributed_systems.failure_modes": 0.567891,
    "databases.indexing": -2.098765,
}


@pytest.fixture
def persona():
    return Persona(
        id="p-firewall",
        theta_star=dict(THETA),
        style=StyleProfile(
            id="neutral", verbosity=1.0, hedging=0.1, assertiveness=0.1, l1_transfer=0.0
        ),
        behavior="honest",
        resume="Worked on distributed systems and databases.",
        jd_id="jd-backend",
        split="eval",
    )


class PersonaBackedCandidate(AnswerSource):
    """Holds a persona and emits only prose.

    This is the boundary in miniature: ability shapes *what the answer says*,
    and nothing downstream ever receives the number itself.
    """

    def __init__(self, persona: Persona) -> None:
        self._persona = persona
        self.id = persona.id
        self.style_id = persona.style.id

    def answer(self, question: Question, transcript: Transcript) -> AnswerResult:
        theta = self._persona.ability(question.competency_id)
        pool = question.anchor(5).required_concepts
        n = 1 if theta < -0.5 else (3 if theta < 1.0 else len(pool))
        named = ", ".join(pool[:n])
        return AnswerResult(text=f"The key considerations here are {named}.", seconds=60.0)


def _leak_forms(value: float) -> list[str]:
    """Every plausible rendering of a float, because a leak via ``f"{x:.2f}"``
    is still a leak."""
    return [
        repr(value),
        str(value),
        f"{value:.6f}",
        f"{value:.4f}",
        f"{value:.3f}",
        f"{value:.2f}",
        f"{value:+.2f}",
    ]


def _run_interview(persona, rubric, bank, config, store):
    client = FakeLLM(by_role={LLMRole.GRADE: constant_grade(score=3)}, strict=False)
    traced = TracedClient(client, store=store, run_id="firewall-1")
    loop = InterviewLoop(
        rubric=rubric,
        bank=bank,
        policy=FixedPolicy(rubric),
        belief=PriorOnlyBelief(rubric),
        grader=LLMGrader(traced),
        candidate=PersonaBackedCandidate(persona),
        config=config,
        store=store,
        run_id="firewall-1",
        seed=7,
        persona_id=persona.id,
    )
    return loop.run(), client


def test_no_theta_value_appears_in_any_logged_prompt(persona, taxonomy, config, store):
    rubric = taxonomy.stub_rubric(candidate_id=persona.id, n=6)
    from probe.bank.loader import stub_bank

    bank = stub_bank(taxonomy, competency_ids=rubric.ids, per_competency=2)

    _run_interview(persona, rubric, bank, config, store)

    prompts = store.all_prompts()
    assert prompts, "the test is worthless if nothing was logged"

    for role, prompt in prompts:
        assert LLMRole(role) in INTERVIEW_PLANE_ROLES
        for cid, value in persona.theta_star.items():
            for form in _leak_forms(value):
                assert form not in prompt, f"{cid} leaked as {form!r} into a {role} prompt"


def test_no_ground_truth_vocabulary_in_prompts(persona, taxonomy, config, store):
    """Catches the subtler leak: not the number, but a field name that implies
    the interview plane was handed the persona object."""
    rubric = taxonomy.stub_rubric(candidate_id=persona.id, n=6)
    from probe.bank.loader import stub_bank

    bank = stub_bank(taxonomy, competency_ids=rubric.ids, per_competency=2)
    _run_interview(persona, rubric, bank, config, store)

    banned = ("theta_star", "theta*", "ground_truth", "true_ability", "hidden_ability")
    for _role, prompt in store.all_prompts():
        lowered = prompt.lower()
        for token in banned:
            assert token not in lowered, f"{token!r} appears in an interview-plane prompt"


def test_request_context_is_also_clean(persona, taxonomy, config, store):
    """The sim backend reads ``LLMRequest.context`` instead of re-parsing the
    prompt, so the firewall has to cover it too — otherwise the leak just
    moves one field to the left."""
    rubric = taxonomy.stub_rubric(candidate_id=persona.id, n=6)
    from probe.bank.loader import stub_bank

    bank = stub_bank(taxonomy, competency_ids=rubric.ids, per_competency=2)
    _, client = _run_interview(persona, rubric, bank, config, store)

    for call in client.calls:
        if call.role not in INTERVIEW_PLANE_ROLES:
            continue
        blob = json.dumps(call.context, default=str)
        for value in persona.theta_star.values():
            for form in _leak_forms(value):
                assert form not in blob
        assert "theta" not in blob.lower()


def test_persona_public_view_exposes_nothing_hidden(persona):
    public = persona.public_view()
    assert set(public) == {"id", "resume", "jd_id"}
    assert "theta_star" not in json.dumps(public)


def test_the_firewall_test_can_actually_fail(persona, taxonomy, config, store):
    """A guard that never fails is decoration. Prove the detector works by
    handing it a prompt that really does contain ground truth."""
    poisoned = f"Candidate ability on consistency is {persona.theta_star['distributed_systems.consistency']}"
    leaked = any(
        form in poisoned
        for form in _leak_forms(persona.theta_star["distributed_systems.consistency"])
    )
    assert leaked, "the leak detector failed to detect a planted leak"
