"""The persona simulator.

The load-bearing invariant is the content/style split: two style variants of
the same answer must contain the same concepts, or "style drift" is confounded
with a genuine content difference and Q3 becomes unanswerable. It is tested
here directly, and separately for the one case where it is deliberately
violated (first-language paraphrase).
"""

from __future__ import annotations

import random

import pytest

from probe.models import Behavior
from probe.sim.answers import answer_seed, compose_answer, draw_level
from probe.sim.behavior import INJECTION_PAYLOADS, is_adversarial, plan_content
from probe.sim.style import (
    MAIN_SWEEP_STYLES,
    STYLE_PRESETS,
    SegmentKind,
    is_terse,
    paraphrase_concept,
)
from probe.sim.textfeatures import match_concepts, style_features

SEED = 4242


def _compose(question, theta, behavior=Behavior.HONEST, style_id="neutral", seed=SEED):
    return compose_answer(
        question=question,
        theta=theta,
        behavior=behavior,
        style=STYLE_PRESETS[style_id],
        distractor_pool=["hash partitioning", "consistent hashing", "virtual nodes"],
        seed=seed,
    )


# --------------------------------------------------------- ability -> text


def test_concepts_named_appear_verbatim_in_the_answer(starter):
    """The channel has to actually exist: if the concepts a persona 'named'
    are not in the text, the grader cannot recover anything."""
    question = starter.questions[0]
    for theta in (-2.0, -0.5, 0.5, 2.0):
        text, level, plan = _compose(question, theta)
        matched = match_concepts(text, plan.concepts)
        assert len(matched) == len(plan.concepts), (
            f"theta={theta}: planned {plan.concepts}, found {[m.concept for m in matched]}"
        )


@pytest.mark.slow
def test_higher_ability_names_more_concepts_on_average(starter):
    question = starter.questions[0]
    means = []
    for theta in (-2.0, -1.0, 0.0, 1.0, 2.0):
        counts = [
            _compose(question, theta, seed=SEED + i)[2].n_concepts for i in range(300)
        ]
        means.append(sum(counts) / len(counts))
    assert means == sorted(means), f"concept count is not monotone in ability: {means}"
    assert means[-1] - means[0] > 2.0, "ability barely moves content; channel is too weak"


def test_composition_is_deterministic(starter):
    question = starter.questions[0]
    a = _compose(question, 0.7)
    b = _compose(question, 0.7)
    assert a == b


def test_answer_seed_is_stable_and_independent_of_style():
    """Content is a function of (persona, question, seed) and nothing else.

    Style used to be in this hash, and it quietly broke the fairness design:
    each style variant of a persona drew a *different* response level, so the
    terse and verbose variants were answering with different content and the
    measured "style drift" was mostly content variance. Style renders content;
    it must never change it.
    """
    a = answer_seed("p001", "q1", "neutral", 7)
    assert a == answer_seed("p001", "q1", "neutral", 7)
    assert a == answer_seed("p001", "q1", "terse", 7), "style changed the content draw"
    assert a == answer_seed("p001", "q1", "name_b", 7)
    assert a != answer_seed("p002", "q1", "neutral", 7)
    assert a != answer_seed("p001", "q2", "neutral", 7)
    assert a != answer_seed("p001", "q1", "neutral", 8)


def test_drawn_level_respects_the_response_model(starter):
    """Level is drawn from the GRM, not assigned. A strong candidate should
    almost never bottom out and a weak one should almost never top out."""
    question = starter.questions[0]
    rng = random.Random(1)
    strong = [draw_level(2.5, question, rng) for _ in range(400)]
    weak = [draw_level(-2.5, question, rng) for _ in range(400)]
    assert sum(strong) / len(strong) > sum(weak) / len(weak) + 2.0
    assert max(weak) <= 5 and min(strong) >= 1


# ------------------------------------------------- the content/style split


def test_style_variants_carry_identical_content(starter):
    """The invariant the fairness suite rests on.

    ``l1_transfer`` is excluded because it paraphrases concepts on purpose —
    that exception is tested separately below.
    """
    question = starter.questions[0]
    styles = [s for s in MAIN_SWEEP_STYLES if s != "l1_transfer"]
    baseline = _compose(question, 1.2, style_id=styles[0])[2].concepts

    for style_id in styles[1:]:
        text, _level, plan = _compose(question, 1.2, style_id=style_id)
        assert plan.concepts == baseline, f"{style_id} changed the content plan"
        found = {m.concept for m in match_concepts(text, baseline)}
        assert found == set(baseline), f"{style_id} lost concepts: {set(baseline) - found}"


def test_terseness_removes_padding_not_substance(starter):
    """This is the bug the Phase 1 fidelity gate caught. It stays tested."""
    question = starter.questions[0]
    verbose_text, _, verbose_plan = _compose(question, 1.5, style_id="verbose")
    terse_text, _, terse_plan = _compose(question, 1.5, style_id="terse")

    assert terse_plan.concepts == verbose_plan.concepts
    assert len(match_concepts(terse_text, terse_plan.concepts)) == len(terse_plan.concepts)
    assert len(terse_text.split()) < len(verbose_text.split())
    assert is_terse(STYLE_PRESETS["terse"])


def test_style_moves_surface_features_without_moving_content(starter):
    question = starter.questions[0]
    verbose = style_features(_compose(question, 1.0, style_id="verbose")[0])
    terse = style_features(_compose(question, 1.0, style_id="terse")[0])
    hedged = style_features(_compose(question, 1.0, style_id="hedged")[0])
    assertive = style_features(_compose(question, 1.0, style_id="assertive")[0])

    assert verbose.words > terse.words
    assert hedged.hedge_density > assertive.hedge_density
    assert assertive.assertive_density > hedged.assertive_density


def test_l1_transfer_paraphrases_concepts_on_purpose(starter):
    """The documented exception, and the mechanism behind the residual drift
    the content-style intervention cannot remove."""
    question = starter.questions[0]
    lost = 0
    for i in range(40):
        text, _level, plan = _compose(question, 1.8, style_id="l1_transfer", seed=SEED + i)
        if plan.concepts:
            found = len(match_concepts(text, plan.concepts))
            lost += len(plan.concepts) - found
    assert lost > 0, "L1 paraphrase never defeated exact matching; the residual bug is gone"


def test_paraphrase_changes_the_surface_form():
    rng = random.Random(0)
    for phrase in ("read-your-writes", "index selectivity", "quorum reads"):
        variants = {paraphrase_concept(phrase, rng) for _ in range(20)}
        assert any(v != phrase for v in variants)


# ------------------------------------------------------------- behaviours


def test_terse_behaviour_is_short_but_still_correct(starter):
    question = starter.questions[0]
    honest = _compose(question, 1.8, behavior=Behavior.HONEST)
    terse = _compose(question, 1.8, behavior=Behavior.TERSE)
    assert terse[2].concepts == honest[2].concepts
    assert len(terse[0].split()) < len(honest[0].split())


def test_dodger_says_less_than_it_knows(starter):
    question = starter.questions[0]
    honest = _compose(question, 2.0, behavior=Behavior.HONEST)[2]
    dodger = _compose(question, 2.0, behavior=Behavior.DODGER)[2]
    assert dodger.n_concepts < honest.n_concepts
    assert dodger.deflections


def test_bluffer_borrows_vocabulary_without_gaining_knowledge(starter):
    question = starter.questions[0]
    honest = _compose(question, -1.0, behavior=Behavior.HONEST)[2]
    bluffer = _compose(question, -1.0, behavior=Behavior.BLUFFER)[2]
    assert bluffer.concepts == honest.concepts, "bluffing must not add real knowledge"
    assert bluffer.distractors


def test_injector_payload_is_present_verbatim(starter):
    """The robustness number is meaningless unless the payload really made it
    into the transcript. Phase 5 greps for exactly this."""
    question = starter.questions[0]
    text, _level, plan = _compose(question, 0.5, behavior=Behavior.INJECTOR)
    assert plan.injection in INJECTION_PAYLOADS
    assert plan.injection in text


def test_overclaimer_adds_claims_not_content(starter):
    question = starter.questions[0]
    honest = _compose(question, 0.0, behavior=Behavior.HONEST)[2]
    over = _compose(question, 0.0, behavior=Behavior.OVERCLAIMER)[2]
    assert over.concepts == honest.concepts
    assert over.overclaims


def test_adversarial_set_membership():
    assert not is_adversarial(Behavior.HONEST)
    assert all(
        is_adversarial(b) for b in Behavior if b is not Behavior.HONEST
    )


def test_every_behaviour_produces_a_plan(starter):
    pool = list(starter.questions[0].anchor(5).required_concepts)
    rng = random.Random(0)
    for behavior in Behavior:
        for level in range(1, 6):
            plan = plan_content(behavior, level, pool, ["hash partitioning"], rng)
            assert plan.n_concepts <= len(pool)


def test_segment_kinds_are_exhaustive():
    assert {k.value for k in SegmentKind} == {"concept", "prose", "filler"}
