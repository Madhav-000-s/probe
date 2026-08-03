"""Composing a simulated answer.

The pipeline, in order:

1. draw a response level from the graded-response model applied to the
   persona's hidden ability for the question's competency;
2. turn that level into a content plan (:mod:`probe.sim.behavior`);
3. lay the plan out as tagged segments — concepts and prose;
4. render, styling prose only (:mod:`probe.sim.style`).

Step 1 is the whole reason any of this is measurable. Ability determines
content; content determines what a grader reading the text can recover. The
grader never sees step 1's input, and the ground-truth firewall test enforces
that. If this file ever grows a shortcut that writes the score directly into
the answer, every recovery number in the project becomes decoration.
"""

from __future__ import annotations

import hashlib
import random

from probe.belief.grm import sample_score
from probe.models import Behavior, Question, StyleProfile
from probe.sim.behavior import ContentPlan, plan_content
from probe.sim.style import (
    FILLER,
    Segment,
    SegmentKind,
    filler_count,
    paraphrase_concept,
    render,
)

_OPENERS = (
    "The way I would approach this is to look at",
    "The first thing I would check is",
    "In my experience what matters here is",
    "I would start from",
    "The core of it comes down to",
)

_CONNECTORS = (
    "The other piece is",
    "Alongside that you have",
    "You also have to weigh",
    "Closely related is",
    "And then there is",
)

_CLOSERS = (
    "That is usually where the real problem turns out to be",
    "Which one dominates depends on the traffic and the failure budget",
    "I would validate that with a measurement before committing to it",
)


def answer_seed(persona_id: str, question_id: str, style_id: str, seed: int) -> int:
    """A stable per-answer seed.

    Derived from the identifiers rather than a counter so an answer is
    reproducible in isolation — which is what makes a resumed run identical to
    an uninterrupted one, and what lets the reliability suite re-grade the
    *same* answer under five different grader seeds.
    """
    h = hashlib.sha256(f"{persona_id}|{question_id}|{style_id}|{seed}".encode()).digest()
    return int.from_bytes(h[:8], "big")


def draw_level(theta: float, question: Question, rng: random.Random) -> int:
    """The graded response the persona's ability implies for this item."""
    return sample_score(theta, question.grm.a, question.grm.b, rng)


def build_segments(
    plan: ContentPlan,
    style: StyleProfile,
    rng: random.Random,
    candidate_name: str | None = None,
) -> list[Segment]:
    """Lay the content plan out, concepts first.

    Front-loading concepts is deliberate: the terse style clips from the end,
    so a terse answer loses padding rather than substance. A terse candidate
    who knows the material still says the words that matter, which is what
    makes "terse candidates are under-rated" a grader failure rather than a
    tautology.
    """
    segs: list[Segment] = []

    if plan.overclaims:
        segs.append(Segment(SegmentKind.PROSE, f"{plan.overclaims[0]}, so"))
    if plan.bluff_framing:
        segs.append(Segment(SegmentKind.PROSE, f"{plan.bluff_framing} —"))

    segs.append(Segment(SegmentKind.PROSE, rng.choice(_OPENERS)))

    for i, concept in enumerate(plan.concepts):
        if i > 0:
            segs.append(Segment(SegmentKind.PROSE, rng.choice(_CONNECTORS)))
        rendered = concept
        # First-language transfer paraphrases a concept some of the time. An
        # exact-match extractor misses it, which is the residual fairness bug
        # the content-style intervention cannot reach.
        if style.l1_transfer > 0 and rng.random() < style.l1_transfer * 0.35:
            rendered = paraphrase_concept(concept, rng)
        segs.append(Segment(SegmentKind.CONCEPT, rendered))
        segs.append(Segment(SegmentKind.PROSE, "."))

    for distractor in plan.distractors:
        segs.append(Segment(SegmentKind.PROSE, "You also want to think about"))
        segs.append(Segment(SegmentKind.CONCEPT, distractor))
        segs.append(Segment(SegmentKind.PROSE, "."))

    for deflection in plan.deflections:
        segs.append(Segment(SegmentKind.PROSE, f"Honestly, {deflection}."))

    n_filler = filler_count(style) + (2 if plan.length_scale > 2.0 else 0)
    if plan.length_scale < 0.5:
        n_filler = 0
    for _ in range(n_filler):
        segs.append(Segment(SegmentKind.FILLER, rng.choice(FILLER) + "."))

    if plan.concepts and plan.length_scale >= 0.5:
        segs.append(Segment(SegmentKind.FILLER, rng.choice(_CLOSERS) + "."))

    if not plan.concepts and not plan.deflections:
        segs.append(
            Segment(SegmentKind.PROSE, "I have not run into that specific situation before.")
        )

    if candidate_name:
        segs.append(Segment(SegmentKind.PROSE, f"— {candidate_name}"))

    if plan.injection:
        # Appended verbatim. The transcript must contain the real payload or
        # the injection-resistance number is measuring nothing; a Phase 5 test
        # greps for exactly this.
        segs.append(Segment(SegmentKind.PROSE, plan.injection))

    return segs


def compose_answer(
    *,
    question: Question,
    theta: float,
    behavior: Behavior,
    style: StyleProfile,
    distractor_pool: list[str],
    seed: int,
    candidate_name: str | None = None,
) -> tuple[str, int, ContentPlan]:
    """Produce ``(answer_text, drawn_level, plan)``.

    ``drawn_level`` is returned for the measurement plane only — it is the
    latent content level the answer was built to express, and the eval harness
    uses it to separate "the candidate did not say it" from "the grader did not
    see it". It must never reach the interview plane.
    """
    rng = random.Random(seed)
    level = draw_level(theta, question, rng)
    pool = list(question.anchor(5).required_concepts)
    plan = plan_content(behavior, level, pool, distractor_pool, rng)
    segments = build_segments(plan, style, rng, candidate_name=candidate_name)
    text = render(segments, style, rng)
    return text, level, plan


def estimate_seconds(question: Question, text: str, style: StyleProfile) -> float:
    """How long this answer would have taken to give.

    Anchored on the item's expected time and adjusted by how much was actually
    said. The efficiency suite reports a time-normalised variant precisely
    because "fewer questions" is not the same claim as "less of the
    candidate's time", and an adaptive policy that asks three long questions
    instead of five short ones has not obviously won.
    """
    words = max(1, len(text.split()))
    ratio = words / 70.0
    return round(question.expected_seconds * (0.45 + 0.75 * ratio), 2)
