"""Generated follow-up questions.

When an answer is high-variance or flagged evasive, drawing the next item from
the bank wastes the one thing a follow-up is good at: asking about the
specific thing that was left out. So the agent may generate a targeted
question instead.

The item parameters are the interesting part. A generated question has no
calibration data at all, so ``a`` is estimated online by shrinking toward the
parent item's fitted discrimination:

    a_followup = w * a_parent + (1 - w) * a_prior

with ``w`` fixed and conservative. Treating a brand-new question as though it
discriminated as well as a calibrated one would let the belief state update
harder on the least trustworthy evidence in the interview — precisely
backwards.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from probe.models import (
    GRMParams,
    LLMRole,
    Question,
    RubricAnchor,
    Transcript,
)
from probe.runtime.llm import LLMRequest
from probe.runtime.retry import structured_call

#: Weight on the parent's discrimination. Conservative: a follow-up inherits
#: most of the parent's measurement quality but is never trusted more than it.
PARENT_SHRINKAGE = 0.65
#: The population-level prior the rest is shrunk toward.
PRIOR_A = 1.0

#: Posterior SD above which a competency is still worth a follow-up.
HIGH_VARIANCE_TRIGGER = 0.75


class FollowUpSpec(BaseModel):
    text: str = Field(min_length=15, max_length=600)
    targets_concepts: list[str] = Field(default_factory=list)
    rationale: str = ""


def shrunk_discrimination(parent_a: float, weight: float = PARENT_SHRINKAGE) -> float:
    """Online estimate for a generated item, bounded by construction."""
    return round(weight * parent_a + (1.0 - weight) * PRIOR_A, 4)


def shrinkage_bounds(parent_a: float) -> tuple[float, float]:
    """The interval any valid online estimate must fall in.

    A shrinkage estimator can never leave the segment between the two things
    it is blending, which makes this a property a test can assert directly
    rather than a range somebody eyeballed.
    """
    return (min(parent_a, PRIOR_A), max(parent_a, PRIOR_A))


def should_follow_up(
    posterior_sd: float, flags: list, followups_available: bool
) -> bool:
    """Trigger on genuine ambiguity, not on a low score.

    A confidently-wrong answer needs no follow-up; an evasive or
    high-variance one does. Following up on every weak answer would turn the
    interview into an interrogation and spend the budget on candidates already
    well measured.
    """
    if not followups_available:
        return False
    evasive = any(getattr(f, "value", f) in {"non_answer", "off_topic"} for f in flags)
    return evasive or posterior_sd >= HIGH_VARIANCE_TRIGGER


def build_followup(
    parent: Question,
    spec: FollowUpSpec,
    index: int,
) -> Question:
    """A follow-up inherits the parent's competency and anchors.

    Inheriting the anchors keeps the follow-up on the same measurement scale —
    a generated question with freshly generated anchors would be measuring
    something subtly different and the posterior would be folding in evidence
    from two different instruments.
    """
    return Question(
        id=f"{parent.id}::followup{index}",
        competency_id=parent.competency_id,
        probe_family=parent.probe_family,
        text=spec.text,
        anchors=[
            RubricAnchor(
                level=a.level,
                descriptor=a.descriptor,
                required_concepts=list(a.required_concepts),
            )
            for a in parent.anchors
        ],
        grm=GRMParams(
            a=shrunk_discrimination(parent.grm.a),
            b=list(parent.grm.b),
            calibrated=False,
        ),
        expected_seconds=max(30.0, parent.expected_seconds * 0.6),
        parent_question_id=parent.id,
    )


class FollowUpGenerator:
    def __init__(self, client, *, seed: int = 0) -> None:
        self.client = client
        self.seed = seed
        self.generated = 0

    def generate(
        self, parent: Question, answer: str, transcript: Transcript, unnamed: list[str]
    ) -> Question | None:
        request = LLMRequest(
            role=LLMRole.FOLLOWUP_GEN,
            prompt=(
                "The candidate's answer left something out. Write ONE short "
                "follow-up question that targets the gap specifically. Do not "
                "restate the original question, do not ask a second question, "
                "and do not hint at the answer.\n\n"
                f"Original question: {parent.text}\n"
                f"Competency: {parent.competency_id}\n"
                f"Ideas not addressed: {', '.join(unnamed[:4]) or 'unclear'}\n\n"
                'Return JSON: {"text": "...", "targets_concepts": [...], '
                '"rationale": "..."}\n'
            ),
            seed=self.seed + len(transcript.turns),
            temperature=0.3,
            context={
                "parent_question": parent.text,
                "competency_id": parent.competency_id,
                "unnamed_concepts": unnamed,
                "answer": answer,
            },
            run_id=transcript.run_id or None,
        )
        result = structured_call(
            self.client,
            request,
            FollowUpSpec,
            degraded=lambda: FollowUpSpec(
                text=(
                    f"You did not touch on {unnamed[0] if unnamed else 'part of that'}. "
                    "How does that factor in here?"
                ),
                targets_concepts=unnamed[:2],
                rationale="deterministic fallback",
            ),
            max_repairs=1,
        )
        if result.value is None:
            return None
        self.generated += 1
        return build_followup(parent, result.value, self.generated)
