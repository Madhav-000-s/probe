"""A persona wired into the interview loop.

This class is the firewall in code form. It holds a ``Persona`` — hidden
ability and all — and exposes exactly one thing to the interview plane: text.
The ability goes into an LLM call under a measurement-plane role; the answer
comes back; the interview plane sees a string.

The extra fields it records (the drawn response level, how many concepts were
actually named) go to the eval harness, never into the loop. They are what
lets the harness distinguish "the candidate did not say it" from "the grader
did not see it" — a distinction the interview plane must not be able to make.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from probe.models import LLMRole, Persona, Question, Transcript
from probe.rubric.taxonomy import Taxonomy, load_taxonomy
from probe.runtime.candidate import AnswerResult, AnswerSource
from probe.runtime.llm import LLMRequest
from probe.runtime.retry import extract_json, structured_call
from probe.sim.answers import answer_seed, estimate_seconds, plan_answer


class AnswerEnvelope(BaseModel):
    """What a persona-answer call returns.

    Only the prose is taken from the model. ``drawn_level`` and the concept
    counts are accepted so the offline backend's richer payload validates
    unchanged, but the harness records its own values for them — see
    :meth:`PersonaCandidate.answer`.
    """

    answer: str = Field(min_length=1)
    drawn_level: int | None = None
    n_concepts: int | None = None
    n_distractors: int | None = None


def extract_prose(raw: str) -> str:
    """Salvage the answer from a response that ignored the envelope.

    A model that replied with the answer and nothing else has done the work;
    only the wrapper is missing. Dropping it would quietly shorten the
    transcript, and it would do so unevenly — biasing the interview toward
    whichever personas happened to draw a compliant response.
    """
    raw = (raw or "").strip()
    if not raw:
        return ""
    candidate = extract_json(raw)
    if candidate.startswith("{"):
        import json

        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict) and isinstance(payload.get("answer"), str):
            return payload["answer"].strip()
    # Not JSON at all, or JSON we cannot use: treat the whole reply as prose.
    return raw


@dataclass
class AnswerRecord:
    """Measurement-plane detail about one answer. Never enters the loop."""

    question_id: str
    competency_id: str
    theta: float
    drawn_level: int
    n_concepts: int
    n_distractors: int
    text: str


class PersonaCandidate(AnswerSource):
    def __init__(
        self,
        persona: Persona,
        client,
        *,
        taxonomy: Taxonomy | None = None,
        seed: int = 0,
        include_name: bool = False,
        distractors_per_question: int = 6,
    ) -> None:
        self._persona = persona
        self.client = client
        self.taxonomy = taxonomy or load_taxonomy()
        self.seed = seed
        self.id = persona.id
        self.style_id = persona.style.id
        self.include_name = include_name
        self.distractors_per_question = distractors_per_question
        #: Ground-truth-adjacent audit trail for the eval harness.
        self.records: list[AnswerRecord] = []

    # ------------------------------------------------------------- internals

    def _distractor_pool(self, competency_id: str) -> list[str]:
        """Plausible-but-wrong vocabulary: concepts from *sibling* competencies
        in the same area, which is exactly what a bluffer reaches for."""
        area = competency_id.split(".", 1)[0]
        siblings = [
            n for n in self.taxonomy if n.id != competency_id and n.id.startswith(area + ".")
        ]
        if not siblings:
            siblings = [n for n in self.taxonomy if n.id != competency_id][:4]
        pool: list[str] = []
        for node in siblings:
            pool.extend(node.concepts[:3])
        return pool[: self.distractors_per_question]

    # ---------------------------------------------------------------- answer

    def answer(self, question: Question, transcript: Transcript) -> AnswerResult:
        theta = self._persona.ability(question.competency_id)
        seed = answer_seed(self._persona.id, question.id, self._persona.style.id, self.seed)
        distractors = self._distractor_pool(question.competency_id)

        # Drawn here, not by the model. What this persona knows is ground truth
        # the harness owns; a live model asked to pick its own depth would be
        # scored against a theta it never saw, and recovery would be measuring
        # its self-report. The model is told what to express, never how much it
        # knows — which is also why this stays firewall-clean: the *number*
        # never reaches the prompt.
        level, plan, _ = plan_answer(
            question=question,
            theta=theta,
            behavior=self._persona.behavior,
            distractor_pool=distractors,
            seed=seed,
        )

        request = LLMRequest(
            role=LLMRole.PERSONA_ANSWER,
            prompt=self._prompt(question, plan),
            seed=self.seed,
            temperature=0.7,
            context={
                "question": question.model_dump(mode="json"),
                "style": self._persona.style.model_dump(mode="json"),
                "behavior": self._persona.behavior.value,
                "theta": theta,
                "distractor_pool": distractors,
                "answer_seed": seed,
                # Never signed into the answer text. The name-swap slice needs
                # byte-identical transcripts differing only in metadata, so the
                # name reaches the grader through its context — the way a real
                # grader would see it — rather than through the prose.
                "candidate_name": None,
            },
        )
        result = structured_call(self.client, request, AnswerEnvelope)
        if result.usable:
            text = result.value.answer
        else:
            # A live model that answered the question in prose has done the job
            # asked of it; only the envelope is missing. Discarding a perfectly
            # good answer over its wrapper would silently shrink the transcript
            # and bias the interview toward candidates whose model complied.
            text = extract_prose(result.raw[-1] if result.raw else "")
            if not text:
                raise RuntimeError(
                    f"persona answer for {question.id} was unrecoverable: "
                    f"{result.errors[-1] if result.errors else 'empty response'}"
                )

        self.records.append(
            AnswerRecord(
                question_id=question.id,
                competency_id=question.competency_id,
                theta=theta,
                drawn_level=level,
                n_concepts=plan.n_concepts,
                n_distractors=len(plan.distractors),
                text=text,
            )
        )
        return AnswerResult(
            text=text,
            seconds=estimate_seconds(question, text, self._persona.style),
            tokens=result.tokens,
        )

    def _prompt(self, question: Question, plan) -> str:
        """The prompt a real provider would receive.

        It carries the persona's behaviour and style, and the *content* the
        harness has already decided this answer expresses — the concepts to
        cover and, for a bluffer, the plausible-but-wrong vocabulary to reach
        for. What it never carries is ``theta`` or the drawn level as a number.
        The firewall test reads this prompt, so if it ever starts interpolating
        either, the suite fails.

        Depth is expressed as coverage rather than as a score for the same
        reason: "mention these three things and not the other two" is a
        reproducible instruction, where "answer as a level-3 candidate" invites
        the model to apply its own idea of what a 3 is.
        """
        style = self._persona.style
        length = "in two or three sentences" if style.verbosity < 0.7 else "in a short paragraph"
        lines = [
            f"You are a candidate in a technical interview. Answer {length}, "
            f"in a {style.tone} register.",
            f"Behaviour to adopt: {self._persona.behavior.value}.",
        ]
        if plan.concepts:
            lines.append("Cover exactly these ideas, in your own words: " + ", ".join(plan.concepts))
        else:
            lines.append("You do not know this topic well. Do not cover it convincingly.")
        if plan.distractors:
            lines.append(
                "Also reach for this related-but-off-target vocabulary, as somebody "
                "overstating their experience would: " + ", ".join(plan.distractors)
            )
        lines.append(f"\nQuestion: {question.text}\n")
        lines.append('Return only: {"answer": "<your answer>"}')
        return "\n".join(lines)

    # ------------------------------------------------------- eval-side views

    def record_for(self, question_id: str) -> AnswerRecord | None:
        for record in self.records:
            if record.question_id == question_id:
                return record
        return None


@dataclass
class BlindRatingRequest:
    """One item in a blind-rating batch for the fidelity gate."""

    persona_id: str
    competency_id: str
    question_id: str
    answer: str
    concept_pool: list[str]
    theta: float = field(repr=False, default=0.0)
