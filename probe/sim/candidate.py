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

import json
from dataclasses import dataclass, field

from probe.models import LLMRole, Persona, Question, Transcript
from probe.rubric.taxonomy import Taxonomy, load_taxonomy
from probe.runtime.candidate import AnswerResult, AnswerSource
from probe.runtime.llm import LLMRequest
from probe.sim.answers import answer_seed, estimate_seconds


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
        request = LLMRequest(
            role=LLMRole.PERSONA_ANSWER,
            prompt=self._prompt(question),
            seed=self.seed,
            temperature=0.7,
            context={
                "question": question.model_dump(mode="json"),
                "style": self._persona.style.model_dump(mode="json"),
                "behavior": self._persona.behavior.value,
                "theta": theta,
                "distractor_pool": self._distractor_pool(question.competency_id),
                "answer_seed": answer_seed(
                    self._persona.id, question.id, self._persona.style.id, self.seed
                ),
                # Never signed into the answer text. The name-swap slice needs
                # byte-identical transcripts differing only in metadata, so the
                # name reaches the grader through its context — the way a real
                # grader would see it — rather than through the prose.
                "candidate_name": None,
            },
        )
        response = self.client.complete(request)
        payload = json.loads(response.text)
        text = payload["answer"]

        self.records.append(
            AnswerRecord(
                question_id=question.id,
                competency_id=question.competency_id,
                theta=theta,
                drawn_level=int(payload["drawn_level"]),
                n_concepts=int(payload["n_concepts"]),
                n_distractors=int(payload.get("n_distractors", 0)),
                text=text,
            )
        )
        return AnswerResult(
            text=text,
            seconds=estimate_seconds(question, text, self._persona.style),
            tokens=response.total_tokens,
        )

    def _prompt(self, question: Question) -> str:
        """The prompt a real provider would receive.

        It carries the persona's *behaviour and style* but not the number: a
        live backend is told to answer as somebody with a given depth of
        experience, and the depth is expressed in words. The firewall test
        reads this prompt, so if it ever starts interpolating ``theta`` the
        suite fails.
        """
        style = self._persona.style
        length = "in two or three sentences" if style.verbosity < 0.7 else "in a short paragraph"
        return (
            f"You are a candidate in a technical interview. Answer {length}, "
            f"in a {style.tone} register.\n"
            f"Behaviour to adopt: {self._persona.behavior.value}.\n\n"
            f"Question: {question.text}\n"
        )

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
