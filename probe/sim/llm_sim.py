"""``SimLLM`` — a deterministic, seeded stand-in for the model provider.

Read this before believing any number this repo produces.

SimLLM is not a mock returning canned strings. It is a generative model of
each role, and the channel it implements is real:

    theta*  ->  the persona names concepts at a rate governed by the GRM
            ->  the answer text literally contains those concept phrases
            ->  the grader reads the text, extracts concepts, and scores
            ->  the belief state updates on that score

The grader never sees ``theta*``. It recovers ability from prose, and it can
fail: bluffers pad with borrowed vocabulary, dodgers omit, terse answers get
clipped, first-language paraphrase defeats exact matching, and the grader
carries genuine noise, position bias and (with the intervention off) a
sensitivity to length and tone. Those failure modes are what the reliability,
robustness and fairness suites measure.

**What this does and does not license.** Q1 — does an EIG policy reach
confidence in fewer questions — is a psychometric claim about selection under
a known response model, and simulation is how adaptive-testing research
answers it. Q2 and Q3 measured against SimLLM are measurements of parameters
chosen here, not findings about any real grader. The README says so first,
before any result. Point ``--backend anthropic`` at the same interfaces to
replace the simulated cells with measured ones.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from functools import lru_cache
from typing import Any

from probe.bank.loader import LEVEL_THRESHOLDS
from probe.grader.flags import (
    NON_ANSWER_WORDS,
    detect_injection,
    has_deflection,
    has_overclaim,
)
from probe.models import GradeFlag, LLMRole
from probe.rubric.taxonomy import Taxonomy, load_taxonomy
from probe.runtime.llm import LLMRequest, LLMResponse, estimate_tokens
from probe.sim.textfeatures import (
    count_offpool_concepts,
    level_from_matches,
    match_concepts,
    style_features,
)


@lru_cache(maxsize=4096)
def _keyword_pattern(keyword: str) -> re.Pattern[str]:
    return re.compile(rf"(?<!\w){re.escape(keyword.lower())}(?!\w)")


def _mentions(text: str, keyword: str) -> bool:
    """Word-boundary keyword match.

    Plain substring matching was silently wrong and expensively so: the
    two-letter keywords ("ml", "ci", "cd") matched inside unrelated words, so a
    data/ML job description came out requiring half the backend taxonomy. The
    rubric then filled with competencies the bank had no items for and the
    interview ran short. Boundaries are not pedantry here; they decide which
    competencies get interviewed.
    """
    return bool(_keyword_pattern(keyword).search(text.lower()))


class SimLLM:
    """Deterministic given ``(prompt, seed)``.

    Every knob below is an explicit modelling choice with a stated purpose.
    They are attributes rather than constants so ablations — a noisier grader,
    a cleaner one, the style intervention on and off — are a constructor
    argument rather than a code edit.
    """

    name = "sim"

    def __init__(
        self,
        seed: int = 0,
        *,
        taxonomy: Taxonomy | None = None,
        model: str = "sim-grader-v1",
        #: SD of grader noise on the latent score, in rubric levels. Drives
        #: test-retest variance; 0 would make the reliability suite vacuous.
        grader_noise_sd: float = 0.42,
        #: Weight on surface features when the content-style separation
        #: intervention is OFF. This is the bug Q3 is looking for.
        style_weight: float = 0.62,
        #: Weight that survives the intervention. Small but not zero — an
        #: intervention that provably zeroed its own target would be a
        #: tautology, and real ones do not.
        residual_style_weight: float = 0.07,
        #: How much a grader is swayed by borrowed vocabulary from other
        #: competencies. The bluffer's entire strategy.
        distractor_credit: float = 0.30,
        #: Drift per turn of position in the transcript.
        position_bias: float = 0.035,
        #: Probability a response comes back structurally invalid, exercising
        #: the repair ladder in real runs so schema-violation and
        #: repair-success rates are measured rather than assumed.
        schema_violation_rate: float = 0.035,
        #: Noise for the blind rater used by the fidelity gate. Lower than the
        #: grader's: it stands in for a stronger model doing an easier task.
        blind_noise_sd: float = 0.30,
    ) -> None:
        self.seed = seed
        self.model = model
        self.taxonomy = taxonomy or load_taxonomy()
        self.grader_noise_sd = grader_noise_sd
        self.style_weight = style_weight
        self.residual_style_weight = residual_style_weight
        self.distractor_credit = distractor_credit
        self.position_bias = position_bias
        self.schema_violation_rate = schema_violation_rate
        self.blind_noise_sd = blind_noise_sd
        self._world_concepts = [c for node in self.taxonomy for c in node.concepts]

    # ------------------------------------------------------------ dispatch

    def complete(self, request: LLMRequest) -> LLMResponse:
        handler = {
            LLMRole.GRADE: self._grade,
            LLMRole.RUBRIC_COMPILE: self._compile_rubric,
            LLMRole.BLIND_RATE: self._blind_rate,
            LLMRole.FLAG_CLASSIFY: self._flag_classify,
            LLMRole.PERSONA_ANSWER: self._persona_answer,
        }.get(request.role)
        if handler is None:
            raise NotImplementedError(
                f"SimLLM has no handler for role {request.role.value!r} yet"
            )
        text = handler(request, self._rng(request))
        return LLMResponse(
            text=text,
            model=self.model,
            prompt_tokens=estimate_tokens(request.prompt),
            completion_tokens=estimate_tokens(text),
            latency_ms=0.0,
        )

    def _rng(self, request: LLMRequest) -> random.Random:
        """Seeded by the call's identity, so the same prompt under the same
        seed always yields the same output, and a different grader seed yields
        a genuinely different draw. Both halves are needed: the first for
        reproducibility, the second for test-retest variance."""
        digest = hashlib.sha256(
            f"{request.hash}|{self.seed}|{request.seed}".encode()
        ).digest()
        return random.Random(int.from_bytes(digest[:8], "big"))

    # --------------------------------------------------------------- grading

    def _grade(self, request: LLMRequest, rng: random.Random) -> str:
        ctx = request.context
        answer: str = ctx.get("answer", "")
        pool: list[str] = list(ctx.get("concept_pool", []))
        competency_id: str = ctx.get("competency_id", "unknown")
        style_separation: bool = bool(ctx.get("style_separation", True))
        n_prior: int = int(ctx.get("n_prior_turns", 0))
        resume_claims: list[str] = list(ctx.get("resume_claims", []))

        if rng.random() < self.schema_violation_rate:
            return self._malformed(rng)

        matches = match_concepts(answer, pool)
        level = level_from_matches(len(matches), LEVEL_THRESHOLDS)
        latent = float(level)

        # Borrowed vocabulary from other competencies. A grader that counts
        # impressive terms rather than relevant ones rewards this.
        offpool = count_offpool_concepts(answer, pool, self._world_concepts)
        if offpool:
            latent += self.distractor_credit * min(2, offpool)

        feats = style_features(answer)
        weight = self.style_weight if not style_separation else self.residual_style_weight
        latent += weight * (
            0.9 * feats.length_z
            + 0.7 * feats.assertive_density
            - 0.8 * feats.hedge_density
            - 0.5 * feats.l1_density
        )

        latent += self.position_bias * (n_prior - 3)
        latent += rng.gauss(0.0, self.grader_noise_sd)

        score = max(1, min(5, int(round(latent))))
        flags = self._flags(answer, matches, offpool, score, resume_claims)
        if GradeFlag.NON_ANSWER in flags:
            score = min(score, 2)

        spans = self._spans(answer, matches)
        margin = abs(latent - score)
        confidence = max(0.05, min(0.97, 0.92 - 0.35 * margin - (0.2 if flags else 0.0)))

        return json.dumps(
            {
                "competency_id": competency_id,
                "score": score,
                "confidence": round(confidence, 3),
                "evidence_spans": spans,
                "flags": [f.value for f in flags],
                "rationale": (
                    f"names {len(matches)} of {len(pool)} anchor concepts"
                    + (f"; {offpool} off-topic technical terms" if offpool else "")
                ),
            }
        )

    def _flags(
        self,
        answer: str,
        matches: list,
        offpool: int,
        score: int,
        resume_claims: list[str],
    ) -> list[GradeFlag]:
        flags: list[GradeFlag] = []
        if detect_injection(answer):
            flags.append(GradeFlag.INJECTION_ATTEMPT)
        words = len(answer.split())
        if words < NON_ANSWER_WORDS or (has_deflection(answer) and not matches):
            flags.append(GradeFlag.NON_ANSWER)
        if has_overclaim(answer):
            flags.append(GradeFlag.UNSUPPORTED_CLAIM)
        # The bluff signature: lots of borrowed vocabulary, little on point.
        if offpool >= 2 and len(matches) <= 2:
            flags.append(GradeFlag.UNSUPPORTED_CLAIM)
        if not matches and not has_deflection(answer) and words >= NON_ANSWER_WORDS:
            flags.append(GradeFlag.OFF_TOPIC)
        # The resume said strong, the answer says otherwise.
        if resume_claims and score <= 2:
            flags.append(GradeFlag.RESUME_CONTRADICTION)
        return sorted(set(flags), key=lambda f: f.value)

    @staticmethod
    def _spans(answer: str, matches: list) -> list[dict[str, Any]]:
        """Evidence spans, always resolvable against the answer.

        When nothing matched there is still a span — over the opening clause —
        because a grade with no citation is rejected upstream, and a grader
        that could never cite a weak answer would make every weak answer
        unrecoverable rather than merely low-scoring.
        """
        if matches:
            return [
                {"start": m.start, "end": m.end, "text": answer[m.start : m.end]}
                for m in matches[:3]
            ]
        body = answer.strip()
        if not body:
            return []
        start = answer.index(body[0])
        end = min(len(answer), start + min(len(body), 60))
        return [{"start": start, "end": end, "text": answer[start:end]}]

    def _malformed(self, rng: random.Random) -> str:
        """A structurally invalid response, of the kinds models actually
        produce: truncation, prose, and a plausible-looking object with the
        wrong field types."""
        return rng.choice(
            [
                '{"competency_id": "x", "score": 4, "confidence": 0.8, "evidence_spans": [',
                "Based on the rubric I would place this answer at about a 4 out of 5.",
                '{"competency_id": "x", "score": "four", "confidence": "high", '
                '"evidence_spans": []}',
            ]
        )

    # ------------------------------------------------------------ blind rate

    def _blind_rate(self, request: LLMRequest, rng: random.Random) -> str:
        """A stronger model rating an answer blind.

        Used only by the Phase 1 fidelity gate and the Phase 4 human-anchor
        comparison. It is deliberately *not* the grader: no style term, no
        position bias, less noise. If it agreed with the grader by
        construction, correlating the two would be circular.
        """
        answer = request.context.get("answer", "")
        pool = list(request.context.get("concept_pool", []))
        matches = match_concepts(answer, pool)
        latent = float(level_from_matches(len(matches), LEVEL_THRESHOLDS))
        latent += rng.gauss(0.0, self.blind_noise_sd)
        rating = max(1, min(5, int(round(latent))))
        return json.dumps({"rating": rating, "n_concepts": len(matches)})

    # ---------------------------------------------------------------- flags

    def _flag_classify(self, request: LLMRequest, rng: random.Random) -> str:
        answer = request.context.get("answer", "")
        patterns = detect_injection(answer)
        return json.dumps(
            {
                "injection": bool(patterns),
                "matched_patterns": patterns[:3],
                "deflection": has_deflection(answer),
                "overclaim": has_overclaim(answer),
            }
        )

    # -------------------------------------------------------- persona answer

    def _persona_answer(self, request: LLMRequest, rng: random.Random) -> str:
        """Measurement-plane role: the persona speaking.

        Routed through the provider boundary rather than composed inline so
        that pointing ``--backend anthropic`` at the population generates real
        answers with no other change, and so every answer is traced like any
        other model call.
        """
        from probe.models import Behavior, Question, StyleProfile
        from probe.sim.answers import compose_answer

        ctx = request.context
        question = Question.model_validate(ctx["question"])
        style = StyleProfile.model_validate(ctx["style"])
        text, level, plan = compose_answer(
            question=question,
            theta=float(ctx["theta"]),
            behavior=Behavior(ctx["behavior"]),
            style=style,
            distractor_pool=list(ctx.get("distractor_pool", [])),
            seed=int(ctx["answer_seed"]),
            candidate_name=ctx.get("candidate_name"),
        )
        return json.dumps(
            {
                "answer": text,
                "drawn_level": level,
                "n_concepts": plan.n_concepts,
                "n_distractors": len(plan.distractors),
            }
        )

    # ------------------------------------------------------- rubric compiler

    def _compile_rubric(self, request: LLMRequest, rng: random.Random) -> str:
        """Map JD and resume text onto taxonomy ids.

        Keyword and concept matching, which is a fair model of what an
        extraction prompt with a strict schema does: it finds what is named
        and cites where. Crucially it can only *select* from the taxonomy, so
        the "never invents an id" commitment holds by construction here in the
        same way a schema-constrained prompt enforces it against a real model.
        """
        ctx = request.context
        jd_text: str = ctx.get("jd_text", "")
        resume: str = ctx.get("resume", "")
        seniority_level: int = int(ctx.get("seniority_level", 3))
        resume_lower = resume.lower()

        out: list[dict[str, Any]] = []
        for node in self.taxonomy:
            in_jd = _mentions(jd_text, node.label) or any(
                _mentions(jd_text, kw) for kw in node.jd_keywords
            )
            hits = match_concepts(resume, node.concepts)
            label_hit = node.label.lower() in resume_lower
            if not in_jd and not hits and not label_hit:
                continue

            spans = [
                {"start": m.start, "end": m.end, "text": resume[m.start : m.end]}
                for m in hits[:3]
            ]
            evidence = min(1.0, len(hits) / 3.0) if spans else 0.0
            out.append(
                {
                    "id": node.id,
                    "label": node.label,
                    "required_level": seniority_level if in_jd else 2,
                    "evidence_in_resume": round(evidence, 3),
                    "resume_spans": spans,
                    "probe_families": [p.value for p in node.probe_families],
                }
            )
        return json.dumps({"competencies": out})
