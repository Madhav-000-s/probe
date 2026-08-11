"""Typed contracts shared by both planes.

Every LLM call in this project parses into one of these models. Nothing
downstream ever handles a raw string that was supposed to be structured: the
repair loop in :mod:`probe.runtime.retry` is the only place a malformed
response is tolerated, and it is tolerated for exactly two attempts.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# --------------------------------------------------------------------------
# enumerations
# --------------------------------------------------------------------------


class ProbeFamily(StrEnum):
    """The shape of a question, independent of what it targets.

    The policy penalises drawing the same family repeatedly so an interview
    does not degenerate into six consecutive "tell me about a time when".
    """

    SCENARIO = "scenario"
    TRADEOFF = "tradeoff"
    DEBUG = "debug"
    PAST_PROJECT = "past_project"


class RoleFamily(StrEnum):
    BACKEND = "backend"
    DATA_ML = "data_ml"


class Behavior(StrEnum):
    """How a simulated candidate converts ability into text.

    ``HONEST`` is the Phase 1 population. The remaining six are the adversarial
    subset introduced in Phase 5 and make up roughly a quarter of the final
    population.
    """

    HONEST = "honest"
    BLUFFER = "bluffer"
    TERSE = "terse"
    RAMBLER = "rambler"
    INJECTOR = "injector"
    DODGER = "dodger"
    OVERCLAIMER = "overclaimer"


class GradeFlag(StrEnum):
    UNSUPPORTED_CLAIM = "unsupported_claim"
    OFF_TOPIC = "off_topic"
    INJECTION_ATTEMPT = "injection_attempt"
    RESUME_CONTRADICTION = "resume_contradiction"
    NON_ANSWER = "non_answer"


class StopReason(StrEnum):
    """Which of the three stop conditions fired. Logged per run; the
    distribution over these is itself a reported result."""

    CONFIDENCE = "confidence"
    BUDGET_QUESTIONS = "budget_questions"
    BUDGET_TOKENS = "budget_tokens"
    BUDGET_WALLCLOCK = "budget_wallclock"
    NO_INFORMATIVE_QUESTION = "no_informative_question"
    BANK_EXHAUSTED = "bank_exhausted"
    UNRECOVERABLE = "unrecoverable"


class LLMRole(StrEnum):
    """Every distinct prompt template in the system.

    Roles on the interview plane may never receive ground truth; roles on the
    measurement plane may. The firewall test keys off this distinction.
    """

    RUBRIC_COMPILE = "rubric_compile"
    GRADE = "grade"
    FLAG_CLASSIFY = "flag_classify"
    POLICY_CHOOSE = "policy_choose"
    FOLLOWUP_GEN = "followup_gen"
    # measurement plane only
    PERSONA_ANSWER = "persona_answer"
    PERSONA_RESUME = "persona_resume"
    BLIND_RATE = "blind_rate"
    ENTAILMENT_AUDIT = "entailment_audit"


#: Roles that run inside an interview. A prompt issued under one of these roles
#: is subject to the ground-truth firewall.
INTERVIEW_PLANE_ROLES: frozenset[LLMRole] = frozenset(
    {
        LLMRole.RUBRIC_COMPILE,
        LLMRole.GRADE,
        LLMRole.FLAG_CLASSIFY,
        LLMRole.POLICY_CHOOSE,
        LLMRole.FOLLOWUP_GEN,
    }
)


# --------------------------------------------------------------------------
# rubric
# --------------------------------------------------------------------------


class TaxonomyNode(BaseModel):
    """One competency in the fixed taxonomy.

    The compiler maps job-description and resume text *onto* these nodes; it
    never invents an id. Stable ids are what make question-bank parameters
    reusable and candidates comparable across runs.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    label: str
    family: RoleFamily
    probe_families: list[ProbeFamily]
    #: Canonical concept phrases a strong answer is expected to touch. These
    #: drive both rubric anchor authoring and, on the measurement plane, the
    #: content of simulated answers.
    concepts: list[str]
    jd_keywords: list[str] = Field(default_factory=list)
    default_required_level: int = Field(default=3, ge=1, le=5)


class EvidenceSpan(BaseModel):
    """A half-open character range into a source document, plus the text it
    covers. Storing the text alongside the offsets makes the span checkable
    after the fact without re-reading the source."""

    model_config = ConfigDict(frozen=True)

    start: int = Field(ge=0)
    end: int = Field(ge=0)
    text: str

    @model_validator(mode="after")
    def _ordered_and_nonempty(self) -> EvidenceSpan:
        if self.end <= self.start:
            raise ValueError(f"span end {self.end} must exceed start {self.start}")
        if not self.text.strip():
            raise ValueError("span text must be non-empty")
        return self

    def verify_against(self, source: str) -> bool:
        """True when the offsets are in range and actually cover ``text``."""
        if self.end > len(source):
            return False
        return source[self.start : self.end] == self.text

    def relocated_in(self, source: str) -> EvidenceSpan | None:
        """This span with offsets corrected, or ``None`` if the quote is absent.

        Separates the two failures that used to look identical. A grader that
        quotes the answer accurately but mis-reports where the quote sits has
        made an arithmetic error — real models cannot count characters, and a
        repair prompt asking them to try again produces the same mistake. A
        grader that quotes text which is *not in the answer* has fabricated
        evidence, and that must still be rejected.

        Exact substring only. Accepting near-matches here would let a
        paraphrase pass as a quotation, which is the thing the span exists to
        prevent.
        """
        if self.verify_against(source):
            return self
        quote = self.text.strip()
        start = source.find(quote)
        if start < 0:
            return None
        return EvidenceSpan(start=start, end=start + len(quote), text=quote)


class Competency(BaseModel):
    """A taxonomy node instantiated for one candidate.

    ``prior_var`` is the mechanism the whole gap-probing behaviour emerges
    from: a competency the job requires and the resume is silent about starts
    with a wide prior, which makes it the highest-expected-information target
    for the policy. Nothing in the policy knows the word "gap".
    """

    id: str
    label: str
    required_level: int = Field(ge=1, le=5)
    evidence_in_resume: float = Field(ge=0.0, le=1.0)
    prior_mean: float
    prior_var: float = Field(gt=0.0)
    probe_families: list[ProbeFamily]
    resume_spans: list[EvidenceSpan] = Field(default_factory=list)

    @model_validator(mode="after")
    def _evidence_requires_span(self) -> Competency:
        # Design commitment: evidence_in_resume must be span-backed or zero.
        if self.evidence_in_resume > 0.0 and not self.resume_spans:
            raise ValueError(
                f"{self.id}: evidence_in_resume={self.evidence_in_resume} with no resume span"
            )
        return self


class Rubric(BaseModel):
    """The compiled interview target for one candidate."""

    candidate_id: str
    role_title: str
    competencies: list[Competency]
    taxonomy_version: str

    @property
    def ids(self) -> list[str]:
        return [c.id for c in self.competencies]

    @property
    def required(self) -> list[Competency]:
        """Competencies the job description actually demands.

        The confidence stop rule is evaluated over these only — being unsure
        about something the role does not need is not a reason to keep asking.
        """
        return [c for c in self.competencies if c.required_level >= 3]

    def get(self, competency_id: str) -> Competency:
        for c in self.competencies:
            if c.id == competency_id:
                return c
        raise KeyError(competency_id)


# --------------------------------------------------------------------------
# question bank
# --------------------------------------------------------------------------


class RubricAnchor(BaseModel):
    """What a given score level looks like, written in terms of *concepts
    named* rather than prose quality.

    This phrasing is the content–style separation intervention: an anchor that
    says "names partition tolerance and quorum reads" cannot be satisfied by
    fluency alone, where "gives a clear, articulate explanation" can.
    """

    model_config = ConfigDict(frozen=True)

    level: int = Field(ge=1, le=5)
    descriptor: str
    required_concepts: list[str] = Field(default_factory=list)


class GRMParams(BaseModel):
    """Samejima graded-response parameters for one item.

    ``a`` is discrimination; ``b`` holds the four category thresholds for
    P(score >= k), k = 2..5. Thresholds must be strictly increasing or the
    category probabilities go negative.
    """

    a: float = Field(gt=0.0)
    b: list[float] = Field(min_length=4, max_length=4)
    #: True once fitted from data rather than seeded from an authoring default.
    calibrated: bool = False
    #: Set by the calibration sanity check; quarantined items never reach eval.
    quarantined: bool = False
    quarantine_reason: str | None = None

    @field_validator("b")
    @classmethod
    def _strictly_increasing(cls, v: list[float]) -> list[float]:
        if any(v[i] >= v[i + 1] for i in range(len(v) - 1)):
            raise ValueError(f"GRM thresholds must be strictly increasing, got {v}")
        return v


class Question(BaseModel):
    """One item in the bank."""

    id: str
    competency_id: str
    probe_family: ProbeFamily
    text: str
    anchors: list[RubricAnchor] = Field(min_length=5, max_length=5)
    grm: GRMParams
    #: Expected answer time in seconds. Makes the policy budget-aware rather
    #: than greedy on bits: a question worth 0.4 nats in 30s beats one worth
    #: 0.5 nats in 180s.
    expected_seconds: float = Field(gt=0.0)
    #: Populated for generated follow-ups only.
    parent_question_id: str | None = None

    @property
    def is_followup(self) -> bool:
        return self.parent_question_id is not None

    def anchor(self, level: int) -> RubricAnchor:
        for a in self.anchors:
            if a.level == level:
                return a
        raise KeyError(level)


class QuestionBank(BaseModel):
    version: str
    taxonomy_version: str
    questions: list[Question]

    def __len__(self) -> int:
        return len(self.questions)

    def get(self, question_id: str) -> Question:
        for q in self.questions:
            if q.id == question_id:
                return q
        raise KeyError(question_id)

    def for_competency(self, competency_id: str, include_quarantined: bool = False) -> list[Question]:
        return [
            q
            for q in self.questions
            if q.competency_id == competency_id
            and (include_quarantined or not q.grm.quarantined)
        ]

    def live(self) -> list[Question]:
        """Non-quarantined items — the only ones eval is allowed to use."""
        return [q for q in self.questions if not q.grm.quarantined]


# --------------------------------------------------------------------------
# grading
# --------------------------------------------------------------------------


class Grade(BaseModel):
    """A grader verdict on one answer.

    There is deliberately no field here that can express "set all scores" or
    "ignore the rubric". The schema is the first line of injection defence:
    even a fully compromised grader can only emit a score in 1..5 for the one
    competency the question targeted.
    """

    competency_id: str
    score: int = Field(ge=1, le=5)
    confidence: float = Field(ge=0.0, le=1.0)
    #: Mandatory. A grade whose spans do not validate against the answer text
    #: is rejected and regenerated — this is what grounds every score in the
    #: transcript and gives auditability for free.
    evidence_spans: list[EvidenceSpan]
    flags: list[GradeFlag] = Field(default_factory=list)
    rationale: str = ""

    def spans_valid_for(self, answer: str) -> bool:
        return bool(self.evidence_spans) and all(
            s.verify_against(answer) for s in self.evidence_spans
        )


# --------------------------------------------------------------------------
# simulated candidates (measurement plane)
# --------------------------------------------------------------------------


class StyleProfile(BaseModel):
    """Surface characteristics applied to an answer *after* its content is
    fixed. Style must never change what concepts an answer contains — that
    invariant is what makes the fairness suite measure grader drift rather
    than a genuine content difference."""

    model_config = ConfigDict(frozen=True)

    id: str
    #: Multiplier on answer length. 0.35 = terse, 2.0 = verbose.
    verbosity: float = Field(gt=0.0)
    #: Density of hedging constructions ("I think", "possibly").
    hedging: float = Field(ge=0.0, le=1.0)
    #: Density of assertive constructions ("clearly", "definitively").
    assertiveness: float = Field(ge=0.0, le=1.0)
    #: Density of first-language-transfer surface features (article drop,
    #: non-idiomatic collocations). A proxy for non-native register, and
    #: explicitly *only* a proxy — see the README limitations.
    l1_transfer: float = Field(ge=0.0, le=1.0)
    #: Free-text label for the register (formal, casual, ...). Named ``tone``
    #: rather than ``register`` because the latter shadows a Pydantic method.
    tone: str = "neutral"


class Persona(BaseModel):
    """A simulated candidate.

    ``theta_star`` is the hidden ground truth. It lives in ``data/personas/``
    and is joined in at eval time only. If you find yourself passing a Persona
    into anything under ``probe/`` other than ``probe/sim/``, stop.
    """

    id: str
    theta_star: dict[str, float]
    style: StyleProfile
    behavior: Behavior
    resume: str
    jd_id: str
    #: Competencies deliberately under-represented in the resume relative to
    #: theta_star. These are the cases gap-probing has to find.
    understated: list[str] = Field(default_factory=list)
    #: "calibration" or "eval". Split before any fitting happens.
    split: str = "eval"
    seed: int = 0

    def ability(self, competency_id: str) -> float:
        return self.theta_star.get(competency_id, 0.0)

    def public_view(self) -> dict[str, Any]:
        """Everything the interview plane is allowed to know."""
        return {"id": self.id, "resume": self.resume, "jd_id": self.jd_id}


# --------------------------------------------------------------------------
# transcript and traces
# --------------------------------------------------------------------------


class BeliefSnapshot(BaseModel):
    """Posterior summary after a turn.

    Persisted every turn so accuracy-vs-budget curves are computed post hoc
    from traces rather than by re-running interviews once per budget.
    """

    means: dict[str, float]
    sds: dict[str, float]
    entropies: dict[str, float]

    def mean(self, competency_id: str) -> float:
        return self.means[competency_id]


class Turn(BaseModel):
    run_id: str
    turn_idx: int
    question_id: str
    competency_id: str
    question_text: str
    answer: str
    grade: Grade | None
    belief_after: BeliefSnapshot
    #: Expected information gain the policy attributed to this question at
    #: selection time. None for arms that do not compute it.
    eig_at_selection: float | None = None
    selection_reason: str = ""
    elapsed_seconds: float = 0.0
    tokens_used: int = 0
    #: Set when the repair loop exhausted every fallback for this turn.
    unrecoverable: bool = False

    model_config = ConfigDict(frozen=False)


class Transcript(BaseModel):
    run_id: str
    candidate_id: str
    arm: str
    turns: list[Turn] = Field(default_factory=list)

    def __len__(self) -> int:
        return len(self.turns)

    def asked_question_ids(self) -> set[str]:
        return {t.question_id for t in self.turns}

    def families_asked(self) -> list[ProbeFamily]:
        return [ProbeFamily(t.question_id.split("::")[-1]) for t in self.turns if "::" in t.question_id]

    def render(self) -> str:
        """Plain-text rendering. Used by the byte-reconstruction test: the
        transcript rebuilt from DuckDB must render identically to the live one."""
        lines = [f"# run {self.run_id} | arm={self.arm} | candidate={self.candidate_id}"]
        for t in self.turns:
            lines.append(f"--- turn {t.turn_idx} [{t.competency_id}] ({t.question_id})")
            lines.append(f"Q: {t.question_text}")
            lines.append(f"A: {t.answer}")
            if t.grade is not None:
                flags = ",".join(sorted(f.value for f in t.grade.flags)) or "-"
                lines.append(f"G: score={t.grade.score} conf={t.grade.confidence:.3f} flags={flags}")
            else:
                lines.append("G: <unrecoverable>")
        return "\n".join(lines)


class LLMCallRecord(BaseModel):
    """One provider call, sufficient to reconstruct it exactly."""

    call_id: str
    run_id: str | None
    role: LLMRole
    prompt: str
    prompt_hash: str
    model: str
    seed: int
    temperature: float
    raw_output: str
    parsed_ok: bool
    repair_attempt: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class RunRecord(BaseModel):
    """One interview, start to finish."""

    run_id: str
    arm: str
    persona_id: str
    style_id: str
    bank_version: str
    population_version: str
    code_commit: str
    seed: int
    stop_reason: StopReason | None = None
    n_turns: int = 0
    total_tokens: int = 0
    wallclock_seconds: float = 0.0
    usd_cost: float = 0.0
    followups_enabled: bool = True
    style_separation: bool = True
    grader_model: str = "sim-grader"
    completed: bool = False
    #: Set when a budget ceiling forced early termination; the report is still
    #: emitted, flagged partial, and never raises.
    partial: bool = False


class InterviewReport(BaseModel):
    """The per-candidate deliverable of the interview plane."""

    run_id: str
    candidate_id: str
    arm: str
    per_competency: dict[str, CompetencyVerdict]
    stop_reason: StopReason
    partial: bool = False
    n_questions: int = 0
    notes: list[str] = Field(default_factory=list)


class CompetencyVerdict(BaseModel):
    competency_id: str
    posterior_mean: float
    posterior_sd: float
    #: 80% central credible interval from the grid posterior.
    ci80: tuple[float, float]
    required_level: int
    n_questions: int
    evidence: list[EvidenceSpan] = Field(default_factory=list)
    flags: list[GradeFlag] = Field(default_factory=list)
    #: True when the posterior is tight enough to act on.
    confident: bool = False


InterviewReport.model_rebuild()
