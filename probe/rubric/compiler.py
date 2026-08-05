"""JD + resume -> a compiled rubric.

Three commitments, each testable:

1. **Every emitted id exists in the taxonomy.** The compiler selects; it never
   invents. Anything the model returns that is not a known id is dropped and
   counted, not silently accepted.
2. **``evidence_in_resume`` is span-backed or zero.** A citation whose offsets
   do not resolve against the resume text is discarded and the evidence score
   goes to zero with it. Enforced by the ``Competency`` validator as well as
   here, because this is the number the priors are built from.
3. **Prior variance is a function of resume silence.** JD-required and
   resume-silent means a wide prior, which means the EIG policy attacks it
   first. That is the entire gap-probing mechanism — there is no rule anywhere
   that says "ask about gaps", and there should not be.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from probe.models import Competency, EvidenceSpan, LLMRole, ProbeFamily, Rubric
from probe.rubric.taxonomy import (
    HIGH_UNCERTAINTY_VAR,
    LOW_UNCERTAINTY_VAR,
    Taxonomy,
    load_taxonomy,
)
from probe.runtime.llm import LLMRequest
from probe.runtime.retry import structured_call

#: Prior variance below which a competency counts as "the resume told us
#: something". The compiler test asserts JD-required + resume-silent
#: competencies land above this.
HIGH_UNCERTAINTY_THRESHOLD = 1.0


class CompiledCompetency(BaseModel):
    id: str
    label: str = ""
    required_level: int = Field(default=3, ge=1, le=5)
    evidence_in_resume: float = Field(default=0.0, ge=0.0, le=1.0)
    resume_spans: list[EvidenceSpan] = Field(default_factory=list)
    probe_families: list[ProbeFamily] = Field(default_factory=list)


class CompilerOutput(BaseModel):
    competencies: list[CompiledCompetency]


def prior_from_evidence(evidence: float) -> tuple[float, float]:
    """``(prior_mean, prior_var)`` for a given span-backed evidence score.

    Interpolated rather than bucketed so the policy sees a smooth gradient of
    uncertainty: a competency with one supporting span should be a slightly
    less attractive target than one with none, not identically attractive.
    """
    mean = 0.85 * evidence - 0.20
    var = HIGH_UNCERTAINTY_VAR - (HIGH_UNCERTAINTY_VAR - LOW_UNCERTAINTY_VAR) * evidence
    return round(mean, 4), round(var, 4)


class RubricCompiler:
    def __init__(self, client, taxonomy: Taxonomy | None = None) -> None:
        self.client = client
        self.taxonomy = taxonomy or load_taxonomy()
        #: Ids the model returned that are not in the taxonomy. Surfaced rather
        #: than swallowed — a rising count means the extraction prompt has
        #: started drifting off the fixed vocabulary.
        self.rejected_ids: list[str] = []
        #: Ids that are in the taxonomy and relevant to the role, but which the
        #: question bank has no items for.
        self.unprobeable_ids: list[str] = []

    def compile(
        self,
        *,
        candidate_id: str,
        jd_text: str,
        resume: str,
        role_title: str = "Backend Engineer",
        seniority_level: int = 4,
        seed: int = 0,
        max_competencies: int = 14,
        available_competencies: set[str] | None = None,
    ) -> Rubric:
        """Compile a rubric.

        ``available_competencies`` restricts the result to competencies the
        question bank can actually probe. Without it the compiler happily
        emits an interview plan containing things no question exists for:
        those competencies are then reported as "unprobed" with the prior
        interval intact, they count against every arm's resolved fraction
        identically, and they dilute every efficiency number by a constant
        nobody notices. An interview plan you have no questions for is not a
        plan.
        """
        request = LLMRequest(
            role=LLMRole.RUBRIC_COMPILE,
            prompt=self._prompt(jd_text, resume, role_title),
            seed=seed,
            temperature=0.0,
            context={
                "jd_text": jd_text,
                "resume": resume,
                "seniority_level": seniority_level,
                "taxonomy_version": self.taxonomy.version,
            },
        )
        result = structured_call(
            self.client,
            request,
            CompilerOutput,
            degraded=lambda: self._degraded(jd_text, seniority_level),
            max_repairs=1,
        )
        raw = result.value or self._degraded(jd_text, seniority_level)
        return self._to_rubric(
            raw,
            candidate_id,
            resume,
            role_title,
            max_competencies=max_competencies,
            available_competencies=available_competencies,
        )

    # ---------------------------------------------------------------- prompt

    def _prompt(self, jd_text: str, resume: str, role_title: str) -> str:
        ids = "\n".join(f"  {n.id} — {n.label}" for n in self.taxonomy)
        return (
            "Map this job description and resume onto a FIXED competency "
            "taxonomy. You may only use ids from the list below. Do not invent "
            "ids, and do not return a competency the job description neither "
            "names nor implies.\n\n"
            f"Taxonomy (version {self.taxonomy.version}):\n{ids}\n\n"
            f"Role: {role_title}\n\n"
            f"--- JOB DESCRIPTION ---\n{jd_text}\n\n"
            f"--- RESUME ---\n{resume}\n\n"
            "For each competency return: id, required_level (1-5, from the "
            "seniority the JD signals), evidence_in_resume (0-1), and "
            "resume_spans — character offsets into the RESUME text above, each "
            "quoting text that literally appears there. If you cannot cite a "
            "span, set evidence_in_resume to 0 and return no spans. An "
            "uncited evidence score is worse than no evidence score.\n"
            'Return JSON: {"competencies": [...]}\n'
        )

    # ----------------------------------------------------------- degraded

    def _degraded(self, jd_text: str, seniority_level: int) -> CompilerOutput:
        """Keyword-only fallback when the model cannot produce valid output.

        No resume evidence at all, so every competency starts with the wide
        prior. Degrading toward *more* uncertainty is the safe direction: the
        interview asks more questions than it needed to, rather than
        confidently reporting a number it never gathered evidence for.
        """
        lowered = jd_text.lower()
        out = [
            CompiledCompetency(
                id=node.id,
                label=node.label,
                required_level=seniority_level,
                evidence_in_resume=0.0,
                probe_families=list(node.probe_families),
            )
            for node in self.taxonomy
            if node.label.lower() in lowered
            or any(kw.lower() in lowered for kw in node.jd_keywords)
        ]
        return CompilerOutput(competencies=out)

    # --------------------------------------------------------------- mapping

    def _to_rubric(
        self,
        raw: CompilerOutput,
        candidate_id: str,
        resume: str,
        role_title: str,
        max_competencies: int,
        available_competencies: set[str] | None = None,
    ) -> Rubric:
        competencies: list[Competency] = []
        self.unprobeable_ids = []
        for item in raw.competencies:
            if not self.taxonomy.has(item.id):
                self.rejected_ids.append(item.id)
                continue
            if available_competencies is not None and item.id not in available_competencies:
                # Relevant to the role, but the bank cannot ask about it.
                # Recorded rather than dropped silently: a growing list here
                # means the bank has fallen behind the roles being hired for.
                self.unprobeable_ids.append(item.id)
                continue
            node = self.taxonomy.get(item.id)

            # Discard citations that do not resolve. Keeping a span the audit
            # trail cannot verify is worse than keeping none.
            spans = [s for s in item.resume_spans if s.verify_against(resume)]
            evidence = item.evidence_in_resume if spans else 0.0
            mean, var = prior_from_evidence(evidence)

            competencies.append(
                Competency(
                    id=node.id,
                    label=node.label or item.label,
                    required_level=item.required_level,
                    evidence_in_resume=evidence,
                    prior_mean=mean,
                    prior_var=var,
                    probe_families=list(node.probe_families),
                    resume_spans=spans,
                )
            )

        # Order by what the job needs, breaking ties by taxonomy position.
        #
        # Explicitly NOT by prior variance. Sorting the rubric by uncertainty
        # and then truncating hands the policy a set that is already
        # uncertainty-ranked, which is the policy's entire job -- every arm
        # would look adaptive, including the fixed script, and the comparison
        # would measure the compiler. The rubric's job is to say what matters
        # for the role; deciding what to ask about is downstream.
        order = {cid: i for i, cid in enumerate(self.taxonomy.ids)}
        competencies.sort(key=lambda c: (-c.required_level, order[c.id]))
        return Rubric(
            candidate_id=candidate_id,
            role_title=role_title,
            competencies=competencies[:max_competencies],
            taxonomy_version=self.taxonomy.version,
        )
