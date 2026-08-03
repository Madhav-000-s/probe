"""Reading an answer the way a model would.

Two extractors, and the distinction between them is the whole of Q3:

* :func:`match_concepts` recovers **content** — which ideas the answer names.
* :func:`style_features` recovers **surface** — how long it is, how hedged, how
  assertive, how idiomatic.

A grader that scores on the first is measuring the candidate. A grader that
also scores on the second is measuring their prose, and the fairness suite
exists to put a number on how much of that is happening.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from probe.sim.style import ASSERTIONS, HEDGES

_L1_MARKERS = ("because of that", "for that", "for to", "that is", "have to must")


@dataclass(frozen=True)
class ConceptMatch:
    concept: str
    start: int
    end: int

    @property
    def text_slice(self) -> slice:
        return slice(self.start, self.end)


def match_concepts(answer: str, concepts: list[str]) -> list[ConceptMatch]:
    """Exact, case-insensitive concept lookup with character offsets.

    Exact matching is a deliberate limitation, not an oversight. It is what
    makes the first-language-transfer paraphrase cost a candidate marks they
    should have earned, which is a real fairness failure mode with a real
    remedy (fuzzy or embedding matching) that this version does not implement.
    The README names it as the residual open bug rather than quietly fixing it
    and losing the finding.
    """
    lowered = answer.lower()
    out: list[ConceptMatch] = []
    for concept in concepts:
        idx = lowered.find(concept.lower())
        if idx >= 0:
            out.append(ConceptMatch(concept=concept, start=idx, end=idx + len(concept)))
    return sorted(out, key=lambda m: m.start)


def count_offpool_concepts(answer: str, pool: list[str], world: list[str]) -> int:
    """Technical vocabulary from *other* competencies.

    This is the bluffer's signature: borrowed terminology that sounds like
    domain fluency but does not answer the question asked.
    """
    in_pool = {c.lower() for c in pool}
    lowered = answer.lower()
    return sum(1 for c in world if c.lower() not in in_pool and c.lower() in lowered)


@dataclass(frozen=True)
class StyleFeatures:
    words: int
    sentences: int
    hedge_density: float
    assertive_density: float
    l1_density: float

    @property
    def length_z(self) -> float:
        """Standardised length, clipped. 55 words is roughly a competent
        answer at this rubric's level 3."""
        return max(-1.5, min(1.5, (self.words - 55) / 45.0))


def style_features(answer: str) -> StyleFeatures:
    words = len(answer.split())
    sentences = max(1, len([s for s in re.split(r"[.!?]+", answer) if s.strip()]))
    lowered = answer.lower()
    hedges = sum(1 for h in HEDGES if h.lower() in lowered)
    assertions = sum(1 for a in ASSERTIONS if a.lower().rstrip(":") in lowered)
    l1 = sum(1 for m in _L1_MARKERS if m in lowered)
    return StyleFeatures(
        words=words,
        sentences=sentences,
        hedge_density=hedges / sentences,
        assertive_density=assertions / sentences,
        l1_density=l1 / sentences,
    )


def level_from_matches(n_matched: int, thresholds: dict[int, int]) -> int:
    """Highest rubric level whose concept requirement is satisfied."""
    level = 1
    for candidate in sorted(thresholds):
        if n_matched >= thresholds[candidate]:
            level = candidate
    return level
