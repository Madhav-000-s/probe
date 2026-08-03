"""Pre-grader classifiers.

Injection defence is a stack, not a prompt. Three layers, in order of how much
they can be trusted:

1. **The schema** — ``Grade`` has no field that could express "score
   everything 5". Nothing an attacker writes can create one.
2. **This module** — a deterministic classifier that runs *before* the grader
   is called and flags instruction-like content. No model in the loop, so
   nothing to talk it out of its verdict.
3. **The prompt** — untrusted-input delimiters and an instruction to treat the
   contents as data. Weakest layer, listed last on purpose.

The Phase 5 robustness suite measures whether the stack holds, rather than
asserting that it does.
"""

from __future__ import annotations

import re

from probe.models import GradeFlag

#: Instruction-shaped patterns. Tuned to catch imperative overrides aimed at
#: the grader, not ordinary technical prose — "we ignore stale replicas" must
#: not trip this, or the false-positive rate eats the metric.
INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|rules?)\b",
        r"\bdisregard\s+(the\s+)?(rubric|anchors?|instructions?|scoring)\b",
        r"\b(system|admin|administrator)\s*:\s*",
        r"\b(assign|give|return|record|award)\b[^.]{0,40}\b(score|rating)\b[^.]{0,20}\b(5|five|maximum|max)\b",
        r"\bscore\s*=\s*[0-9]",
        r"\bconfidence\s*=\s*[0-9]",
        r"\bstrong\s+hire\b",
        r"</?(answer|system|instructions?|prompt)>",
        r"\bpre-?approved\b.{0,30}\b(grader|hiring|manager)\b",
        r"\bgrading\s+override\b",
        r"\bnote\s+to\s+the\s+(automated\s+)?grader\b",
        r"\bacting\s+as\s+the\s+(evaluation\s+)?supervisor\b",
        r"\bupdated\s+rubric\b",
    )
)

#: Phrases that route around a question instead of answering it.
DEFLECTION_MARKERS: tuple[str, ...] = (
    "depends on the specifics",
    "need to look at the actual code",
    "hard to answer that in the abstract",
    "every place i have worked did this differently",
    "want to talk to whoever owns",
    "have not run into that specific situation",
)

#: Unverifiable seniority assertions. Not wrong in themselves — flagged so the
#: grader is not quietly swayed by them, and so overclaim recall is measurable.
OVERCLAIM_MARKERS: tuple[str, ...] = (
    "i led the team",
    "i was the person other teams escalated",
    "i have owned systems at a scale",
    "i rewrote our entire platform",
)

#: Minimum words before an answer stops being treated as a non-answer.
NON_ANSWER_WORDS = 8


def detect_injection(text: str) -> list[str]:
    """Return the patterns that matched, for the audit trail.

    Returning the evidence rather than a bare boolean matters: an injection
    number without the matched strings is unfalsifiable, and the Phase 5 suite
    greps transcripts to confirm payloads were genuinely present.
    """
    return [p.pattern for p in INJECTION_PATTERNS if p.search(text)]


def is_injection(text: str) -> bool:
    return bool(detect_injection(text))


def has_deflection(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in DEFLECTION_MARKERS)


def has_overclaim(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in OVERCLAIM_MARKERS)


def classify(text: str, concepts_matched: int = 0) -> list[GradeFlag]:
    """Deterministic pre-grader flags.

    These are merged with whatever the grader itself returns; a flag raised
    here cannot be argued away by the model, which is the point of running it
    first.
    """
    flags: list[GradeFlag] = []
    if is_injection(text):
        flags.append(GradeFlag.INJECTION_ATTEMPT)
    words = len(text.split())
    if words < NON_ANSWER_WORDS or (has_deflection(text) and concepts_matched == 0):
        flags.append(GradeFlag.NON_ANSWER)
    if has_overclaim(text):
        flags.append(GradeFlag.UNSUPPORTED_CLAIM)
    return flags


def sanitise_for_grading(text: str) -> str:
    """Neutralise matched injection spans while preserving offsets.

    Same length in, same length out — every character replaced one-for-one —
    because evidence spans are character offsets into the *original* answer.
    A sanitiser that shortened the text would invalidate every span the grader
    produced, and the audit trail would silently start pointing at the wrong
    words.
    """
    out = text
    for pattern in INJECTION_PATTERNS:
        out = pattern.sub(lambda m: "█" * len(m.group(0)), out)
    assert len(out) == len(text), "sanitiser must be length-preserving"
    return out
