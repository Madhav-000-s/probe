"""Surface style, and the one invariant that makes the fairness suite mean
anything.

Style is applied to *prose segments only*. An answer is assembled as a list of
segments, each tagged either ``CONCEPT`` (a phrase from the competency's
concept pool, which is the content the grader is supposed to be measuring) or
``PROSE`` (everything else). Style transformations touch only ``PROSE``.

That is a structural guarantee, not a convention: it means two style variants
of the same answer contain exactly the same concepts, so any score difference
between them is grader drift by construction. Without it, "style drift" would
be confounded with a real content difference and Q3 would be unanswerable.

The one deliberate exception is first-language transfer, which *paraphrases*
concept phrases at a controlled rate — see :func:`paraphrase_concept`. That is
not a leak in the invariant; it is the mechanism behind the residual drift the
content–style intervention cannot fix, and it is measured separately.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import StrEnum

from probe.models import StyleProfile


class SegmentKind(StrEnum):
    #: A phrase from the competency's concept pool. Never dropped, never
    #: reworded by a style transform. This is the content being measured.
    CONCEPT = "concept"
    #: Connective tissue. Styled, but kept — removing it would leave concepts
    #: floating without a sentence around them.
    PROSE = "prose"
    #: Padding. The only thing verbosity adds and terseness removes.
    FILLER = "filler"


@dataclass
class Segment:
    kind: SegmentKind
    text: str


#: Style variants used in the main policy sweep (four per persona, as planned).
MAIN_SWEEP_STYLES = ("neutral", "verbose", "terse", "l1_transfer")

#: The full slice set used by the fairness suite, in contrast pairs. Pairs
#: rather than a flat list because a drift number is only interpretable
#: against its counterpart.
FAIRNESS_PAIRS = (
    ("verbose", "terse"),
    ("neutral", "l1_transfer"),
    ("hedged", "assertive"),
    ("name_a", "name_b"),
)

STYLE_PRESETS: dict[str, StyleProfile] = {
    "neutral": StyleProfile(
        id="neutral", verbosity=1.0, hedging=0.15, assertiveness=0.15, l1_transfer=0.0
    ),
    "verbose": StyleProfile(
        id="verbose", verbosity=2.1, hedging=0.20, assertiveness=0.20, l1_transfer=0.0,
        tone="expansive",
    ),
    "terse": StyleProfile(
        id="terse", verbosity=0.35, hedging=0.05, assertiveness=0.10, l1_transfer=0.0,
        tone="clipped",
    ),
    "l1_transfer": StyleProfile(
        id="l1_transfer", verbosity=1.0, hedging=0.25, assertiveness=0.10, l1_transfer=0.75,
        tone="non-native",
    ),
    "hedged": StyleProfile(
        id="hedged", verbosity=1.1, hedging=0.75, assertiveness=0.02, l1_transfer=0.0,
        tone="tentative",
    ),
    "assertive": StyleProfile(
        id="assertive", verbosity=1.1, hedging=0.02, assertiveness=0.75, l1_transfer=0.0,
        tone="confident",
    ),
    # The name-swap pair is identical in every measurable dimension. Anything
    # but exact score equality between them is a bug, not a drift statistic.
    "name_a": StyleProfile(
        id="name_a", verbosity=1.0, hedging=0.15, assertiveness=0.15, l1_transfer=0.0
    ),
    "name_b": StyleProfile(
        id="name_b", verbosity=1.0, hedging=0.15, assertiveness=0.15, l1_transfer=0.0
    ),
}

#: Names attached to the name-swap slice. Chosen from different name origins
#: precisely because that is the axis the slice is probing.
SWAP_NAMES = {"name_a": "Alex Morgan", "name_b": "Priya Raghunathan"}

HEDGES = (
    "I think",
    "if I remember correctly",
    "I could be wrong here but",
    "probably",
    "my sense is that",
    "something like",
)

ASSERTIONS = (
    "clearly",
    "without question",
    "the answer is straightforward:",
    "definitively",
    "obviously",
)

FILLER = (
    "and that has bitten me before",
    "which is the part people usually skip",
    "at least in the systems I have worked on",
    "though it depends on the traffic shape",
    "and the same reasoning applies at the next layer down",
    "which is worth checking early rather than late",
)

#: Rendered as prose only. Article dropping and non-idiomatic connectives are
#: the two most legible markers and both are purely surface.
_L1_ARTICLE_DROPS = (" the ", " a ", " an ")
_L1_CONNECTIVES = {
    "because": "because of",
    "so that": "for that",
    "in order to": "for to",
    "which is": "that is",
    "you need to": "you have to must",
}


def style_by_id(style_id: str) -> StyleProfile:
    try:
        return STYLE_PRESETS[style_id]
    except KeyError as exc:
        raise KeyError(f"unknown style {style_id!r}; have {sorted(STYLE_PRESETS)}") from exc


def paraphrase_concept(phrase: str, rng: random.Random) -> str:
    """Render a concept the way a fluent non-native speaker often would.

    Not noise for its own sake. An exact-match extractor misses these, so a
    candidate who *knows* the concept scores lower for reasons that have
    nothing to do with knowing it. That is a real fairness failure, it survives
    the content–style intervention (which removes the fluency *reward*, not the
    recognition *failure*), and the Phase 5 suite reports it as the residual
    open bug rather than pretending the intervention closed it.
    """
    words = phrase.split()
    choice = rng.random()
    if "-" in phrase and choice < 0.5:
        return phrase.replace("-", " ")
    if len(words) == 2 and choice < 0.75:
        return f"{words[1]} of {words[0]}"
    if len(words) >= 2:
        return " ".join(words[::-1])
    return phrase + "s"


def apply_l1_prose(text: str, intensity: float, rng: random.Random) -> str:
    """Surface-level first-language transfer, prose only."""
    if intensity <= 0:
        return text
    out = text
    for phrase, replacement in _L1_CONNECTIVES.items():
        if phrase in out and rng.random() < intensity:
            out = out.replace(phrase, replacement, 1)
    for article in _L1_ARTICLE_DROPS:
        while article in out and rng.random() < intensity * 0.5:
            out = out.replace(article, " ", 1)
    return out


def decorate_prose(text: str, style: StyleProfile, rng: random.Random) -> str:
    """Apply hedging, assertiveness and L1 transfer to one prose segment."""
    if rng.random() < style.hedging:
        text = f"{rng.choice(HEDGES)}, {text}"
    if rng.random() < style.assertiveness:
        text = f"{rng.choice(ASSERTIONS)} {text}"
    return apply_l1_prose(text, style.l1_transfer, rng)


def filler_count(style: StyleProfile) -> int:
    """How many padding clauses this style adds.

    Verbosity moves length without moving content, which is exactly the lever
    Q3 needs: if scores track this number, the grader is rewarding volume.
    """
    return max(0, round((style.verbosity - 1.0) * 2.5))


def render(segments: list[Segment], style: StyleProfile, rng: random.Random) -> str:
    """Assemble segments into an answer.

    Terseness drops FILLER and nothing else. An earlier version clipped
    trailing *sentences*, which quietly deleted concepts from terse answers —
    the fidelity gate caught it as a correlation shortfall. Content loss caused
    by a style transform is precisely the confound the whole content/style
    split exists to prevent, so terseness now removes padding by construction
    rather than by a heuristic that happened to usually hit padding.
    """
    parts: list[str] = []
    for seg in segments:
        if seg.kind is SegmentKind.CONCEPT:
            parts.append(seg.text)
        elif seg.kind is SegmentKind.FILLER:
            if not is_terse(style):
                parts.append(decorate_prose(seg.text, style, rng))
        else:
            parts.append(decorate_prose(seg.text, style, rng))
    return _tidy(" ".join(p for p in parts if p).strip())


def is_terse(style: StyleProfile) -> bool:
    return style.verbosity < 0.6


def _tidy(text: str) -> str:
    out = " ".join(text.split())
    out = out.replace(" ,", ",").replace(" .", ".").replace("..", ".")
    if out and out[0].islower():
        out = out[0].upper() + out[1:]
    if out and not out.endswith((".", "!", "?")):
        out += "."
    return out
