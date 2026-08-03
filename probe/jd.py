"""Synthetic job descriptions.

A JD is an input to the *interview* plane — the compiler reads it, and it
carries no ground truth — so these are ordinary fixtures rather than hidden
state. They exist so the compiler has something realistic to map onto the
taxonomy: prose that names some competencies explicitly, implies others
through keywords, and stays silent about the rest.

That silence is load-bearing. A competency the JD requires and the resume
never mentions is where gap-probing has to happen, and you cannot construct
that case without a JD that asks for more than the resume answers.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from probe.config import JD_DIR
from probe.rubric.taxonomy import Taxonomy


@dataclass
class JobDescription:
    id: str
    title: str
    seniority: str
    text: str
    #: Competency ids the JD genuinely asks for. Kept alongside the prose so
    #: compiler tests can assert recall against a known answer instead of
    #: eyeballing extraction quality.
    required: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "seniority": self.seniority,
            "text": self.text,
            "required": list(self.required),
        }


_SENIORITY_LEVELS = {
    "mid": 3,
    "senior": 4,
    "staff": 5,
}

_INTRO = (
    "We are hiring a {seniority} {title} to work on systems that are already in "
    "production and already load-bearing. You will own services end to end: "
    "design, ship, operate, and answer the pager for them."
)

_RESPONSIBILITY = (
    "You will be expected to reason carefully about {topic} — {keywords}."
)

_CLOSING = (
    "We care much more about how you think through an unfamiliar problem than "
    "about which frameworks appear on your CV. Expect the interview to be "
    "technical and specific."
)


def generate_jd(
    taxonomy: Taxonomy,
    jd_id: str,
    title: str = "Backend Engineer",
    seniority: str = "senior",
    n_required: int = 12,
    family: str | None = None,
    seed: int = 0,
) -> JobDescription:
    rng = random.Random(seed)
    pool = [n for n in taxonomy if family is None or n.family.value == family]
    required_nodes = rng.sample(pool, min(n_required, len(pool)))

    paragraphs = [_INTRO.format(seniority=seniority, title=title)]
    for node in required_nodes:
        keywords = ", ".join(node.jd_keywords[:3]) if node.jd_keywords else node.label.lower()
        paragraphs.append(
            _RESPONSIBILITY.format(topic=node.label.lower(), keywords=keywords)
        )
    paragraphs.append(_CLOSING)

    return JobDescription(
        id=jd_id,
        title=title,
        seniority=seniority,
        text="\n\n".join(paragraphs),
        required=[n.id for n in required_nodes],
    )


def seniority_level(seniority: str) -> int:
    """JD seniority -> the required level a matched competency inherits."""
    return _SENIORITY_LEVELS.get(seniority, 3)


def save_jd(jd: JobDescription, path: Path | None = None) -> Path:
    path = path or JD_DIR / f"{jd.id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jd.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path


def load_jd(jd_id: str, directory: Path | None = None) -> JobDescription:
    path = (directory or JD_DIR) / f"{jd_id}.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    return JobDescription(**raw)


def default_jds(taxonomy: Taxonomy, seed: int = 20260801) -> list[JobDescription]:
    """The two role families the taxonomy covers, one JD each."""
    return [
        generate_jd(
            taxonomy,
            "jd-backend",
            title="Backend Engineer",
            seniority="senior",
            n_required=12,
            family="backend",
            seed=seed,
        ),
        generate_jd(
            taxonomy,
            "jd-data-ml",
            title="Machine Learning Engineer",
            seniority="senior",
            # 12 + 8 = 20 competencies across the two roles, which is exactly
            # what the 60-item starter bank covers at three items each.
            n_required=8,
            family="data_ml",
            seed=seed + 1,
        ),
    ]
