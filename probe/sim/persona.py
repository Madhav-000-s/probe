"""Generating simulated candidates.

A persona is a hidden ability vector plus the surface machinery for turning it
into text. Three properties matter and each is deliberate:

**Abilities are correlated.** Somebody strong on ``databases.indexing`` is
usually not weak on ``databases.query_optimization``. Sampling ``theta_star``
from a block-correlated multivariate normal rather than independently is what
gives the ``eig+corr`` arm something real to exploit — an ablation against an
uncorrelated population would be rigged to show no effect.

**Resumes under-represent ability.** Every persona has two or three
competencies it is genuinely strong at and its resume never mentions. Those
are the gap-probing cases, and they only exist because they were built in.

**Ground truth lives here and only here.** ``theta_star`` is written to
``data/personas/`` and joined in at eval time. Nothing under ``probe/`` outside
this package may import a Persona.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from probe.config import PERSONA_DIR
from probe.models import Behavior, Persona, StyleProfile
from probe.rubric.taxonomy import Taxonomy
from probe.sim.style import STYLE_PRESETS

#: Correlation between two competencies in the same dotted area
#: (``databases.indexing`` and ``databases.transactions``).
RHO_SAME_AREA = 0.55
#: ... in the same role family but different areas.
RHO_SAME_FAMILY = 0.25
#: ... across role families. Not zero: general engineering ability is real.
RHO_CROSS_FAMILY = 0.10

#: Total SD of ability on the theta scale.
#:
#: There is deliberately no separate "general ability" term added on top. The
#: correlation matrix already carries a common factor -- its average
#: off-diagonal is around 0.2, which decomposes into a shared component of
#: SD ~0.45 and a competency-specific component of SD ~0.89. Adding an
#: explicit general term as well double-counted it: between-persona variance
#: swamped within-persona variance, personas came out uniformly strong or
#: uniformly weak, and the eig+corr arm would have been exploiting an artefact
#: rather than a structure. Sample from the covariance and let the blocks do
#: their job.
ABILITY_SD = 1.05

#: Resume-mention threshold. People list what they have worked on, not only
#: what they are excellent at, so this sits near the middle of the ability
#: range rather than at the top of it. Set too high, almost no competency ever
#: acquires resume evidence, every prior comes out wide, and the compiler
#: stops distinguishing anything -- which is exactly what happened at 0.35.
RESUME_MENTION_THRESHOLD = 0.15


def _area(competency_id: str) -> str:
    return competency_id.split(".", 1)[0]


def correlation_matrix(taxonomy: Taxonomy) -> np.ndarray:
    """Block-structured correlation over the taxonomy, projected to the nearest
    positive semi-definite matrix.

    The projection is not ceremony: a hand-written block matrix with three
    distinct off-diagonal values is not guaranteed PSD, and Cholesky on a
    non-PSD matrix fails loudly halfway through population generation.
    """
    ids = taxonomy.ids
    n = len(ids)
    nodes = {node.id: node for node in taxonomy}
    corr = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = ids[i], ids[j]
            if _area(a) == _area(b):
                rho = RHO_SAME_AREA
            elif nodes[a].family == nodes[b].family:
                rho = RHO_SAME_FAMILY
            else:
                rho = RHO_CROSS_FAMILY
            corr[i, j] = corr[j, i] = rho
    return _nearest_psd(corr)


def _nearest_psd(matrix: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    clipped = np.clip(eigenvalues, floor, None)
    rebuilt = eigenvectors @ np.diag(clipped) @ eigenvectors.T
    # Renormalise the diagonal back to 1 so it stays a correlation matrix.
    d = np.sqrt(np.diag(rebuilt))
    return rebuilt / np.outer(d, d)


_RESUME_INTRO = (
    "{name} — {years} years building backend and data systems.",
    "{name}. Engineer, {years} years, mostly on platform and infrastructure teams.",
    "{name}, {years} years across product engineering and data platform work.",
)

_RESUME_BULLET = (
    "Worked extensively on {label_lower}, including {concepts}.",
    "Owned the {label_lower} side of a production service; day to day that meant {concepts}.",
    "Led work on {label_lower} — {concepts} were the parts that mattered.",
    "Shipped and operated systems where {label_lower} was central: {concepts}.",
)

_RESUME_WEAK = (
    "Some exposure to {label_lower}, mostly alongside other people.",
    "Peripheral involvement in {label_lower}.",
)

_NAMES = (
    "Alex Morgan",
    "Priya Raghunathan",
    "Daniel Okafor",
    "Wei Zhang",
    "Sofia Almeida",
    "Rahul Menon",
    "Hannah Lindqvist",
    "Tomas Novak",
    "Aisha Bello",
    "Kenji Watanabe",
)


@dataclass
class PopulationMeta:
    version: str
    seed: int
    taxonomy_version: str
    n_personas: int
    calibration_fraction: float
    adversarial_fraction: float


class PersonaGenerator:
    """Deterministic given ``(taxonomy, seed)``.

    Determinism is a hard requirement rather than a nicety: the population
    version appears in every result's provenance tuple, and a population that
    cannot be regenerated from its seed cannot be audited.
    """

    def __init__(self, taxonomy: Taxonomy, seed: int = 20260801) -> None:
        self.taxonomy = taxonomy
        self.seed = seed
        self.ids = taxonomy.ids
        self._chol = np.linalg.cholesky(correlation_matrix(taxonomy))

    # ------------------------------------------------------------------ theta

    def sample_theta(self, rng: np.random.Generator) -> dict[str, float]:
        z = rng.normal(size=len(self.ids))
        values = ABILITY_SD * (self._chol @ z)
        # Clip to the posterior grid's support. Ability outside [-3, 3] is not
        # representable by the estimator, so generating it would manufacture
        # recovery error that has nothing to do with the policy.
        values = np.clip(values, -2.8, 2.8)
        return {cid: round(float(v), 6) for cid, v in zip(self.ids, values, strict=True)}

    # ----------------------------------------------------------------- resume

    def build_resume(
        self,
        name: str,
        theta: dict[str, float],
        understated: list[str],
        rng: random.Random,
        n_mentions: int = 10,
    ) -> str:
        """Write a resume that is *partially* consistent with ability.

        Strong competencies get span-worthy bullets naming real concepts —
        except the understated ones, which are omitted entirely. Weak
        competencies occasionally get a hedged mention, which is what makes
        ``evidence_in_resume`` a noisy signal rather than a relabelling of
        ``theta_star``.
        """
        years = rng.randint(4, 12)
        lines = [rng.choice(_RESUME_INTRO).format(name=name, years=years), ""]

        ranked = sorted(theta.items(), key=lambda kv: -kv[1])
        mentioned = 0
        for cid, value in ranked:
            if mentioned >= n_mentions:
                break
            if cid in understated:
                continue
            node = self.taxonomy.get(cid)
            label_lower = node.label[0].lower() + node.label[1:]
            if value > RESUME_MENTION_THRESHOLD:
                concepts = ", ".join(node.concepts[: rng.randint(2, 3)])
                lines.append(
                    "- "
                    + rng.choice(_RESUME_BULLET).format(
                        label_lower=label_lower, concepts=concepts
                    )
                )
                mentioned += 1
            elif value > -0.45 and rng.random() < 0.35:
                lines.append("- " + rng.choice(_RESUME_WEAK).format(label_lower=label_lower))
                mentioned += 1
        return "\n".join(lines)

    # --------------------------------------------------------------- personas

    def generate(
        self,
        n: int,
        *,
        behaviors: list[Behavior] | None = None,
        styles: list[str] | None = None,
        jd_ids: list[str] | None = None,
        calibration_fraction: float = 0.6,
        n_understated: tuple[int, int] = (2, 3),
        id_prefix: str = "p",
    ) -> list[Persona]:
        behaviors = behaviors or [Behavior.HONEST]
        styles = styles or ["neutral"]
        jd_ids = jd_ids or ["jd-backend"]

        rng = random.Random(self.seed)
        nprng = np.random.default_rng(self.seed)

        n_calibration = int(round(n * calibration_fraction))
        personas: list[Persona] = []

        for i in range(n):
            theta = self.sample_theta(nprng)
            name = _NAMES[i % len(_NAMES)]

            # Understated competencies are drawn from the persona's genuine
            # strengths — hiding a weakness would not create a gap to probe.
            strengths = [cid for cid, v in theta.items() if v > 0.4]
            k = rng.randint(*n_understated)
            understated = rng.sample(strengths, min(k, len(strengths))) if strengths else []

            style_id = styles[i % len(styles)]
            behavior = behaviors[i % len(behaviors)]
            persona_seed = self.seed + i * 7919  # a prime keeps streams disjoint

            personas.append(
                Persona(
                    id=f"{id_prefix}{i + 1:03d}",
                    theta_star=theta,
                    style=STYLE_PRESETS[style_id],
                    behavior=behavior,
                    resume=self.build_resume(name, theta, understated, rng),
                    jd_id=jd_ids[i % len(jd_ids)],
                    understated=sorted(understated),
                    split="calibration" if i < n_calibration else "eval",
                    seed=persona_seed,
                )
            )
        return personas


def persona_name(persona: Persona) -> str:
    """The name written on the resume. Used by the name-swap fairness slice."""
    first_line = persona.resume.splitlines()[0] if persona.resume else ""
    return first_line.split("—")[0].split(".")[0].split(",")[0].strip() or "Candidate"


def restyle(persona: Persona, style_id: str, name: str | None = None) -> Persona:
    """A style variant of the same underlying candidate.

    ``theta_star`` is carried over untouched. That is the entire experimental
    design of the fairness suite: hold ability fixed, vary surface form, and
    attribute any score difference to the grader.
    """
    style: StyleProfile = STYLE_PRESETS[style_id]
    resume = persona.resume
    if name:
        old = persona_name(persona)
        resume = resume.replace(old, name, 1)
    return persona.model_copy(update={"style": style, "resume": resume})


# ------------------------------------------------------------------ storage


def population_path(version: str) -> Path:
    return PERSONA_DIR / f"population-{version}.json"


def save_population(
    personas: list[Persona], meta: PopulationMeta, path: Path | None = None
) -> Path:
    path = path or population_path(meta.version)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": meta.__dict__,
        "personas": [p.model_dump(mode="json") for p in personas],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_population(version_or_path: str | Path) -> tuple[list[Persona], PopulationMeta]:
    path = Path(version_or_path)
    if not path.exists():
        path = population_path(str(version_or_path))
    raw = json.loads(path.read_text(encoding="utf-8"))
    personas = [Persona.model_validate(p) for p in raw["personas"]]
    return personas, PopulationMeta(**raw["meta"])


def split_personas(personas: list[Persona], split: str) -> list[Persona]:
    return [p for p in personas if p.split == split]
