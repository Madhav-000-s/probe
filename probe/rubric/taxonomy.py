"""The fixed competency taxonomy.

Loaded once and cached. Everything that needs to know what competencies exist
— the bank generator, the persona sampler, the compiler, the eval harness —
goes through here, so there is exactly one answer to "what are the ids".
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from probe.config import TAXONOMY_PATH
from probe.models import Competency, ProbeFamily, RoleFamily, Rubric, TaxonomyNode

#: Prior variance assigned to a competency the job requires but the resume is
#: silent about. Wide enough that the EIG policy will always prefer it to a
#: competency the resume already evidences. This single constant is the entire
#: gap-probing mechanism; nothing anywhere says "ask about gaps".
HIGH_UNCERTAINTY_VAR = 1.44  # sd = 1.2
#: Prior variance when the resume gives strong, span-backed evidence.
LOW_UNCERTAINTY_VAR = 0.36  # sd = 0.6


@dataclass(frozen=True)
class Taxonomy:
    version: str
    nodes: tuple[TaxonomyNode, ...]

    def __len__(self) -> int:
        return len(self.nodes)

    def __iter__(self):
        return iter(self.nodes)

    @property
    def ids(self) -> list[str]:
        return [n.id for n in self.nodes]

    def get(self, node_id: str) -> TaxonomyNode:
        for n in self.nodes:
            if n.id == node_id:
                return n
        raise KeyError(f"{node_id} is not in taxonomy {self.version}")

    def has(self, node_id: str) -> bool:
        return any(n.id == node_id for n in self.nodes)

    def by_family(self, family: RoleFamily) -> list[TaxonomyNode]:
        return [n for n in self.nodes if n.family == family]

    def validate(self) -> list[str]:
        """Structural problems, as a list of human-readable strings.

        Returned rather than raised so the CLI can print all of them at once
        instead of one per run.
        """
        problems: list[str] = []
        seen: set[str] = set()
        for n in self.nodes:
            if n.id in seen:
                problems.append(f"duplicate id {n.id}")
            seen.add(n.id)
            if "." not in n.id:
                problems.append(f"{n.id}: ids must be dotted (area.topic)")
            if len(n.concepts) < 6:
                problems.append(f"{n.id}: only {len(n.concepts)} concepts, need >= 6")
            if len(set(n.concepts)) != len(n.concepts):
                problems.append(f"{n.id}: duplicate concepts")
            if not n.probe_families:
                problems.append(f"{n.id}: no probe families")
        if not 30 <= len(self.nodes) <= 50:
            problems.append(f"taxonomy has {len(self.nodes)} nodes, spec says 30-50")
        return problems

    # ------------------------------------------------------------------------

    def stub_rubric(
        self,
        candidate_id: str = "stub",
        n: int = 6,
        role_title: str = "Backend Engineer",
        high_uncertainty: tuple[str, ...] = (),
    ) -> Rubric:
        """A deterministic rubric with no LLM involved.

        Phase 0 needs something for the loop to interview against before the
        compiler exists, and every later phase needs a fixture whose priors are
        known exactly. Competencies listed in ``high_uncertainty`` get the wide
        prior; the rest alternate so a test rubric always contains both kinds.
        """
        chosen = self.nodes[:n]
        comps: list[Competency] = []
        for i, node in enumerate(chosen):
            wide = node.id in high_uncertainty or (not high_uncertainty and i % 2 == 0)
            evidence = 0.0 if wide else 0.7
            spans = []
            if evidence > 0:
                text = node.label
                spans = [{"start": 0, "end": len(text), "text": text}]
            comps.append(
                Competency(
                    id=node.id,
                    label=node.label,
                    required_level=node.default_required_level,
                    evidence_in_resume=evidence,
                    prior_mean=0.0 if wide else 0.3,
                    prior_var=HIGH_UNCERTAINTY_VAR if wide else LOW_UNCERTAINTY_VAR,
                    probe_families=list(node.probe_families),
                    resume_spans=spans,
                )
            )
        return Rubric(
            candidate_id=candidate_id,
            role_title=role_title,
            competencies=comps,
            taxonomy_version=self.version,
        )


def _parse(raw: dict) -> Taxonomy:
    nodes = tuple(
        TaxonomyNode(
            id=n["id"],
            label=n["label"],
            family=RoleFamily(n["family"]),
            probe_families=[ProbeFamily(p) for p in n["probe_families"]],
            concepts=list(n["concepts"]),
            jd_keywords=list(n.get("jd_keywords", [])),
            default_required_level=int(n.get("default_required_level", 3)),
        )
        for n in raw["nodes"]
    )
    return Taxonomy(version=str(raw["version"]), nodes=nodes)


@lru_cache(maxsize=4)
def load_taxonomy(path: str | Path | None = None) -> Taxonomy:
    p = Path(path) if path else TAXONOMY_PATH
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    return _parse(raw)
