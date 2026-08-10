"""Fairness: how much does surface style move a score at fixed ability?

The design is what makes this measurable. Every style variant of a persona
carries the *same* ``theta_star`` and, by construction in
:mod:`probe.sim.style`, the same concepts in its answers. So a score difference
between two variants cannot be a content difference — it is the grader
responding to prose.

Reported per contrast pair rather than as a single number, because the axes
behave differently and averaging them hides the one that matters:

* verbose vs terse — is length being rewarded?
* neutral vs L1-transfer — is non-native phrasing being penalised?
* hedged vs assertive — is confidence being mistaken for competence?
* name_a vs name_b — a hard equality check, not a drift statistic.

Adverse impact is reported in the four-fifths format because that is the form
the question is usually asked in, not because a simulation can establish
anything about a protected class. The README says so.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from evals.metrics.loader import PersonaRuns
from probe.sim.style import FAIRNESS_PAIRS

#: Selection threshold used for the adverse-impact ratio: posterior mean at or
#: above this counts as "would advance".
ADVANCE_THRESHOLD = 0.0
#: Below this, the four-fifths rule flags a disparity.
FOUR_FIFTHS = 0.80


@dataclass
class SliceDrift:
    slice_a: str
    slice_b: str
    #: Mean posterior-mean difference (a - b) over paired competencies.
    drift: float
    #: Same, in absolute value — the headline "how much does style move it".
    abs_drift: float
    n_pairs: int
    max_drift_competency: str | None = None
    max_drift: float = 0.0
    advance_rate_a: float = float("nan")
    advance_rate_b: float = float("nan")

    @property
    def adverse_impact_ratio(self) -> float:
        """Four-fifths format: the lower advance rate over the higher."""
        rates = [self.advance_rate_a, self.advance_rate_b]
        if not all(np.isfinite(r) for r in rates) or max(rates) <= 0:
            return float("nan")
        return min(rates) / max(rates)

    @property
    def flags_disparity(self) -> bool:
        ratio = self.adverse_impact_ratio
        return bool(np.isfinite(ratio) and ratio < FOUR_FIFTHS)

    def to_dict(self) -> dict:
        return {
            "slice_a": self.slice_a,
            "slice_b": self.slice_b,
            "drift": round(self.drift, 4),
            "abs_drift": round(self.abs_drift, 4),
            "n_pairs": self.n_pairs,
            "max_drift_competency": self.max_drift_competency,
            "max_drift": round(self.max_drift, 4),
            "adverse_impact_ratio": round(self.adverse_impact_ratio, 4)
            if np.isfinite(self.adverse_impact_ratio)
            else None,
            "flags_disparity": self.flags_disparity,
        }


@dataclass
class FairnessReport:
    style_separation: bool
    slices: list[SliceDrift] = field(default_factory=list)

    @property
    def mean_abs_drift(self) -> float:
        values = [s.abs_drift for s in self.slices if np.isfinite(s.abs_drift)]
        return float(np.mean(values)) if values else float("nan")

    @property
    def worst(self) -> SliceDrift | None:
        finite = [s for s in self.slices if np.isfinite(s.abs_drift)]
        return max(finite, key=lambda s: s.abs_drift) if finite else None

    def to_dict(self) -> dict:
        return {
            "style_separation": self.style_separation,
            "mean_abs_drift": round(self.mean_abs_drift, 4),
            "worst_slice": self.worst.to_dict() if self.worst else None,
            "slices": [s.to_dict() for s in self.slices],
        }


def _paired_means(
    units: Sequence[PersonaRuns], arm: str, style_a: str, style_b: str
) -> list[tuple[str, float, float]]:
    """``(competency_id, mean_a, mean_b)`` for the same persona and competency
    under two style variants."""
    out: list[tuple[str, float, float]] = []
    for unit in units:
        runs_a = [r for r in unit.by_arm(arm) if r.run.style_id == style_a]
        runs_b = [r for r in unit.by_arm(arm) if r.run.style_id == style_b]
        if not runs_a or not runs_b:
            continue
        snap_a, snap_b = runs_a[0].final_belief(), runs_b[0].final_belief()
        if snap_a is None or snap_b is None:
            continue
        shared = set(snap_a.means) & set(snap_b.means)
        # Only competencies both variants actually probed: comparing a probed
        # posterior against an untouched prior would measure coverage, not
        # style.
        shared &= runs_a[0].probed() & runs_b[0].probed()
        for cid in sorted(shared):
            out.append((cid, snap_a.means[cid], snap_b.means[cid]))
    return out


def slice_drift(
    units: Sequence[PersonaRuns], arm: str, style_a: str, style_b: str
) -> SliceDrift:
    pairs = _paired_means(units, arm, style_a, style_b)
    if not pairs:
        return SliceDrift(style_a, style_b, float("nan"), float("nan"), 0)

    deltas = np.array([a - b for _c, a, b in pairs])
    by_competency: dict[str, list[float]] = {}
    for cid, a, b in pairs:
        by_competency.setdefault(cid, []).append(a - b)

    worst_cid, worst_value = None, 0.0
    for cid, values in by_competency.items():
        mean = float(np.mean(values))
        if abs(mean) > abs(worst_value):
            worst_cid, worst_value = cid, mean

    return SliceDrift(
        slice_a=style_a,
        slice_b=style_b,
        drift=float(np.mean(deltas)),
        abs_drift=float(np.mean(np.abs(deltas))),
        n_pairs=len(pairs),
        max_drift_competency=worst_cid,
        max_drift=worst_value,
        advance_rate_a=float(np.mean([a >= ADVANCE_THRESHOLD for _c, a, _b in pairs])),
        advance_rate_b=float(np.mean([b >= ADVANCE_THRESHOLD for _c, _a, b in pairs])),
    )


def measure(
    units: Sequence[PersonaRuns], arm: str, style_separation: bool
) -> FairnessReport:
    return FairnessReport(
        style_separation=style_separation,
        slices=[slice_drift(units, arm, a, b) for a, b in FAIRNESS_PAIRS],
    )


def name_swap_is_exact(units: Sequence[PersonaRuns], arm: str) -> tuple[bool, int, float]:
    """``(exactly equal, n compared, max difference)``.

    The name-swap pair is identical in every measurable dimension, so anything
    other than exact equality is a bug rather than a drift statistic. Under the
    offline backend this holds by construction — the grader never reads a name
    — and the README says so rather than presenting it as an empirical finding.
    What the test protects against is somebody later adding a name-sensitive
    feature.
    """
    pairs = _paired_means(units, arm, "name_a", "name_b")
    if not pairs:
        return True, 0, 0.0
    diffs = [abs(a - b) for _c, a, b in pairs]
    return max(diffs) < 1e-9, len(pairs), float(max(diffs))


def intervention_delta(before: FairnessReport, after: FairnessReport) -> dict:
    """Before/after the content-style separation toggle.

    ``before`` is the intervention *off*. A negative delta means the
    intervention reduced drift, which is the direction it is supposed to work
    in; anything that survives is the residual, and the residual is named
    rather than rounded away.
    """
    by_pair = {(s.slice_a, s.slice_b): s for s in after.slices}
    rows = []
    for slice_before in before.slices:
        key = (slice_before.slice_a, slice_before.slice_b)
        slice_after = by_pair.get(key)
        if slice_after is None:
            continue
        rows.append(
            {
                "slice": f"{key[0]} vs {key[1]}",
                "abs_drift_off": round(slice_before.abs_drift, 4),
                "abs_drift_on": round(slice_after.abs_drift, 4),
                "reduction": round(slice_before.abs_drift - slice_after.abs_drift, 4),
                "residual_max_drift_competency": slice_after.max_drift_competency,
            }
        )
    return {
        "mean_abs_drift_off": round(before.mean_abs_drift, 4),
        "mean_abs_drift_on": round(after.mean_abs_drift, 4),
        "reduction": round(before.mean_abs_drift - after.mean_abs_drift, 4),
        "slices": rows,
    }
