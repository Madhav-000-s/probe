"""Bootstrap confidence intervals.

Resampling is over **personas**, not turns. A persona contributes many
correlated observations — every competency in their rubric, every style
variant of them — and resampling turns would treat those as independent,
producing intervals several times too narrow. Narrow intervals on a
simulation study are exactly the way to make a null result look like a
finding.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

DEFAULT_RESAMPLES = 2000


@dataclass(frozen=True)
class Interval:
    point: float
    lo: float
    hi: float
    n: int
    resamples: int = DEFAULT_RESAMPLES

    @property
    def width(self) -> float:
        return self.hi - self.lo

    def render(self, places: int = 3) -> str:
        if not np.isfinite(self.point):
            return "n/a"
        return f"{self.point:.{places}f} [{self.lo:.{places}f}, {self.hi:.{places}f}]"

    def excludes(self, value: float) -> bool:
        """True when ``value`` lies outside the interval — the only form of
        'significant' this project uses."""
        return value < self.lo or value > self.hi

    def to_dict(self) -> dict[str, Any]:
        return {
            "point": self.point,
            "lo": self.lo,
            "hi": self.hi,
            "n": self.n,
            "resamples": self.resamples,
        }


def bootstrap_ci(
    units: Sequence[Any],
    statistic: Callable[[Sequence[Any]], float],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    mass: float = 0.95,
    seed: int = 20260808,
) -> Interval:
    """Percentile bootstrap over independent units.

    ``units`` must be the *independent* unit of analysis — one entry per
    persona, each carrying whatever that persona contributed.
    """
    units = list(units)
    point = statistic(units)
    if len(units) < 2:
        return Interval(point=point, lo=float("nan"), hi=float("nan"), n=len(units), resamples=0)

    rng = np.random.default_rng(seed)
    draws = np.empty(resamples)
    n = len(units)
    for i in range(resamples):
        idx = rng.integers(0, n, size=n)
        draws[i] = statistic([units[int(j)] for j in idx])

    finite = draws[np.isfinite(draws)]
    if finite.size == 0:
        return Interval(point=point, lo=float("nan"), hi=float("nan"), n=n, resamples=resamples)

    alpha = (1.0 - mass) / 2.0
    return Interval(
        point=point,
        lo=float(np.quantile(finite, alpha)),
        hi=float(np.quantile(finite, 1.0 - alpha)),
        n=n,
        resamples=resamples,
    )


def paired_difference_ci(
    units: Sequence[Any],
    statistic_a: Callable[[Sequence[Any]], float],
    statistic_b: Callable[[Sequence[Any]], float],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    mass: float = 0.95,
    seed: int = 20260808,
) -> Interval:
    """Interval on ``a - b`` with the *same* personas resampled for both arms.

    Pairing matters. Every arm interviews the same population, so a naive
    difference of two independent intervals throws away the correlation and is
    far more conservative than the design warrants.
    """

    def difference(sample: Sequence[Any]) -> float:
        return statistic_a(sample) - statistic_b(sample)

    return bootstrap_ci(units, difference, resamples=resamples, mass=mass, seed=seed)
