"""The ``eig+corr`` arm.

Identical selection maths to :class:`~probe.policy.eig.EIGPolicy`. The only
difference is the belief it runs against: a
:class:`~probe.belief.correlation.CorrelatedGridBelief` whose updates
propagate to correlated competencies.

Keeping the policy itself unchanged is the point of the ablation. If this arm
differs from `eig`, the difference is attributable to the correlation
structure and to nothing else — not to a different objective, a different cost
model or a different stop rule.
"""

from __future__ import annotations

from probe.config import ExperimentConfig
from probe.models import Rubric
from probe.policy.eig import EIGPolicy


class EIGCorrPolicy(EIGPolicy):
    name = "eig+corr"

    def __init__(self, rubric: Rubric, config: ExperimentConfig, **kwargs) -> None:
        super().__init__(rubric, config, **kwargs)
