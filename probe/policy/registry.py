"""Arm construction, in one place.

Every arm is built here so that a sweep cannot accidentally give one of them a
different bank, a different stop rule or a different grader. The only thing
that may differ between arms is question selection; anything else is a
confound, and confounds in a four-way comparison are very hard to spot after
the fact.
"""

from __future__ import annotations

from probe.config import ExperimentConfig
from probe.models import Rubric
from probe.policy.base import Policy
from probe.policy.fixed import FixedPolicy

#: Arms available at this phase. Later phases register the rest.
ARMS: tuple[str, ...] = ("fixed",)


def make_policy(
    name: str,
    rubric: Rubric,
    config: ExperimentConfig,
    client=None,
) -> Policy:
    if name == "fixed":
        return FixedPolicy(rubric)
    raise ValueError(f"unknown arm {name!r}; available: {', '.join(ARMS)}")
