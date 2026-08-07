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
from probe.policy.eig import EIGPolicy
from probe.policy.eig_corr import EIGCorrPolicy
from probe.policy.fixed import FixedPolicy
from probe.policy.heuristic import HeuristicPolicy

#: The four arms of the main experiment.
ARMS: tuple[str, ...] = ("fixed", "heuristic", "eig", "eig+corr")

#: Arms that require a grid posterior. The prior-only belief would score them
#: at chance, so building one against it is a configuration error rather than
#: a degraded mode.
BELIEF_ARMS: frozenset[str] = frozenset({"eig", "eig+corr"})


def make_policy(
    name: str,
    rubric: Rubric,
    config: ExperimentConfig,
    client=None,
    *,
    seed: int = 0,
) -> Policy:
    if name == "fixed":
        return FixedPolicy(rubric)
    if name == "heuristic":
        if client is None:
            raise ValueError("the heuristic arm needs an LLM client — it is the model chooser")
        return HeuristicPolicy(rubric, client, seed=seed)
    if name == "eig":
        return EIGPolicy(rubric, config)
    if name == "eig+corr":
        # Same selection maths as `eig`; the ablation lives entirely in the
        # belief state it is paired with (see runtime.session).
        return EIGCorrPolicy(rubric, config)
    raise ValueError(f"unknown arm {name!r}; available: {', '.join(ARMS)}")


def needs_grid_belief(name: str) -> bool:
    return name in BELIEF_ARMS
