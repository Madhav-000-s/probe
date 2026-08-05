"""Belief state over latent ability.

Phase 0 ships the interface and a prior-only implementation so the runtime
spine can be built and traced end to end. Phase 2 replaces the implementation
with the grid posterior and the graded-response likelihood, which is where the
numerical verification lives.
"""

from probe.belief.grid import THETA_GRID, GridBelief
from probe.belief.state import BeliefState, PriorOnlyBelief

__all__ = ["BeliefState", "GridBelief", "PriorOnlyBelief", "THETA_GRID"]
