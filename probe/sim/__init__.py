"""The measurement plane.

Everything in here knows ``theta_star``. Nothing in here may be imported by
``probe.rubric``, ``probe.belief``, ``probe.policy``, ``probe.grader`` or
``probe.runtime`` — the one exception being :mod:`probe.sim.llm_sim`, which is
a stand-in for the model *provider* and is reached only through the
``LLMClient`` protocol, never by type.

A test in ``tests/phase1`` asserts that import direction, because the firewall
is much easier to keep than to restore.
"""

from probe.sim.persona import PersonaGenerator, load_population, save_population
from probe.sim.style import STYLE_PRESETS, style_by_id

__all__ = [
    "PersonaGenerator",
    "load_population",
    "save_population",
    "STYLE_PRESETS",
    "style_by_id",
]
