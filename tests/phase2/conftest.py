"""Phase 2 reuses the Phase 1 population and bank."""

from __future__ import annotations

import pytest

from probe.bank.generate import starter_bank
from probe.jd import default_jds
from probe.models import Behavior
from probe.runtime.llm import get_client
from probe.sim.persona import PersonaGenerator
from probe.sim.style import MAIN_SWEEP_STYLES

SEED = 20260805


@pytest.fixture(scope="session")
def jds(taxonomy):
    return {jd.id: jd for jd in default_jds(taxonomy, seed=SEED)}


@pytest.fixture(scope="session")
def starter(taxonomy, jds):
    return starter_bank(taxonomy, jds=list(jds.values()))


@pytest.fixture(scope="session")
def personas(taxonomy, jds):
    return PersonaGenerator(taxonomy, seed=SEED).generate(
        10,
        behaviors=[Behavior.HONEST],
        styles=list(MAIN_SWEEP_STYLES),
        jd_ids=sorted(jds),
    )


@pytest.fixture
def sim():
    return get_client("sim", seed=SEED)
