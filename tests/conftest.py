"""Shared fixtures.

Everything here is deterministic. A test that depends on wall-clock time, an
unseeded RNG or a network call has no place in a suite whose job is to make
statistical claims believable.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from probe.bank.loader import stub_bank
from probe.config import Budgets, ExperimentConfig
from probe.models import LLMRole
from probe.rubric.taxonomy import load_taxonomy
from probe.runtime.llm import FakeLLM
from probe.runtime.tracing import TraceStore


@pytest.fixture(scope="session")
def taxonomy():
    return load_taxonomy()


@pytest.fixture
def rubric(taxonomy):
    return taxonomy.stub_rubric(candidate_id="p-test", n=6)


@pytest.fixture
def bank(taxonomy, rubric):
    return stub_bank(taxonomy, competency_ids=rubric.ids, per_competency=2)


@pytest.fixture
def config():
    return ExperimentConfig(
        budgets=Budgets(max_questions=5, max_tokens=1_000_000, max_wallclock_seconds=1e9),
        seed_set=[7],
    )


@pytest.fixture
def store():
    s = TraceStore(":memory:")
    yield s
    s.close()


@pytest.fixture
def trace_file(tmp_path):
    return tmp_path / "probe-test.duckdb"


def fake_client(grade_responder) -> FakeLLM:
    return FakeLLM(by_role={LLMRole.GRADE: grade_responder}, strict=False)


def with_budget(config: ExperimentConfig, **kwargs) -> ExperimentConfig:
    return replace(config, budgets=replace(config.budgets, **kwargs))
