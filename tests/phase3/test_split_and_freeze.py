"""Split hygiene, the correlation matrix's provenance, and the frozen constants.

The held-out split is the answer to the single most likely reviewer objection:
that the bank was calibrated on the same data it is scored against. That answer
is only worth anything if it is enforced, so it is tested rather than asserted
in prose.
"""

from __future__ import annotations

import numpy as np
import pytest

from probe.belief.correlation import (
    MAX_BORROW,
    MIN_ABS_CORRELATION,
    CompetencyCorrelation,
    CorrelatedGridBelief,
    estimate_correlation,
)
from probe.config import EXPERIMENT_CONFIG_PATH, ExperimentConfig
from probe.experiment import CORRELATION_PATH
from probe.models import Behavior
from probe.sim.persona import load_population


@pytest.fixture(scope="module")
def population():
    return load_population("v2")


# --------------------------------------------------------- split hygiene


def test_population_is_split_before_any_fitting(population):
    personas, meta = population
    calibration = [p for p in personas if p.split == "calibration"]
    evaluation = [p for p in personas if p.split == "eval"]

    assert len(personas) == 60
    assert len(calibration) == 36
    assert len(evaluation) == 24
    assert meta.calibration_fraction == pytest.approx(0.6)


def test_no_eval_persona_appears_in_any_calibration_input(population):
    """The automated check PLAN.md asks for.

    Ids are disjoint by construction, and the assertion is on ids rather than
    on objects so that a copy or a restyled variant could not slip through.
    """
    personas, _meta = population
    calibration_ids = {p.id for p in personas if p.split == "calibration"}
    eval_ids = {p.id for p in personas if p.split == "eval"}

    assert calibration_ids & eval_ids == set()
    assert len(calibration_ids | eval_ids) == len(personas)


def test_adversarial_fraction_is_about_a_quarter(population):
    personas, meta = population
    adversarial = [p for p in personas if p.behavior is not Behavior.HONEST]
    assert 0.15 <= len(adversarial) / len(personas) <= 0.35, (
        f"adversarial fraction {meta.adversarial_fraction:.0%}; the plan says ~25%"
    )
    assert {p.behavior for p in adversarial} == {
        b for b in Behavior if b is not Behavior.HONEST
    }, "every adversarial behaviour must be represented"


def test_population_is_regenerable_from_its_seed(population, taxonomy):
    """A population that cannot be regenerated from its seed cannot be
    audited, and its version tag in a provenance tuple would mean nothing."""
    from probe.jd import default_jds
    from probe.sim.persona import PersonaGenerator
    from probe.sim.style import MAIN_SWEEP_STYLES

    personas, meta = population
    jds = default_jds(taxonomy, seed=meta.seed)
    behaviors = [Behavior.HONEST] * 18 + [
        Behavior.BLUFFER,
        Behavior.TERSE,
        Behavior.RAMBLER,
        Behavior.INJECTOR,
        Behavior.DODGER,
        Behavior.OVERCLAIMER,
    ]
    regenerated = PersonaGenerator(taxonomy, seed=meta.seed).generate(
        meta.n_personas,
        behaviors=behaviors,
        styles=list(MAIN_SWEEP_STYLES),
        jd_ids=[jd.id for jd in jds],
        calibration_fraction=meta.calibration_fraction,
    )
    assert [p.id for p in regenerated] == [p.id for p in personas]
    assert regenerated[0].theta_star == personas[0].theta_star
    assert regenerated[-1].resume == personas[-1].resume


# ------------------------------------------------------------ correlation


def test_correlation_records_calibration_provenance():
    """The arm must not be exploiting structure measured on the split it is
    scored against."""
    correlation = CompetencyCorrelation.load(CORRELATION_PATH)
    assert correlation.provenance == "calibration"
    assert correlation.n_respondents == 36
    assert len(correlation.competency_ids) == 50


def test_correlation_matrix_is_well_formed():
    correlation = CompetencyCorrelation.load(CORRELATION_PATH)
    matrix = correlation.matrix
    assert matrix.shape == (50, 50)
    assert np.allclose(np.diag(matrix), 1.0)
    assert np.allclose(matrix, matrix.T)
    assert np.all(np.abs(matrix) <= 1.0 + 1e-9)


def test_correlation_finds_within_area_structure():
    """Abilities were generated with a block structure, so the empirical
    estimate should find more agreement inside an area than across families."""
    correlation = CompetencyCorrelation.load(CORRELATION_PATH)
    ids = correlation.competency_ids
    same_area, cross = [], []
    for i, a in enumerate(ids):
        for j, b in enumerate(ids):
            if i >= j:
                continue
            (same_area if a.split(".")[0] == b.split(".")[0] else cross).append(
                correlation.matrix[i, j]
            )
    assert np.mean(same_area) > np.mean(cross)


def test_weak_correlations_are_ignored():
    correlation = estimate_correlation(
        np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 3.0], [4.0, 2.0]] * 4),
        ["a.x", "b.y"],
    )
    neighbours = correlation.neighbours("a.x", ["a.x", "b.y"])
    for _other, rho in neighbours:
        assert abs(rho) >= MIN_ABS_CORRELATION


def test_correlated_belief_propagates_and_is_capped(taxonomy):
    """A neighbour should move, but never as if it had been asked directly."""
    from probe.bank.loader import stub_bank

    rubric = taxonomy.stub_rubric(n=4)
    ids = rubric.ids
    matrix = np.eye(len(ids))
    matrix[0, 1] = matrix[1, 0] = 0.9
    correlation = CompetencyCorrelation(ids, matrix, "calibration", 36)

    bank = stub_bank(taxonomy, competency_ids=ids, per_competency=1)
    question = bank.for_competency(ids[0])[0]

    coupled = CorrelatedGridBelief(rubric, correlation)
    before_neighbour = coupled.mean(ids[1])
    before_target = coupled.mean(ids[0])
    coupled.update(question, 5)

    moved_target = coupled.mean(ids[0]) - before_target
    moved_neighbour = coupled.mean(ids[1]) - before_neighbour

    assert moved_target > 0
    assert moved_neighbour > 0, "correlated neighbour did not move"
    assert abs(moved_neighbour) < abs(moved_target), (
        "a neighbour must never move as much as the competency actually asked about"
    )
    assert coupled.borrowed_updates == 1
    assert MAX_BORROW < 1.0


def test_uncorrelated_competencies_do_not_move(taxonomy):
    from probe.bank.loader import stub_bank

    rubric = taxonomy.stub_rubric(n=4)
    ids = rubric.ids
    correlation = CompetencyCorrelation(ids, np.eye(len(ids)), "calibration", 36)
    bank = stub_bank(taxonomy, competency_ids=ids, per_competency=1)

    coupled = CorrelatedGridBelief(rubric, correlation)
    before = {cid: coupled.mean(cid) for cid in ids}
    coupled.update(bank.for_competency(ids[0])[0], 5)

    for cid in ids[1:]:
        assert coupled.mean(cid) == pytest.approx(before[cid])
    assert coupled.borrowed_updates == 0


def test_correlated_belief_without_a_matrix_is_plain(taxonomy):
    from probe.bank.loader import stub_bank

    rubric = taxonomy.stub_rubric(n=3)
    bank = stub_bank(taxonomy, competency_ids=rubric.ids, per_competency=1)
    coupled = CorrelatedGridBelief(rubric, None)
    coupled.update(bank.questions[0], 4)
    assert coupled.borrowed_updates == 0


# ------------------------------------------------------- frozen constants


def test_experiment_config_exists_and_is_frozen():
    assert EXPERIMENT_CONFIG_PATH.exists()
    config = ExperimentConfig.load()
    assert config.frozen is True
    assert config.tau == pytest.approx(0.80)
    assert config.epsilon == pytest.approx(0.01)
    assert config.budgets.max_questions == 12
    assert config.bank_version == "v2"
    assert config.population_version == "v2"
    assert config.seed_set == [20260807]


def test_frozen_config_carries_a_dated_change_log():
    import yaml

    raw = yaml.safe_load(EXPERIMENT_CONFIG_PATH.read_text(encoding="utf-8"))
    log = raw.get("change_log")
    assert log, "freezing without a justification is not freezing"
    assert log[0]["date"] and log[0]["justification"]


def test_provenance_tuple_is_complete():
    """Every reported number has to carry this."""
    provenance = ExperimentConfig.load().provenance
    assert set(provenance) == {
        "population_version",
        "bank_version",
        "taxonomy_version",
        "code_commit",
        "seed_set",
    }
    assert all(provenance.values())
