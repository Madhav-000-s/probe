"""The Phase 4 exit gate: the results table, its reproducibility, and the
calibration failure it exposed.

These run against the committed traces, so they are checking the numbers the
README actually quotes rather than a fixture that resembles them.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from evals.metrics import efficiency
from evals.metrics.loader import load_views
from probe.config import RESULTS_DIR, ExperimentConfig
from probe.policy.registry import ARMS

TRACES = "traces/probe.duckdb"
MAIN_TABLE = RESULTS_DIR / "main-table.json"
GOLD_AGREEMENT = RESULTS_DIR / "gold-agreement.json"


@pytest.fixture(scope="module")
def table():
    if not MAIN_TABLE.exists():
        pytest.skip("run `make eval` first")
    return json.loads(MAIN_TABLE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def units():
    if not Path(TRACES).exists():
        pytest.skip("run `make experiment` first")
    config = ExperimentConfig.load()
    return load_views(TRACES, config.population_version, arms=ARMS)


# ------------------------------------------------------------ the artefact


def test_main_table_covers_every_arm_with_intervals(table):
    assert {row["arm"] for row in table["arms"]} == set(ARMS)
    for row in table["arms"]:
        for metric in ("recovery_rho", "resolved_fraction", "ece", "mean_posterior_sd"):
            interval = row[metric]
            assert interval["point"] is not None, f"{row['arm']}/{metric}"
            assert interval["lo"] <= interval["point"] <= interval["hi"]
            assert interval["n"] >= 20, "bootstrap ran on too few personas"


def test_every_number_carries_its_provenance(table):
    provenance = table["provenance"]
    assert set(provenance) == {
        "population_version",
        "bank_version",
        "taxonomy_version",
        "code_commit",
        "seed_set",
    }
    assert provenance["bank_version"] == "v2"
    assert provenance["population_version"] == "v2"


def test_eval_is_byte_reproducible():
    """`make eval` twice on the same traces must produce identical bytes.

    This is the reproducibility contract in DELIVERABLES.md, and it is the
    reason nothing in the eval path samples without a fixed seed or rounds at
    display time.
    """
    from evals.run_eval import main

    before = MAIN_TABLE.read_bytes()
    assert main(["--traces", TRACES, "--suites", "evals/suites"]) == 0
    assert MAIN_TABLE.read_bytes() == before


def test_figure_is_generated_not_committed_by_hand():
    from probe.config import FIGURE_DIR

    figure = FIGURE_DIR / "accuracy-vs-budget.png"
    assert figure.exists()
    assert figure.stat().st_size > 10_000


# ------------------------------------------------------- curve integrity


def test_accuracy_curve_is_computed_from_stored_snapshots(units):
    """Recompute one run's curve by replaying its turns and diff it against
    the aggregate path.

    Two independent implementations. If the persisted snapshots had drifted
    from what the interview actually believed, they would disagree.
    """
    budgets = (2, 4, 6, 8)
    run = next(r for u in units for r in u.by_arm("eig") if len(r.turns) >= 8)

    from evals.metrics.loader import PersonaRuns

    single = [PersonaRuns(run.persona.id, [run])]
    aggregate = efficiency.accuracy_vs_budget(single, "eig", budgets)
    replayed = efficiency.replay_curve(run, budgets)

    for budget in budgets:
        a, b = aggregate[budget], replayed[budget]
        if np.isnan(a) and np.isnan(b):
            continue
        assert a == pytest.approx(b, abs=1e-9), f"curve mismatch at budget {budget}"


def test_curves_are_monotone_enough_to_be_believable(table):
    """More questions should not systematically make recovery worse."""
    curves = json.loads((RESULTS_DIR / "curves.json").read_text(encoding="utf-8"))
    for arm, curve in curves.items():
        points = [v for _b, v in sorted(curve.items(), key=lambda kv: int(kv[0])) if v is not None]
        if len(points) < 6:
            continue
        assert points[-1] >= points[0] - 0.05, f"{arm} recovery degrades with more questions"


# ------------------------------------------------------------- the result


def test_adaptive_arms_beat_the_heuristic_competitor(table):
    """Q1, with intervals. Asserted against the *heuristic* arm, because
    beating the fixed script proves nothing.

    The claim is deliberately split, because the evidence is:

    * **efficiency** — `eig` resolves far more competencies below tau, and the
      paired interval excludes zero comfortably. This is the headline.
    * **recovery accuracy** — `eig` is directionally better, but at n = 24
      personas the paired interval on rho *includes* zero. It is not
      established.

    An earlier version of this test asserted both. That was true of an earlier
    sweep and stopped being true once the fairness fix removed content variance
    from the style variants — which is exactly why the assertion is here rather
    than only in prose. The README says "faster to the same accuracy", not
    "more accurate", and this test is what keeps it honest.
    """
    contrasts = {
        (c["arm"], c["metric"]): c
        for c in table["contrasts"]
        if c["baseline"] == "heuristic"
    }

    resolved = contrasts[("eig", "resolved_fraction")]
    assert resolved["difference"]["point"] > 0
    assert resolved["excludes_zero"], "the efficiency claim is no longer supported"

    rho = contrasts[("eig", "recovery_rho")]
    assert rho["difference"]["point"] > 0, "eig no longer even points the right way on rho"
    assert not rho["excludes_zero"], (
        "eig's recovery advantage now excludes zero — better than expected, but "
        "the README currently claims only an efficiency win and needs updating"
    )


def test_correlation_ablation_helps_rather_than_hurts(table):
    rows = {row["arm"]: row for row in table["arms"]}
    assert rows["eig+corr"]["recovery_rho"]["point"] >= rows["eig"]["recovery_rho"]["point"]
    assert (
        rows["eig+corr"]["questions_to_confidence"]["point"]
        <= rows["eig"]["questions_to_confidence"]["point"]
    )


def test_correlation_arm_is_not_manufacturing_confidence(table):
    """The copula's obvious failure mode: borrowing its way to a tight
    posterior it has not earned.

    If it were doing that, its coverage would be *worse* than the independent
    arm's and its ECE higher. Both go the other way, which is the evidence
    that the borrowed information is real.
    """
    rows = {row["arm"]: row for row in table["arms"]}
    assert rows["eig+corr"]["coverage_80"] >= rows["eig"]["coverage_80"]
    assert rows["eig+corr"]["ece"]["point"] <= rows["eig"]["ece"]["point"]


def test_credible_intervals_are_overconfident(table):
    """A failure, pinned so it cannot be quietly forgotten.

    Nominal 80% intervals cover about 68-71% in the full system, even though
    the Phase 2 coverage test passes at 78-82% in isolation. The difference is
    model misspecification: the belief update treats the grader as a clean
    graded-response model, and the real grader adds noise -- seed variance,
    position bias, a style term -- that the likelihood does not represent. The
    posterior therefore tightens faster than the evidence warrants.

    This is named in the README limitations and is the top item in what's next.
    The test asserts the *current* state so that a fix registers as a failure
    here and forces the write-up to be updated.
    """
    for row in table["arms"]:
        coverage = row["coverage_80"]
        assert 0.60 <= coverage <= 0.75, (
            f"{row['arm']} coverage is {coverage:.3f}; if this has moved above "
            f"0.75 the overconfidence has been fixed and the limitations "
            f"section needs rewriting"
        )


def test_cost_is_reported_per_arm(table):
    """If EIG wins on questions but loses on dollars, the harness has to say
    so."""
    for row in table["arms"]:
        assert row["cost"]["usd_per_interview"] > 0
        assert row["cost"]["tokens_per_interview"] > 0
    rows = {row["arm"]: row for row in table["arms"]}
    assert rows["eig"]["cost"]["usd_per_interview"] < rows["heuristic"]["cost"]["usd_per_interview"]


def test_stop_reasons_are_reported_and_informative(table):
    rows = {row["arm"]: row for row in table["arms"]}
    assert rows["eig+corr"]["stop_reasons"].get("confidence", 0) > 0.9
    assert rows["fixed"]["stop_reasons"].get("confidence", 0) < 0.5


def test_schema_health_is_measured(table):
    health = table["schema_health"]
    assert health["n_calls"] > 1000
    assert 0.0 < health["violation_rate"] < 0.15
    assert health["repair_success_rate"] > 0.5


# ------------------------------------------------------------ human anchor


def test_kappa_clears_the_threshold():
    """PLAN.md: if kappa < 0.5 the grader is not measuring the construct and
    no arm comparison can be trusted."""
    if not GOLD_AGREEMENT.exists():
        pytest.skip("run `python -m evals.build_gold` first")
    report = json.loads(GOLD_AGREEMENT.read_text(encoding="utf-8"))
    assert report["n"] >= 60
    assert report["cohens_kappa"] >= 0.5, report
    assert report["within_one_agreement"] > 0.9


def test_released_gold_set_leaks_no_ground_truth():
    """D5: answers and grades only."""
    from probe.config import GOLD_DIR

    path = GOLD_DIR / "gold-set.csv"
    if not path.exists():
        pytest.skip("run `python -m evals.build_gold` first")
    text = path.read_text(encoding="utf-8")
    assert "theta" not in text.lower()
    assert "persona_id" not in text.lower()


def test_gold_protocol_is_documented_and_honest():
    from evals.gold import PROTOCOL

    assert PROTOCOL["blind_to_llm_score"] and PROTOCOL["blind_to_theta_star"]
    assert "NOT human" in PROTOCOL["rater"], (
        "the anchor set was not produced by a human and must never say it was"
    )
