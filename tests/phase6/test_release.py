"""Phase 6 release checklist.

The question this suite answers is not "does the code work" — the other five
phases cover that — but "can a stranger clone this and get the same numbers,
and does the README say what the numbers actually say".
"""

from __future__ import annotations

import json
import re

import pytest

from probe.config import FIGURE_DIR, RESULTS_DIR, ROOT, ExperimentConfig
from probe.runtime.tracing import TraceStore

TRACES = "traces/probe.duckdb"
README = ROOT / "README.md"
ABLATIONS = RESULTS_DIR / "ablations.json"


@pytest.fixture(scope="module")
def config():
    return ExperimentConfig.load()


# --------------------------------------------------------------- artefacts


def test_every_committed_artefact_exists():
    """The three artefacts D2 requires, plus what they were computed from."""
    required = [
        RESULTS_DIR / "main-table.json",
        RESULTS_DIR / "curves.json",
        RESULTS_DIR / "fairness.json",
        RESULTS_DIR / "robustness.json",
        RESULTS_DIR / "gold-agreement.json",
        RESULTS_DIR / "ablations.json",
        FIGURE_DIR / "accuracy-vs-budget.png",
        ROOT / "experiment-config.yaml",
        ROOT / "results-log.md",
        ROOT / "data" / "gold" / "gold-set.csv",
        ROOT / "data" / "correlation.json",
    ]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
    assert not missing, f"missing release artefacts: {missing}"


def test_committed_traces_are_present_and_complete(config):
    """`make eval` on a fresh clone reads these, so they have to be in the
    repo rather than regenerated."""
    store = TraceStore(TRACES, read_only=True)
    try:
        counts = store.counts()
    finally:
        store.close()
    assert counts["runs"] >= 384
    assert counts["turns"] > 3000
    assert counts["llm_calls"] > 10_000


def test_provenance_is_consistent_across_every_result_file(config):
    """A results table that quoted a different bank version from the fairness
    table would be two experiments presented as one."""
    for name in ("main-table.json", "fairness.json", "robustness.json"):
        payload = json.loads((RESULTS_DIR / name).read_text(encoding="utf-8"))
        provenance = payload["provenance"]
        assert provenance["bank_version"] == config.bank_version
        assert provenance["population_version"] == config.population_version
        assert provenance["taxonomy_version"] == config.taxonomy_version


# ------------------------------------------------------------- ablations


def test_followup_ablation_earns_its_place():
    """Follow-ups exist to rescue candidates who say less than they know, so
    the ablation has to show that on terse candidates specifically."""
    ablations = json.loads(ABLATIONS.read_text(encoding="utf-8"))
    on = ablations["followup_ablation"]["followups on"]
    off = ablations["followup_ablation"]["followups off"]

    assert on["followup_rate"] > 0.1 and off["followup_rate"] == 0.0
    assert on["terse_recovery_rho"] > off["terse_recovery_rho"], (
        "follow-ups do not help terse candidates, which is the only reason "
        "they are in the system"
    )
    # And they cost something — otherwise the ablation would be free.
    assert on["mean_questions"] > off["mean_questions"]


def test_terse_followup_result_is_reported_as_thin():
    """One terse persona reached the eval split. The finding points the right
    way and is not strong evidence, and the README has to say so rather than
    quoting 0.57 vs 0.37 as though n were large."""
    ablations = json.loads(ABLATIONS.read_text(encoding="utf-8"))
    n = ablations["followup_ablation"]["followups on"]["n_terse_personas"]
    assert n <= 3
    assert re.search(
        r"(one|single|1)\s+terse|n\s*=\s*1", README.read_text(encoding="utf-8"), re.I
    ), "the README quotes the terse ablation without flagging the sample size"


def test_budget_ablation_is_monotone_and_ordered():
    """More questions must not make recovery worse, and the arm ordering should
    hold at every budget rather than only at the end."""
    ablations = json.loads(ABLATIONS.read_text(encoding="utf-8"))
    by_budget = ablations["recovery_by_budget"]
    for arm in ("fixed", "heuristic", "eig"):
        series = [by_budget[b][arm] for b in sorted(by_budget, key=int)]
        assert series[-1] >= series[0], f"{arm} recovery degrades with more questions"
    for budget in by_budget:
        assert by_budget[budget]["eig"] >= by_budget[budget]["fixed"]


# ----------------------------------------------------------- trace viewer


def test_viewer_renders_a_committed_run():
    from probe.report.viewer import render_run

    store = TraceStore(TRACES, read_only=True)
    try:
        run_id = store.run_ids(arm="eig")[0]
        rendered = render_run(store, run_id)
    finally:
        store.close()

    assert run_id in rendered
    assert "theta" in rendered
    assert "EIG=" in rendered


def test_side_by_side_demo_is_a_rendering_not_a_recording():
    """D4: both panes reconstructed from committed traces."""
    demo = ROOT / "analysis" / "demo-side-by-side.txt"
    assert demo.exists()
    text = demo.read_text(encoding="utf-8")
    assert "rendered from committed traces" in text
    assert "fixed" in text and "eig" in text
    assert text.count("|") > 20, "the two panes are not both populated"


# ---------------------------------------------------------------- README


@pytest.fixture(scope="module")
def readme():
    return README.read_text(encoding="utf-8")


def test_readme_states_the_three_questions(readme):
    lowered = readme.lower()
    for phrase in ("efficiency", "grader reliability", "style invariance"):
        assert phrase in lowered


def test_readme_leads_with_limitations_not_conclusions(readme):
    """The credibility move the plan asks for: limitations before results."""
    limitations = readme.lower().find("what this is not")
    results = readme.lower().find("## results")
    assert limitations != -1, "no limitations section"
    assert results != -1
    assert limitations < results, "limitations must appear before the results"


def test_readme_names_every_required_caveat(readme):
    lowered = readme.lower()
    required = [
        "simulated",          # simulated-construct caveat
        "no human",           # single-evaluator / gold set honesty
        "calibration",        # calibration/eval split design
        "unidimensional",     # one theta per competency
        "proxy",              # synthetic style as a proxy
        "overconfident",      # the credible-interval failure
    ]
    missing = [c for c in required if c not in lowered]
    assert not missing, f"README omits required caveats: {missing}"


def test_readme_reports_where_the_adaptive_policy_loses(readme):
    """D2: results where the policy loses go in the main text, not an
    appendix."""
    lowered = readme.lower()
    assert "bluff" in lowered, "the bluffing negative result is missing"
    assert "includes zero" in lowered or "not established" in lowered, (
        "the README does not say the recovery-rho difference is inside the noise"
    )


def test_readme_numbers_match_the_generated_artefacts(readme):
    """Every quoted number has to come from a `make eval` output cell.

    Spot-checked on the headline figures — if the table is regenerated and the
    README is not, this fails.
    """
    table = json.loads((RESULTS_DIR / "main-table.json").read_text(encoding="utf-8"))
    rows = {row["arm"]: row for row in table["arms"]}

    for arm in ("fixed", "heuristic", "eig", "eig+corr"):
        rho = rows[arm]["recovery_rho"]["point"]
        assert f"{rho:.3f}" in readme, f"README does not quote {arm} recovery rho {rho:.3f}"

    robustness = json.loads((RESULTS_DIR / "robustness.json").read_text(encoding="utf-8"))
    resistance = robustness["pooled"]["injection_resistance"]
    assert f"{resistance:.3f}" in readme or f"{resistance:.1%}" in readme

    gold = json.loads((RESULTS_DIR / "gold-agreement.json").read_text(encoding="utf-8"))
    assert f"{gold['cohens_kappa']:.3f}" in readme


def test_readme_links_resolve(readme):
    """No broken relative links on the front page."""
    broken = []
    for match in re.finditer(r"\[[^\]]+\]\(([^)]+)\)", readme):
        target = match.group(1)
        if target.startswith(("http", "#", "mailto:")):
            continue
        if not (ROOT / target.split("#")[0]).exists():
            broken.append(target)
    assert not broken, f"broken README links: {broken}"


def test_readme_does_not_link_to_ignored_planning_docs(readme):
    """ARCHITECTURE.md, PLAN.md and DELIVERABLES.md are deliberately not in the
    repo, so the front page must not point at them."""
    for doc in ("ARCHITECTURE.md", "PLAN.md", "DELIVERABLES.md"):
        assert f"({doc})" not in readme, f"README links to gitignored {doc}"


def test_every_make_recipe_invokes_something_that_exists():
    """`make experiment` invoked `probe experiment run`, which Typer rejects as
    an unexpected argument — the target had never been exercised because the
    sweeps were driven by the CLI directly. Checking that a target *exists*
    (above) is not the same as checking that what it runs does.
    """
    import importlib.util

    from typer.main import get_command

    from probe.cli import app

    root = get_command(app)
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

    unknown = []
    # ``[^\S\n]`` rather than ``\s``: a greedy match across the newline runs
    # one recipe's arguments into the next target's name.
    for match in re.finditer(r"\$\(PROBE\)((?:[^\S\n]+[a-z][\w-]*)+)", makefile):
        words = match.group(1).split()
        command, path = root, []
        for word in words:
            sub = getattr(command, "commands", {}).get(word)
            if sub is None:
                unknown.append(" ".join(path + [word]))
                break
            command, path = sub, path + [word]

    for match in re.finditer(r"\$\(PY\) -m ([\w.]+)", makefile):
        module = match.group(1)
        if importlib.util.find_spec(module) is None:
            unknown.append(f"-m {module}")

    assert not unknown, f"Makefile recipes that resolve to nothing: {unknown}"


def test_make_targets_referenced_by_the_readme_exist(readme):
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    for match in re.finditer(r"`make ([a-z0-9]+(?:-[a-z0-9]+)*)", readme):
        target = match.group(1)
        if target.startswith("gate"):
            # `make gate-N` is a template for gate-0 .. gate-6.
            assert re.search(r"^gate-0:", makefile, re.M)
            continue
        assert re.search(rf"^{re.escape(target)}:", makefile, re.M), (
            f"README mentions `make {target}` but the Makefile has no such target"
        )
