"""The command-line surface.

Thin by design: every command here is a few lines of wiring over a library
call, so anything the CLI can do, a test or a notebook can do the same way.
Nothing in the pipeline is reachable only through Typer.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import typer
from rich.console import Console

from probe import __version__
from probe.bank.generate import bank_summary, full_bank, starter_bank
from probe.bank.loader import load_bank, save_bank
from probe.belief.calibration import calibrate_bank
from probe.belief.correlation import estimate_correlation
from probe.config import DEFAULT_TRACE_DB, EXPERIMENT_CONFIG_PATH, ExperimentConfig, ensure_dirs
from probe.experiment import CORRELATION_PATH, SweepPlan, run_sweep
from probe.jd import default_jds, load_jd, save_jd
from probe.models import Behavior
from probe.policy.registry import ARMS
from probe.report.render import render_report
from probe.rubric.taxonomy import load_taxonomy
from probe.runtime.llm import get_client
from probe.runtime.session import InterviewSpec, build_interview
from probe.runtime.tracing import TracedClient, TraceStore
from probe.sim.administer import administer
from probe.sim.fidelity import run_fidelity_gate
from probe.sim.persona import (
    PersonaGenerator,
    PopulationMeta,
    load_population,
    save_population,
)
from probe.sim.style import MAIN_SWEEP_STYLES

console = Console()

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="probe — adaptive interviewing agent and evaluation harness.",
)
taxonomy_app = typer.Typer(no_args_is_help=True, help="Inspect the competency taxonomy.")
bank_app = typer.Typer(no_args_is_help=True, help="Build and calibrate the question bank.")
population_app = typer.Typer(no_args_is_help=True, help="Generate the persona population.")
app.add_typer(taxonomy_app, name="taxonomy")
app.add_typer(bank_app, name="bank")
app.add_typer(population_app, name="population")


@app.callback()
def _root(
    version: bool = typer.Option(False, "--version", help="Print the version and exit."),
) -> None:
    if version:
        console.print(f"probe {__version__}")
        raise typer.Exit()


# ------------------------------------------------------------------ taxonomy


@taxonomy_app.command("validate")
def taxonomy_validate() -> None:
    """Check the taxonomy for structural problems. Exits non-zero on any."""
    tax = load_taxonomy()
    problems = tax.validate()
    if problems:
        console.print(f"[red]{len(problems)} problem(s) in taxonomy {tax.version}[/red]")
        for p in problems:
            console.print(f"  - {p}")
        raise typer.Exit(code=1)
    backend = sum(1 for n in tax if n.family.value == "backend")
    console.print(
        f"[green]taxonomy {tax.version} ok[/green] — {len(tax)} nodes "
        f"({backend} backend, {len(tax) - backend} data/ml), "
        f"{sum(len(n.concepts) for n in tax)} concepts"
    )


@taxonomy_app.command("list")
def taxonomy_list(family: str = typer.Option("", help="backend | data_ml")) -> None:
    tax = load_taxonomy()
    for node in tax:
        if family and node.family.value != family:
            continue
        console.print(
            f"{node.id:<42} L{node.default_required_level} "
            f"{node.family.value:<8} {len(node.concepts)} concepts"
        )


# ---------------------------------------------------------------------- bank


@bank_app.command("generate")
def bank_generate(
    size: str = typer.Option("starter", help="starter (60 items) | full (200 items)"),
    version: str = typer.Option("", help="Bank version tag; defaults per size."),
    seed: int = typer.Option(20260801),
) -> None:
    """Write a bank with authoring-default item parameters."""
    ensure_dirs()
    tax = load_taxonomy()
    if size == "starter":
        # Same seed as `population generate`, so the bank covers the roles the
        # personas are actually interviewed for.
        bank = starter_bank(tax, version=version or "v1-starter", jds=default_jds(tax, seed=seed))
    elif size == "full":
        bank = full_bank(tax, version=version or "v2-raw")
    else:
        console.print("[red]size must be 'starter' or 'full'[/red]")
        raise typer.Exit(code=2)

    path = save_bank(bank)
    summary = bank_summary(bank)
    console.print(f"[green]wrote[/green] {path}")
    console.print(json.dumps(summary, indent=2))


@bank_app.command("show")
def bank_show(version: str = typer.Option("v1-starter")) -> None:
    console.print(json.dumps(bank_summary(load_bank(version)), indent=2))


@bank_app.command("calibrate")
def bank_calibrate(
    raw_version: str = typer.Option("v2-raw", help="Uncalibrated bank to fit."),
    new_version: str = typer.Option("v2", help="Version tag for the fitted bank."),
    population: str = typer.Option("v2", help="Population to calibrate against."),
    seed: int = typer.Option(20260807),
) -> None:
    """Fit GRM item parameters on the calibration split and emit a new bank.

    Split hygiene is enforced here, not assumed: only calibration-split
    personas are administered, and the correlation matrix estimated alongside
    records that it came from them. An eval-split persona reaching this code
    path would make the held-out design a fiction.
    """
    ensure_dirs()
    personas, meta = load_population(population)
    calibration = [p for p in personas if p.split == "calibration"]
    if not calibration:
        console.print("[red]population has no calibration split[/red]")
        raise typer.Exit(code=2)

    bank = load_bank(raw_version)
    client = get_client("sim", seed=seed)

    console.print(
        f"administering {len(bank)} items to {len(calibration)} calibration personas "
        f"({len(bank) * len(calibration):,} responses)…"
    )
    administration = administer(calibration, bank, client, seed=seed)
    console.print(
        f"  graded {administration.n_graded:,}, "
        f"unrecoverable {administration.n_unrecoverable}"
    )

    fitted, report = calibrate_bank(bank, administration.responses, new_version=new_version)
    path = save_bank(fitted)
    console.print(f"[green]wrote[/green] {path}")
    console.print(json.dumps(report.summary(), indent=2))

    competency_ids = sorted({q.competency_id for q in bank.questions})
    correlation = estimate_correlation(
        administration.score_matrix(competency_ids),
        competency_ids,
        provenance="calibration",
    )
    correlation.save(CORRELATION_PATH)
    off_diagonal = correlation.matrix[~np.eye(len(competency_ids), dtype=bool)]
    console.print(
        f"[green]wrote[/green] {CORRELATION_PATH}\n"
        f"  correlation over {len(competency_ids)} competencies from "
        f"{correlation.n_respondents} calibration personas\n"
        f"  mean |rho| = {float(np.abs(off_diagonal).mean()):.3f}, "
        f"max = {float(off_diagonal.max()):.3f}"
    )

    if report.quarantined:
        console.print(f"\nquarantined {len(report.quarantined)} item(s):")
        for qid in report.quarantined[:10]:
            console.print(f"  {qid}: {report.fits[qid].reason}")


@app.command("freeze")
def freeze_constants(
    tau: float = typer.Option(..., help="Posterior SD threshold for confidence."),
    epsilon: float = typer.Option(0.01, help="EIG floor."),
    bank_version: str = typer.Option("v2"),
    population: str = typer.Option("v2"),
    budget: int = typer.Option(12),
    seed: int = typer.Option(20260807),
    justification: str = typer.Option(..., help="Why these values; goes in the change log."),
) -> None:
    """Write experiment-config.yaml and mark the constants frozen.

    After this, tau, epsilon, budgets, bank version, population version and
    seeds change only with a dated entry in results-log.md, and any published
    number computed under the old values is re-run or retracted.
    """
    from dataclasses import replace as dc_replace

    existing = ExperimentConfig.load()
    config = dc_replace(
        existing,
        tau=tau,
        epsilon=epsilon,
        bank_version=bank_version,
        population_version=population,
        seed_set=[seed],
        budgets=dc_replace(existing.budgets, max_questions=budget),
        frozen=True,
    )
    config.dump(
        change_log=[
            {
                "date": "2026-08-07",
                "change": f"froze tau={tau}, epsilon={epsilon}, budget={budget}",
                "justification": justification,
            }
        ]
    )
    console.print(f"[green]froze[/green] constants -> {EXPERIMENT_CONFIG_PATH}")
    console.print(json.dumps(config.provenance, indent=2))


@app.command("calibrate-tau")
def calibrate_tau(
    population: str = typer.Option("v2"),
    bank_version: str = typer.Option("v2"),
    budget: int = typer.Option(12),
    target: float = typer.Option(0.70, help="Fraction of calibration personas to resolve."),
    seed: int = typer.Option(20260807),
) -> None:
    """Find the tau at which the fixed arm reaches confidence on ~target of
    honest calibration personas at the given budget.

    Set on the *calibration* split, never the eval split — picking a constant
    using the data you will later report against is the circularity this
    project is at pains to avoid.
    """
    from probe.tuning import sweep_tau

    personas, _meta = load_population(population)
    # Honest calibration personas only. Tau describes when an interview has
    # learned enough about a cooperative candidate; anchoring it on bluffers
    # and dodgers would set the threshold by how hard the adversarial subset is
    # to measure, which is a different question and one Phase 5 owns.
    calibration = [
        p for p in personas if p.split == "calibration" and p.behavior is Behavior.HONEST
    ]
    bank = load_bank(bank_version)

    table, chosen = sweep_tau(
        calibration, bank, get_client("sim", seed=seed), budget=budget, target=target, seed=seed
    )
    console.print(f"{'tau':>6}  {'resolved fully':>14}  {'mean SD':>8}")
    for tau, fraction, mean_sd in table:
        marker = "  <-- chosen" if tau == chosen else ""
        console.print(f"{tau:6.2f}  {fraction:13.1%}  {mean_sd:8.3f}{marker}")
    console.print(
        f"\n[green]tau = {chosen:.2f}[/green] puts the fixed arm nearest "
        f"{target:.0%} at budget {budget} on the calibration split"
    )


# --------------------------------------------------------------- population


@population_app.command("generate")
def population_generate(
    n: int = typer.Option(10, help="Number of personas."),
    version: str = typer.Option("v1", help="Population version tag."),
    seed: int = typer.Option(20260801),
    adversarial: bool = typer.Option(False, help="Include the adversarial behaviours."),
    calibration_fraction: float = typer.Option(0.6),
) -> None:
    """Generate personas with hidden ground truth and write them to data/personas.

    The job descriptions are (re)generated alongside, because a population is
    only regenerable from its seed if everything it was conditioned on is too.
    """
    ensure_dirs()
    tax = load_taxonomy()

    jds = default_jds(tax, seed=seed)
    for jd in jds:
        save_jd(jd)

    behaviors = [Behavior.HONEST]
    if adversarial:
        # ~25% adversarial. The cycle is round-robin, so the honest count is
        # what sets the ratio: 18 honest slots against the six adversarial
        # behaviours gives 18/24 = 75% honest. (Three honest slots gives 33%,
        # which is how the first version of this ended up at 65% adversarial.)
        behaviors = [Behavior.HONEST] * 18 + [
            Behavior.BLUFFER,
            Behavior.TERSE,
            Behavior.RAMBLER,
            Behavior.INJECTOR,
            Behavior.DODGER,
            Behavior.OVERCLAIMER,
        ]

    personas = PersonaGenerator(tax, seed=seed).generate(
        n,
        behaviors=behaviors,
        styles=list(MAIN_SWEEP_STYLES),
        jd_ids=[jd.id for jd in jds],
        calibration_fraction=calibration_fraction,
    )
    n_adv = sum(1 for p in personas if p.behavior is not Behavior.HONEST)
    meta = PopulationMeta(
        version=version,
        seed=seed,
        taxonomy_version=tax.version,
        n_personas=len(personas),
        calibration_fraction=calibration_fraction,
        adversarial_fraction=round(n_adv / max(1, len(personas)), 3),
    )
    path = save_population(personas, meta)
    n_cal = sum(1 for p in personas if p.split == "calibration")
    console.print(
        f"[green]wrote[/green] {len(personas)} personas -> {path}\n"
        f"  split: {n_cal} calibration / {len(personas) - n_cal} eval\n"
        f"  adversarial: {n_adv} ({meta.adversarial_fraction:.0%})\n"
        f"  understated competencies per persona: "
        f"{sum(len(p.understated) for p in personas) / max(1, len(personas)):.1f} mean"
    )


@population_app.command("fidelity")
def population_fidelity(
    version: str = typer.Option("v1", help="Population version."),
    bank_version: str = typer.Option("v1-starter"),
    sample: int = typer.Option(100, help="Answers to rate blind."),
    seed: int = typer.Option(20260801),
) -> None:
    """Run the simulator fidelity gate. Exits non-zero on failure.

    Nothing downstream is meaningful until this passes, so it exits non-zero
    rather than printing a warning — a gate that can be ignored is not a gate.
    """
    personas, _meta = load_population(version)
    bank = load_bank(bank_version)
    client = get_client("sim", seed=seed)

    result = run_fidelity_gate(personas, bank, client, sample_size=sample, seed=seed)
    colour = "green" if result.passed else "red"
    console.print(f"[{colour}]{result.summary()}[/{colour}]")

    worst = sorted(result.per_competency_rho.items(), key=lambda kv: kv[1])[:5]
    if worst:
        console.print("\nweakest competencies by rho:")
        for cid, rho in worst:
            console.print(f"  {cid:<42} {rho:+.3f}  tertiles={result.tertile_means.get(cid)}")

    if not result.passed:
        raise typer.Exit(code=1)


# ----------------------------------------------------------------------- run


@app.command("experiment")
def experiment_run(
    arms: str = typer.Option("", help="Comma-separated arms; default is all four."),
    styles: str = typer.Option("", help="Comma-separated styles; default is the main sweep."),
    split: str = typer.Option("eval", help="eval | calibration | all"),
    backend: str = typer.Option("sim", help="sim | fake | anthropic"),
    population: str = typer.Option("", help="Defaults to the frozen population."),
    bank_version: str = typer.Option("", help="Defaults to the frozen bank."),
    traces: str = typer.Option(str(DEFAULT_TRACE_DB)),
    concurrency: int = typer.Option(8),
    style_separation: bool = typer.Option(True),
    followups: bool = typer.Option(True),
    suffix: str = typer.Option("", help="Label appended to run ids for ablations."),
    limit: int = typer.Option(0, help="Cap personas, for smoke runs."),
    seed: int = typer.Option(0, help="Defaults to the frozen seed."),
    yes: bool = typer.Option(False, "--yes", help="Skip the cost confirmation."),
) -> None:
    """Re-run the interview sweep from scratch, printing a cost estimate first.

    A sweep that silently costs three figures is a bad surprise; one that tells
    you first is a decision. Under the offline backend the estimate is zero and
    the prompt is skipped.
    """
    ensure_dirs()
    config = ExperimentConfig.load()
    plan = SweepPlan(
        arms=tuple(a.strip() for a in arms.split(",") if a.strip()) or ARMS,
        styles=tuple(s.strip() for s in styles.split(",") if s.strip()) or MAIN_SWEEP_STYLES,
        split=split,
        backend=backend,
        seed=seed or config.seed_set[0],
        concurrency=concurrency,
        style_separation=style_separation,
        followups_enabled=followups,
        suffix=suffix,
        limit=limit or None,
    )

    personas, meta = load_population(population or config.population_version)
    subjects = [p for p in personas if split in ("all", p.split)][: (limit or None)]
    n_runs = len(subjects) * len(plan.styles) * len(plan.arms)

    if backend == "anthropic":
        from probe.runtime.anthropic_client import estimate_cost

        estimate = estimate_cost(n_runs, turns_per_interview=config.budgets.max_questions)
        console.print(f"[yellow]live backend[/yellow]: {n_runs} interviews, {estimate.render()}")
        if not yes and not typer.confirm("proceed?"):
            raise typer.Exit(code=1)
    else:
        console.print(f"offline backend '{backend}': {n_runs} interviews, $0.00 estimated")

    result = run_sweep(
        plan,
        population_version=population or config.population_version,
        bank_version=bank_version or config.bank_version,
        config=config,
        traces=traces,
    )
    console.print(json.dumps(result.summary(), indent=2))
    if result.outcome.failures:
        console.print(f"[red]{len(result.outcome.failures)} run(s) failed[/red]")
        for label, error in result.outcome.failures[:5]:
            console.print(f"  {label}: {error}")
        raise typer.Exit(code=1)


@app.command("run")
def run_interview(
    persona: str = typer.Option("p001", help="Persona id, or 'stub' for the fixture path."),
    backend: str = typer.Option("sim", help="sim | fake | anthropic"),
    arm: str = typer.Option("fixed", help="Policy arm."),
    population: str = typer.Option("v1"),
    bank_version: str = typer.Option("v1-starter"),
    questions: int = typer.Option(12, help="Question budget."),
    style_separation: bool = typer.Option(True, help="Content-only grading instruction."),
    traces: str = typer.Option(str(DEFAULT_TRACE_DB)),
    seed: int = typer.Option(20260801),
    resume: bool = typer.Option(True),
    show_transcript: bool = typer.Option(False),
) -> None:
    """Run one interview end to end and persist it."""
    ensure_dirs()
    config = _with_question_budget(ExperimentConfig.load(), questions)

    personas, meta = load_population(population)
    matches = [p for p in personas if p.id == persona]
    if not matches:
        console.print(f"[red]no persona {persona!r} in population {population}[/red]")
        raise typer.Exit(code=2)

    bank = load_bank(bank_version)
    store = TraceStore(traces)
    traced = TracedClient(get_client(backend, seed=seed), store=store)

    spec = InterviewSpec(
        persona=matches[0],
        jd=load_jd(matches[0].jd_id),
        arm=arm,
        seed=seed,
        style_separation=style_separation,
    )
    loop = build_interview(
        spec,
        bank=bank,
        config=replace(config, population_version=meta.version, bank_version=bank.version),
        client=traced,
        store=store,
    )
    result = loop.run(resume=resume)

    console.print(render_report(result.report))
    if show_transcript:
        console.print("")
        console.print(result.transcript.render())
    console.print(
        f"\n[green]persisted[/green] run={spec.run_id} turns={len(result.transcript)} "
        f"stop={result.run.stop_reason.value} tokens={result.run.total_tokens} -> {traces}"
    )
    store.close()


def _with_question_budget(config: ExperimentConfig, questions: int) -> ExperimentConfig:
    return replace(config, budgets=replace(config.budgets, max_questions=questions))


if __name__ == "__main__":  # pragma: no cover
    app()


# --------------------------------------------------------------- inspection


@app.command("viewer")
def viewer_show(
    run: str = typer.Option(..., help="Run id to render."),
    traces: str = typer.Option(str(DEFAULT_TRACE_DB)),
) -> None:
    """Render one interview: transcript plus belief trajectory."""
    from probe.report.viewer import render_run

    store = TraceStore(traces, read_only=True)
    try:
        console.print(render_run(store, run))
    finally:
        store.close()


@app.command("demo")
def demo_render(
    traces: str = typer.Option(str(DEFAULT_TRACE_DB)),
    behavior: str = typer.Option("dodger", help="Adversarial behaviour to feature."),
    left: str = typer.Option("fixed"),
    right: str = typer.Option("eig"),
    out: str = typer.Option("analysis/demo-side-by-side.txt"),
) -> None:
    """The D4 deliverable: one adversarial candidate, two arms, side by side.

    A rendering of committed traces, never a re-recording — both runs are
    reconstructed from DuckDB, so what is shown is what happened.
    """
    from probe.report.viewer import render_side_by_side

    config = ExperimentConfig.load()
    personas, _meta = load_population(config.population_version)
    candidates = [
        p for p in personas if p.behavior.value == behavior and p.split == "eval"
    ]
    if not candidates:
        console.print(f"[red]no eval-split persona with behaviour {behavior!r}[/red]")
        raise typer.Exit(code=2)

    persona = candidates[0]
    seed = config.seed_set[0]
    left_id = f"{left}.{persona.id}.{persona.style.id}.s{seed}"
    right_id = f"{right}.{persona.id}.{persona.style.id}.s{seed}"

    store = TraceStore(traces, read_only=True)
    try:
        rendered = render_side_by_side(store, left_id, right_id)
    finally:
        store.close()

    header = (
        f"probe — {left} vs {right} on a {behavior} ({persona.id}, "
        f"{len(persona.understated)} understated competencies)\n"
        f"Both panes are rendered from committed traces.\n\n"
    )
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + rendered + "\n", encoding="utf-8")
    console.print(rendered)
    console.print(f"\n[green]wrote[/green] {path}")
