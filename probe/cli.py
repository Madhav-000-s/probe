"""The command-line surface.

Thin by design: every command here is a few lines of wiring over a library
call, so anything the CLI can do, a test or a notebook can do the same way.
Nothing in the pipeline is reachable only through Typer.
"""

from __future__ import annotations

import json
from dataclasses import replace

import typer
from rich.console import Console

from probe import __version__
from probe.bank.generate import bank_summary, full_bank, starter_bank
from probe.bank.loader import load_bank, save_bank
from probe.config import DEFAULT_TRACE_DB, ExperimentConfig, ensure_dirs
from probe.jd import default_jds, load_jd, save_jd
from probe.models import Behavior
from probe.report.render import render_report
from probe.rubric.taxonomy import load_taxonomy
from probe.runtime.llm import get_client
from probe.runtime.session import InterviewSpec, build_interview
from probe.runtime.tracing import TracedClient, TraceStore
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
        # ~25% adversarial: three honest slots for every one of the six others.
        behaviors = [Behavior.HONEST] * 3 + [
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
