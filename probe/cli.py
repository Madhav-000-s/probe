"""The command-line surface.

Thin by design: every command here is a few lines of wiring over a library
call, so that anything the CLI can do, a test or a notebook can do the same
way. Nothing in the pipeline is reachable only through Typer.
"""

from __future__ import annotations

import typer
from rich.console import Console

from probe import __version__
from probe.bank.loader import save_bank, stub_bank
from probe.belief.state import PriorOnlyBelief
from probe.config import DEFAULT_TRACE_DB, ExperimentConfig, ensure_dirs
from probe.grader.base import LLMGrader
from probe.grader.fixtures import length_proportional_grade
from probe.models import LLMRole
from probe.policy.fixed import FixedPolicy
from probe.report.render import render_report
from probe.rubric.taxonomy import load_taxonomy
from probe.runtime.candidate import StubCandidate
from probe.runtime.llm import FakeLLM
from probe.runtime.loop import InterviewLoop
from probe.runtime.tracing import TraceStore, TracedClient, new_run_id

console = Console()

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="probe — adaptive interviewing agent and evaluation harness.",
)
taxonomy_app = typer.Typer(no_args_is_help=True, help="Inspect the competency taxonomy.")
bank_app = typer.Typer(no_args_is_help=True, help="Build and calibrate the question bank.")
app.add_typer(taxonomy_app, name="taxonomy")
app.add_typer(bank_app, name="bank")


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
    per_competency: int = typer.Option(2, help="Items per competency."),
    version: str = typer.Option("v0-stub", help="Bank version tag."),
    seed: int = typer.Option(20260801, help="Unused for the stub bank; kept for parity."),
) -> None:
    """Write a deterministic bank with authoring-default item parameters."""
    ensure_dirs()
    tax = load_taxonomy()
    bank = stub_bank(tax, per_competency=per_competency, version=version)
    path = save_bank(bank)
    console.print(f"[green]wrote[/green] {len(bank)} questions -> {path}")


# ----------------------------------------------------------------------- run


@app.command("run")
def run_interview(
    fake: bool = typer.Option(False, "--fake", help="Use the canned offline backend."),
    persona: str = typer.Option("stub", help="Candidate identifier."),
    questions: int = typer.Option(5, help="Question budget for this interview."),
    competencies: int = typer.Option(6, help="Rubric size."),
    traces: str = typer.Option(str(DEFAULT_TRACE_DB), help="DuckDB trace file."),
    resume: bool = typer.Option(True, help="Resume a partially-persisted run."),
    show_transcript: bool = typer.Option(False, help="Print the full transcript."),
) -> None:
    """Run one interview end to end and persist it.

    ``--fake`` is the Phase 0 path: no model, no network, fully deterministic,
    and every byte of it reconstructable from the trace afterwards.
    """
    if not fake:
        console.print(
            "[yellow]only --fake is wired at this phase; "
            "later phases add the sim and anthropic backends[/yellow]"
        )
        raise typer.Exit(code=2)

    ensure_dirs()
    config = ExperimentConfig.load()
    config = _with_question_budget(config, questions)

    tax = load_taxonomy()
    rubric = tax.stub_rubric(candidate_id=persona, n=competencies)
    bank = stub_bank(tax, competency_ids=rubric.ids, per_competency=2)

    client = FakeLLM(by_role={LLMRole.GRADE: length_proportional_grade()}, strict=False)
    store = TraceStore(traces)
    traced = TracedClient(client, store=store)

    run_id = new_run_id("fixed", persona, "neutral", config.seed_set[0])
    traced.bind_run(run_id)

    loop = InterviewLoop(
        rubric=rubric,
        bank=bank,
        policy=FixedPolicy(rubric),
        belief=PriorOnlyBelief(rubric),
        grader=LLMGrader(traced),
        candidate=StubCandidate(candidate_id=persona),
        config=config,
        store=store,
        run_id=run_id,
        seed=config.seed_set[0],
        grader_model=client.model,
    )
    result = loop.run(resume=resume)

    console.print(render_report(result.report))
    if show_transcript:
        console.print("")
        console.print(result.transcript.render())
    console.print(
        f"\n[green]persisted[/green] run={run_id} turns={len(result.transcript)} "
        f"stop={result.run.stop_reason.value} -> {traces}"
    )
    store.close()


def _with_question_budget(config: ExperimentConfig, questions: int) -> ExperimentConfig:
    from dataclasses import replace

    return replace(config, budgets=replace(config.budgets, max_questions=questions))


if __name__ == "__main__":  # pragma: no cover
    app()
