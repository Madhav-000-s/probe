"""The two planes must not touch.

The ground-truth firewall test greps prompts at runtime. This one checks the
structure that makes such a leak possible in the first place: no module on the
interview plane may import the measurement plane. A firewall is far easier to
keep than to restore, and an import is how it gets breached.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from probe.config import ROOT

#: Packages that conduct an interview. None of them may know what a Persona is.
INTERVIEW_PLANE = ("rubric", "belief", "policy", "bank", "grader", "report")

#: ``probe.runtime`` is mostly interview plane, but two modules are the seam
#: where the harness assembles a run:
#:
#: * ``session.py`` sees a Persona so it can hand the resume to the compiler
#:   and the persona itself to the candidate adapter;
#: * ``llm.py`` names the sim backend in its factory, lazily and by name — the
#:   provider boundary, covered by its own test below.
RUNTIME_ALLOWED = {"session.py", "llm.py"}


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.append(node.module)
    return out


def _modules(package: str) -> list[Path]:
    return sorted((ROOT / "probe" / package).rglob("*.py"))


@pytest.mark.parametrize("package", INTERVIEW_PLANE)
def test_interview_plane_never_imports_the_simulator(package):
    offenders = []
    for path in _modules(package):
        for module in _imports(path):
            if module.startswith("probe.sim"):
                offenders.append(f"{path.relative_to(ROOT)} imports {module}")
    assert not offenders, "interview plane reached into the measurement plane:\n" + "\n".join(
        offenders
    )


def test_runtime_only_touches_the_simulator_at_the_documented_seam():
    offenders = []
    for path in _modules("runtime"):
        if path.name in RUNTIME_ALLOWED:
            continue
        for module in _imports(path):
            if module.startswith("probe.sim"):
                offenders.append(f"{path.relative_to(ROOT)} imports {module}")
    assert not offenders, "\n".join(offenders)


def test_persona_type_is_confined():
    """Grepping for the type name catches what import analysis misses — a
    module that never imports Persona but happily accepts one."""
    offenders = []
    for package in INTERVIEW_PLANE:
        for path in _modules(package):
            text = path.read_text(encoding="utf-8")
            if "Persona" in text or "theta_star" in text:
                offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, f"ground-truth vocabulary on the interview plane: {offenders}"


def test_the_provider_boundary_is_the_only_crossing():
    """``probe.sim.llm_sim`` is reached through the LLMClient protocol, never
    by type. The factory in ``runtime.llm`` imports it lazily and by name, so
    a machine that never runs the simulator never loads it."""
    source = (ROOT / "probe" / "runtime" / "llm.py").read_text(encoding="utf-8")
    assert "from probe.sim.llm_sim import SimLLM" in source
    top_level = [
        m for m in _imports(ROOT / "probe" / "runtime" / "llm.py") if m.startswith("probe.sim")
    ]
    assert top_level == ["probe.sim.llm_sim"], "sim import must stay inside the factory"


def test_public_view_is_the_only_persona_surface_the_loop_sees():
    from probe.models import Persona, StyleProfile

    persona = Persona(
        id="p",
        theta_star={"databases.indexing": 2.5},
        style=StyleProfile(
            id="neutral", verbosity=1.0, hedging=0.1, assertiveness=0.1, l1_transfer=0.0
        ),
        behavior="honest",
        resume="r",
        jd_id="jd",
    )
    assert "theta" not in json.dumps(persona.public_view()).lower()
