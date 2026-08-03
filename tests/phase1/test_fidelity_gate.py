"""The fidelity gate — the hard gate.

If a simulated candidate's answers do not encode their hidden ability, every
downstream number is decoration. The plan is explicit that failure here stops
the project rather than being noted as a caveat, so this test is a gate and
not a diagnostic.

It failed twice during Phase 1 before passing. Both failures were real bugs
(terse styling deleting content; the gate's own monotonicity check running on
tertiles of two) and both fixes are covered by their own tests elsewhere. The
history is in results-log.md.
"""

from __future__ import annotations

import pytest

from probe.sim.fidelity import (
    FIDELITY_RHO_THRESHOLD,
    MONOTONE_FRACTION_THRESHOLD,
    run_fidelity_gate,
)


@pytest.mark.gate
@pytest.mark.slow
def test_fidelity_gate_passes(personas, starter, sim):
    result = run_fidelity_gate(personas, starter, sim, sample_size=400, seed=20260803)

    assert result.n == 400
    assert result.rho >= FIDELITY_RHO_THRESHOLD, result.summary()
    assert result.monotone_fraction >= MONOTONE_FRACTION_THRESHOLD, result.summary()
    assert result.p_value < 1e-6
    assert result.passed


@pytest.mark.slow
def test_gate_reports_per_competency_detail(personas, starter, sim):
    """A pooled correlation can be carried by a handful of competencies while
    the rest are flat, so the gate has to be able to show its working."""
    result = run_fidelity_gate(personas, starter, sim, sample_size=400, seed=20260803)

    assert result.per_competency_rho
    assert set(result.tertile_means) <= set(result.per_competency_rho) | set(
        result.tertile_means
    )
    for means in result.tertile_means.values():
        assert len(means) == 3


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore:An input array is constant:")
def test_gate_can_fail(personas, starter, sim, monkeypatch):
    """A gate that cannot fail is decoration.

    Sever the channel — rate every answer identically regardless of content —
    and the gate must reject it.
    """
    import json

    from probe.models import LLMRole
    from probe.runtime.llm import LLMResponse

    original = sim.complete

    def blind_rate_is_constant(request):
        if request.role is LLMRole.BLIND_RATE:
            return LLMResponse(text=json.dumps({"rating": 3, "n_concepts": 0}), model="broken")
        return original(request)

    monkeypatch.setattr(sim, "complete", blind_rate_is_constant)
    result = run_fidelity_gate(personas, starter, sim, sample_size=120, seed=20260803)

    assert not result.passed
    assert "FAIL" in result.summary()


def test_thresholds_match_the_plan():
    assert FIDELITY_RHO_THRESHOLD == 0.60
    assert MONOTONE_FRACTION_THRESHOLD == 0.70
