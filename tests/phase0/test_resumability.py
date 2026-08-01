"""A killed run resumes from the last committed turn, exactly once per turn.

800 interviews on a laptop makes this non-negotiable. The interesting failure
is not "resume loses work" — it is "resume duplicates the turn that was
persisted just before the kill", which silently double-counts evidence and
corrupts every metric computed from the trace.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from probe.runtime.candidate import StubCandidate
from probe.runtime.tracing import TraceStore

from ._crash_runner import CRASH_CODE, build_loop

RUNNER = Path(__file__).parent / "_crash_runner.py"


def _kill_after(trace_db: Path, run_id: str, turns: int) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(RUNNER), str(trace_db), run_id, str(turns)],
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_child_process_really_dies(trace_file):
    proc = _kill_after(trace_file, "resume-1", turns=2)
    assert proc.returncode == CRASH_CODE, proc.stderr


def test_resumed_run_completes_with_exactly_one_record_per_turn(trace_file):
    run_id = "resume-2"
    assert _kill_after(trace_file, run_id, turns=2).returncode == CRASH_CODE

    with TraceStore(trace_file) as store:
        assert store.last_turn_idx(run_id) == 1, "two turns should be durable"

    loop = build_loop(str(trace_file), run_id, StubCandidate(candidate_id="p-resume"))
    result = loop.run(resume=True)

    assert result.resumed_from == 1
    assert len(result.transcript) == 5

    with TraceStore(trace_file) as store:
        rows = store.df(
            "SELECT turn_idx, count(*) AS n FROM turns WHERE run_id = ? GROUP BY turn_idx ORDER BY turn_idx",
            [run_id],
        )
    assert list(rows["turn_idx"]) == [0, 1, 2, 3, 4]
    assert set(rows["n"]) == {1}, "idempotency: no turn may be persisted twice"


def test_resume_replays_evidence_into_the_belief_state(trace_file):
    """Resume must rebuild the posterior through the same update path a fresh
    run uses, not restore a snapshot — otherwise resume is a second, untested
    inference implementation."""
    run_id = "resume-3"
    assert _kill_after(trace_file, run_id, turns=2).returncode == CRASH_CODE

    loop = build_loop(str(trace_file), run_id, StubCandidate(candidate_id="p-resume"))
    result = loop.run(resume=True)

    observed = sum(loop.belief.n_observations.values())
    assert observed == 5 == len(result.transcript)


def test_resumed_transcript_matches_a_clean_run(trace_file, tmp_path):
    """The whole point: a resumed run is indistinguishable from one that was
    never interrupted."""
    crashed_id = "resume-4"
    assert _kill_after(trace_file, crashed_id, turns=2).returncode == CRASH_CODE
    resumed = build_loop(str(trace_file), crashed_id, StubCandidate(candidate_id="p-resume")).run()

    clean_db = tmp_path / "clean.duckdb"
    clean = build_loop(str(clean_db), "clean-1", StubCandidate(candidate_id="p-resume")).run()

    def normalise(text: str, run_id: str) -> str:
        return text.replace(run_id, "<run>")

    assert normalise(resumed.transcript.render(), crashed_id) == normalise(
        clean.transcript.render(), "clean-1"
    )


def test_resuming_a_finished_run_is_a_no_op(trace_file):
    run_id = "resume-5"
    first = build_loop(str(trace_file), run_id, StubCandidate(candidate_id="p-resume")).run()
    second = build_loop(str(trace_file), run_id, StubCandidate(candidate_id="p-resume")).run()

    assert len(first.transcript) == len(second.transcript) == 5
    with TraceStore(trace_file) as store:
        assert len(store.load_turns(run_id)) == 5


@pytest.mark.parametrize("crash_after", [1, 3, 4])
def test_resume_from_any_point(trace_file, crash_after):
    run_id = f"resume-any-{crash_after}"
    assert _kill_after(trace_file, run_id, turns=crash_after).returncode == CRASH_CODE

    result = build_loop(str(trace_file), run_id, StubCandidate(candidate_id="p-resume")).run()
    assert result.resumed_from == crash_after - 1
    assert len(result.transcript) == 5
