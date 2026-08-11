"""Rendering a run back out of the trace store.

Two things, both of which exist because a results table is not evidence on its
own — somebody has to be able to look at an actual interview and see why the
policy did what it did.

* :func:`render_run` — one interview: every question, answer, grade and the
  belief trajectory beside it.
* :func:`render_side_by_side` — the same candidate under two arms, which is
  the D4 deliverable: a dodger with a resume gap, the fixed script marching on
  while the adaptive arm re-probes the dodge.

Both are renderings of committed traces. Nothing is re-run, so what you see is
what happened.
"""

from __future__ import annotations

from probe.models import Transcript
from probe.runtime.tracing import TraceStore


def _band(mean: float, sd: float, width: int = 20) -> str:
    lo = max(0, int((mean - sd + 3) / 6 * width))
    hi = min(width, max(lo + 1, int((mean + sd + 3) / 6 * width)))
    return "".join("#" if lo <= i < hi else "." for i in range(width))


def render_run(store: TraceStore, run_id: str, max_answer_chars: int = 220) -> str:
    run = store.load_run(run_id)
    if run is None:
        raise KeyError(run_id)
    turns = store.load_turns(run_id)

    lines = [
        f"=== {run_id}",
        f"    arm={run.arm}  persona={run.persona_id}  style={run.style_id}",
        f"    bank={run.bank_version}  stop={run.stop_reason.value if run.stop_reason else '?'}"
        f"  turns={len(turns)}  tokens={run.total_tokens}",
        "",
    ]
    for turn in turns:
        snap = turn.belief_after
        mean = snap.means.get(turn.competency_id, 0.0)
        sd = snap.sds.get(turn.competency_id, 0.0)
        eig = f"{turn.eig_at_selection:.3f}" if turn.eig_at_selection is not None else "  -  "
        flags = (
            ",".join(f.value for f in turn.grade.flags)
            if turn.grade and turn.grade.flags
            else ""
        )
        score = turn.grade.score if turn.grade else "-"

        lines.append(f"  [{turn.turn_idx:>2}] {turn.competency_id}   EIG={eig}")
        lines.append(f"       Q: {turn.question_text[:150]}")
        answer = turn.answer.replace("\n", " ")
        if len(answer) > max_answer_chars:
            answer = answer[:max_answer_chars] + "…"
        lines.append(f"       A: {answer}")
        lines.append(
            f"       -> score {score}   theta {mean:+.2f} +- {sd:.2f}  "
            f"[{_band(mean, sd)}]" + (f"  flags: {flags}" if flags else "")
        )
        if turn.selection_reason:
            lines.append(f"       why: {turn.selection_reason}")
        lines.append("")
    return "\n".join(lines)


def render_side_by_side(
    store: TraceStore, left_run: str, right_run: str, width: int = 58
) -> str:
    """Two arms on the same candidate, turn for turn.

    The belief band beside each turn is what makes the comparison legible: the
    adaptive arm's questions go where its band is widest, and you can watch it
    happen.
    """
    left, right = store.load_run(left_run), store.load_run(right_run)
    if left is None or right is None:
        raise KeyError(f"{left_run} / {right_run}")
    left_turns, right_turns = store.load_turns(left_run), store.load_turns(right_run)

    def cell(turn) -> list[str]:
        if turn is None:
            return ["", "", ""]
        snap = turn.belief_after
        mean = snap.means.get(turn.competency_id, 0.0)
        sd = snap.sds.get(turn.competency_id, 0.0)
        score = turn.grade.score if turn.grade else "-"
        flags = (
            " " + ",".join(f.value[:12] for f in turn.grade.flags)
            if turn.grade and turn.grade.flags
            else ""
        )
        return [
            turn.competency_id[:width],
            f"  score {score}  theta {mean:+.2f}+-{sd:.2f}{flags}"[:width],
            f"  {_band(mean, sd)}"[:width],
        ]

    header = (
        f"{left.arm + '  (' + left.persona_id + ')':<{width}}  |  "
        f"{right.arm + '  (' + right.persona_id + ')'}"
    )
    lines = [header, "-" * (width * 2 + 5)]

    for i in range(max(len(left_turns), len(right_turns))):
        a = cell(left_turns[i] if i < len(left_turns) else None)
        b = cell(right_turns[i] if i < len(right_turns) else None)
        lines.append(f"{('[' + str(i) + '] ' + a[0]):<{width}}  |  [{i}] {b[0]}")
        lines.append(f"{a[1]:<{width}}  |  {b[1]}")
        lines.append(f"{a[2]:<{width}}  |  {b[2]}")
        lines.append("")

    lines.append(
        f"stopped: {left.stop_reason.value if left.stop_reason else '?'} "
        f"after {len(left_turns)}"
        f"   |   {right.stop_reason.value if right.stop_reason else '?'} "
        f"after {len(right_turns)}"
    )
    return "\n".join(lines)


def transcript_of(store: TraceStore, run_id: str) -> Transcript:
    return store.load_transcript(run_id)
