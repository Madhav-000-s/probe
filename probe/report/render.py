"""Rendering an interview report.

Plain text, deliberately. The report is an artefact a human reads to decide
whether they believe the score, so every competency line carries the interval
and the evidence count that produced it — a bare number would hide exactly the
uncertainty the whole belief state exists to represent.
"""

from __future__ import annotations

from probe.models import InterviewReport


def render_report(report: InterviewReport, show_evidence: bool = True) -> str:
    lines: list[str] = [
        f"Interview report — {report.candidate_id}  (arm={report.arm}, run={report.run_id})",
        f"Questions asked: {report.n_questions}    Stopped: {report.stop_reason.value}"
        + ("    [PARTIAL]" if report.partial else ""),
        "",
        f"{'competency':<40} {'mean':>6} {'sd':>6} {'80% CI':>16} {'n':>3} {'req':>4}  status",
        "-" * 92,
    ]
    for cid, v in sorted(report.per_competency.items()):
        status = "confident" if v.confident else "uncertain"
        if v.n_questions == 0:
            status = "unprobed"
        ci = f"[{v.ci80[0]:+.2f},{v.ci80[1]:+.2f}]"
        lines.append(
            f"{cid:<40} {v.posterior_mean:>+6.2f} {v.posterior_sd:>6.2f} {ci:>16} "
            f"{v.n_questions:>3} {v.required_level:>4}  {status}"
        )
        if v.flags:
            lines.append(f"{'':<40} flags: {', '.join(f.value for f in v.flags)}")
        if show_evidence and v.evidence:
            quote = v.evidence[0].text.replace("\n", " ")
            lines.append(f"{'':<40} evidence: “{quote[:70]}”")

    if report.notes:
        lines += ["", "Notes:"] + [f"  - {n}" for n in report.notes]
    return "\n".join(lines)


def render_belief_trajectory(turns, competency_ids: list[str], width: int = 28) -> str:
    """A per-turn sparkline of posterior mean and SD.

    Shown next to the transcript in the trace viewer so the *why* of each
    question choice is visible: the policy went where the band was widest.
    """
    lines = [f"{'turn':<5} {'competency':<38} {'mean':>6} {'sd':>6}  band"]
    for t in turns:
        snap = t.belief_after
        cid = t.competency_id
        mean, sd = snap.means[cid], snap.sds[cid]
        lo = max(0, int((mean - sd + 3) / 6 * width))
        hi = min(width, max(lo + 1, int((mean + sd + 3) / 6 * width)))
        bar = "".join("█" if lo <= i < hi else "·" for i in range(width))
        lines.append(f"{t.turn_idx:<5} {cid:<38} {mean:>+6.2f} {sd:>6.2f}  {bar}")
    return "\n".join(lines)
