"""The anchor set, and an honest label on what it is.

PLAN.md calls for a human gold set: 60–100 answers graded blind in one sitting,
with Cohen's kappa against the LLM grader, released as the scarcest artefact in
the repo. LLM-graded corpora are everywhere; blind human anchors are not.

**No human graded this one.** It was produced by an independent rater — a
different estimator from the grader, with different noise, no style term and no
position bias — and calling it a human gold set would be a lie that the rest of
the repo's credibility does not survive. It is a *stand-in*: it exercises the
release format, the kappa computation and the analysis script end to end, so
that dropping real labels in later is a data swap rather than a rewrite.

What it can and cannot support:

* it **can** show that two independently-parameterised estimators reading the
  same answers agree at a measurable rate, and that the machinery to compute
  and release that agreement works;
* it **cannot** anchor the grader to human judgement, which is the entire point
  of the artefact. The README states this before quoting the number, and D5's
  acceptance criteria are marked unmet rather than quietly redefined.

The labelling protocol below is the one a human pass would follow, written down
now so that a later human pass is comparable to this one.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from probe.config import GOLD_DIR
from probe.grader.base import LLMGrader
from probe.models import LLMRole, Question, Transcript
from probe.runtime.llm import LLMRequest

EMPTY = Transcript(run_id="gold", candidate_id="-", arm="gold")

#: The protocol a labelling pass follows. Released alongside the labels so the
#: conditions are reproducible.
PROTOCOL = {
    "blind_to_llm_score": True,
    "blind_to_theta_star": True,
    "single_sitting": True,
    "anchor_order": "ascending, fixed for every item",
    "rater": "independent estimator (NOT human) — see module docstring",
    "answers_shown": "question text, five rubric anchors, answer text",
    "scale": "1-5, whole numbers only",
}


@dataclass
class GoldItem:
    answer_id: str
    question_id: str
    competency_id: str
    question_text: str
    answer: str
    anchors: list[str]
    reference_score: int
    llm_score: int
    llm_confidence: float
    llm_flags: list[str] = field(default_factory=list)


def cohens_kappa(a: Sequence[int], b: Sequence[int], categories: Sequence[int] = (1, 2, 3, 4, 5)) -> float:
    """Cohen's kappa: agreement corrected for chance.

    Raw agreement flatters any pair of raters on a five-point scale where most
    answers land in the middle. Kappa is the number that survives that.
    """
    a, b = list(a), list(b)
    if len(a) != len(b) or not a:
        return float("nan")
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n

    expected = 0.0
    for c in categories:
        expected += (a.count(c) / n) * (b.count(c) / n)
    if expected >= 1.0:
        return float("nan")
    return (observed - expected) / (1.0 - expected)


def quadratic_weighted_kappa(a: Sequence[int], b: Sequence[int], k: int = 5) -> float:
    """Kappa that treats a 4-vs-5 disagreement as milder than 1-vs-5.

    The right statistic for an ordinal rubric, reported alongside the
    unweighted one because the plan's threshold is stated on the unweighted
    form.
    """
    a, b = list(a), list(b)
    if len(a) != len(b) or not a:
        return float("nan")
    observed = np.zeros((k, k))
    for x, y in zip(a, b, strict=True):
        observed[x - 1, y - 1] += 1
    observed /= observed.sum()

    hist_a = np.array([a.count(c) for c in range(1, k + 1)], dtype=float) / len(a)
    hist_b = np.array([b.count(c) for c in range(1, k + 1)], dtype=float) / len(b)
    expected = np.outer(hist_a, hist_b)

    weights = np.array([[((i - j) ** 2) / ((k - 1) ** 2) for j in range(k)] for i in range(k)])
    denominator = float((weights * expected).sum())
    if denominator <= 0:
        return float("nan")
    return 1.0 - float((weights * observed).sum()) / denominator


def reference_rating(client, question: Question, answer: str, seed: int) -> int:
    """An independent read of one answer.

    Deliberately routed through the ``BLIND_RATE`` role rather than the grader:
    a "second opinion" that shared the grader's parameters would agree with it
    by construction and the kappa would measure nothing.
    """
    request = LLMRequest(
        role=LLMRole.BLIND_RATE,
        prompt=(
            "Rate this interview answer from 1 to 5 on technical substance "
            "alone. You are not told who wrote it, what they claim to know, or "
            "how anyone else scored it.\n\n"
            f"Question: {question.text}\n"
            + "\n".join(f"  {a.level}: {a.descriptor}" for a in sorted(question.anchors, key=lambda x: x.level))
            + f"\n\nAnswer:\n{answer}\n\n"
            'Return JSON: {"rating": <1-5>}'
        ),
        seed=seed,
        temperature=0.0,
        context={"answer": answer, "concept_pool": list(question.anchor(5).required_concepts)},
    )
    return int(json.loads(client.complete(request).text)["rating"])


def build_gold_set(
    samples: Sequence[tuple[str, Question, str]],
    grader: LLMGrader,
    reference_client,
    *,
    seed: int = 20260808,
) -> list[GoldItem]:
    items: list[GoldItem] = []
    for i, (answer_id, question, answer) in enumerate(samples):
        grade = grader.grade(question, answer, EMPTY, seed=seed).grade
        if grade is None:
            continue
        items.append(
            GoldItem(
                answer_id=answer_id,
                question_id=question.id,
                competency_id=question.competency_id,
                question_text=question.text,
                answer=answer,
                anchors=[a.descriptor for a in sorted(question.anchors, key=lambda x: x.level)],
                reference_score=reference_rating(reference_client, question, answer, seed + i),
                llm_score=grade.score,
                llm_confidence=grade.confidence,
                llm_flags=[f.value for f in grade.flags],
            )
        )
    return items


def kappa_report(items: Sequence[GoldItem]) -> dict[str, float]:
    reference = [i.reference_score for i in items]
    llm = [i.llm_score for i in items]
    exact = sum(a == b for a, b in zip(reference, llm, strict=True)) / max(1, len(items))
    within_one = sum(abs(a - b) <= 1 for a, b in zip(reference, llm, strict=True)) / max(1, len(items))
    return {
        "n": len(items),
        "cohens_kappa": round(cohens_kappa(reference, llm), 4),
        "quadratic_weighted_kappa": round(quadratic_weighted_kappa(reference, llm), 4),
        "exact_agreement": round(exact, 4),
        "within_one_agreement": round(within_one, 4),
        "mean_reference": round(float(np.mean(reference)), 3) if items else float("nan"),
        "mean_llm": round(float(np.mean(llm)), 3) if items else float("nan"),
    }


def release(items: Sequence[GoldItem], directory: Path | None = None) -> tuple[Path, Path]:
    """Write the labels and the protocol.

    No persona ids, no ability values, nothing that could identify which
    simulated candidate produced an answer — the release is answers and grades
    only, exactly as D5 requires.
    """
    directory = directory or GOLD_DIR
    directory.mkdir(parents=True, exist_ok=True)

    csv_path = directory / "gold-set.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "answer_id",
                "competency_id",
                "question_text",
                "answer",
                "anchors",
                "reference_score",
                "llm_score",
                "llm_confidence",
                "llm_flags",
            ]
        )
        for item in items:
            writer.writerow(
                [
                    item.answer_id,
                    item.competency_id,
                    item.question_text,
                    item.answer,
                    " | ".join(item.anchors),
                    item.reference_score,
                    item.llm_score,
                    f"{item.llm_confidence:.3f}",
                    ",".join(item.llm_flags),
                ]
            )

    meta_path = directory / "gold-set-protocol.json"
    meta_path.write_text(
        json.dumps({"protocol": PROTOCOL, "agreement": kappa_report(items)}, indent=2),
        encoding="utf-8",
    )
    return csv_path, meta_path
