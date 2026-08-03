"""The simulator fidelity gate — the project's keystone.

Everything downstream assumes one thing: that a simulated candidate's answers
actually encode their hidden ability. If they do not, then recovery,
efficiency, fairness and robustness are all measuring noise dressed up as a
result, and nothing in the repo means anything.

So it gets tested directly, before anything downstream runs. A blind rater
(a different, cleaner estimator than the grader — see
:meth:`probe.sim.llm_sim.SimLLM._blind_rate`) scores a sample of generated
answers with no access to ``theta_star``. The correlation between those blind
ratings and the true abilities has to clear a threshold, and the relationship
has to be visibly monotone per competency rather than merely correlated in
aggregate.

The plan is explicit that a failure here stops the project rather than being
noted as a caveat. The gate is implemented to fail loudly for that reason.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from probe.models import LLMRole, Persona, QuestionBank
from probe.runtime.llm import LLMRequest
from probe.sim.answers import answer_seed, compose_answer
from probe.sim.style import STYLE_PRESETS

#: Pooled Spearman correlation the gate demands. From PLAN.md.
FIDELITY_RHO_THRESHOLD = 0.60
#: Minimum fraction of competencies whose mean rating is monotone in ability
#: across tertiles. Guards against a pooled correlation carried entirely by a
#: few competencies while the rest are flat.
MONOTONE_FRACTION_THRESHOLD = 0.70
#: Answers a competency needs before its monotonicity is assessed. Three
#: tertiles of two answers each is a coin flip, and an eligibility bar that low
#: made the gate report noise as a failure — which is how the bar got raised.
MIN_PER_COMPETENCY = 12


@dataclass
class FidelityResult:
    rho: float
    p_value: float
    n: int
    monotone_fraction: float
    per_competency_rho: dict[str, float] = field(default_factory=dict)
    tertile_means: dict[str, list[float]] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return (
            self.rho >= FIDELITY_RHO_THRESHOLD
            and self.monotone_fraction >= MONOTONE_FRACTION_THRESHOLD
        )

    def summary(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        return (
            f"fidelity gate {verdict}: rho={self.rho:.3f} "
            f"(threshold {FIDELITY_RHO_THRESHOLD}), n={self.n}, "
            f"monotone competencies={self.monotone_fraction:.0%} "
            f"(threshold {MONOTONE_FRACTION_THRESHOLD:.0%}), p={self.p_value:.2e}"
        )


def _blind_rate(client, answer: str, concept_pool: list[str], seed: int) -> int:
    request = LLMRequest(
        role=LLMRole.BLIND_RATE,
        prompt=(
            "Rate this interview answer from 1 to 5 on technical substance "
            "alone. You are not told who wrote it, what they claim to know, or "
            "how anyone else scored it.\n\n"
            f"Answer:\n{answer}\n\n"
            'Return JSON: {"rating": <1-5>}'
        ),
        seed=seed,
        temperature=0.0,
        context={"answer": answer, "concept_pool": concept_pool},
    )
    return int(json.loads(client.complete(request).text)["rating"])


def run_fidelity_gate(
    personas: list[Persona],
    bank: QuestionBank,
    client,
    *,
    sample_size: int = 240,
    seed: int = 20260801,
    hold_style_neutral: bool = True,
) -> FidelityResult:
    """Generate answers, rate them blind, correlate with ``theta_star``.

    Sampling is over ``(persona, question)`` pairs rather than whole
    interviews, because the question being asked is "does ability show up in
    an answer", not "does an interview recover ability" — that second question
    is what the recovery metric is for, and conflating them would let a good
    policy paper over a bad simulator.

    ``hold_style_neutral`` matters more than it looks. This gate asks whether
    ability reaches the *content* of an answer. Style-induced recognition loss
    — a paraphrased concept an exact matcher misses — is a real effect, but it
    is Q3's subject, and letting it depress the fidelity number here would
    conflate "the simulator does not encode ability" with "the extractor
    cannot always see it". Run with ``hold_style_neutral=False`` to measure the
    combined figure; the fairness suite reports the difference on purpose.
    """
    rng = np.random.default_rng(seed)
    live = bank.live()
    pairs = [(p, q) for p in personas for q in live if q.competency_id in p.theta_star]
    if not pairs:
        raise ValueError("no (persona, question) pairs share a competency")

    idx = rng.choice(len(pairs), size=min(sample_size, len(pairs)), replace=False)
    neutral = STYLE_PRESETS["neutral"]

    thetas: list[float] = []
    ratings: list[int] = []
    by_competency: dict[str, list[tuple[float, int]]] = {}

    for i in idx:
        persona, question = pairs[int(i)]
        theta = persona.ability(question.competency_id)
        style = neutral if hold_style_neutral else persona.style
        text, _level, _plan = compose_answer(
            question=question,
            theta=theta,
            behavior=persona.behavior,
            style=style,
            distractor_pool=[],
            seed=answer_seed(persona.id, question.id, style.id, seed),
        )
        rating = _blind_rate(
            client, text, list(question.anchor(5).required_concepts), seed=int(i)
        )
        thetas.append(theta)
        ratings.append(rating)
        by_competency.setdefault(question.competency_id, []).append((theta, rating))

    rho, p_value = stats.spearmanr(thetas, ratings)

    per_competency: dict[str, float] = {}
    tertile_means: dict[str, list[float]] = {}
    monotone = 0
    eligible = 0
    for cid, rows in by_competency.items():
        if len(rows) < MIN_PER_COMPETENCY:
            continue
        eligible += 1
        t = np.array([r[0] for r in rows])
        r = np.array([r[1] for r in rows], dtype=float)
        c, _ = stats.spearmanr(t, r)
        per_competency[cid] = float(c) if np.isfinite(c) else 0.0

        order = np.argsort(t)
        thirds = np.array_split(r[order], 3)
        means = [float(np.mean(x)) if len(x) else float("nan") for x in thirds]
        tertile_means[cid] = means
        if all(
            means[i] <= means[i + 1] + 1e-9 for i in range(len(means) - 1)
        ) and not any(np.isnan(means)):
            monotone += 1

    return FidelityResult(
        rho=float(rho),
        p_value=float(p_value),
        n=len(thetas),
        monotone_fraction=(monotone / eligible) if eligible else 0.0,
        per_competency_rho=dict(sorted(per_competency.items())),
        tertile_means=tertile_means,
    )
