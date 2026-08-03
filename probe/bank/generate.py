"""Building question banks.

Item *text* is generated from the taxonomy; item *parameters* are fitted from
data in Phase 3. Everything here therefore ships with authoring defaults and
``calibrated=False``, and a results table that quotes a bank version is
quoting which of the two it ran against.

Bank size is a real experimental parameter, not a detail. Too few items and
the adaptive policy runs out of things to ask before the fixed script does,
which would hand the comparison a win it did not earn.
"""

from __future__ import annotations

from probe.bank.loader import build_question
from probe.models import QuestionBank
from probe.rubric.taxonomy import Taxonomy

#: Phase 1: enough to interview against, small enough to inspect by hand.
STARTER_COMPETENCIES = 20
STARTER_PER_COMPETENCY = 3

#: Phase 3: the full instrument. 4 items per competency across 50 competencies
#: gives 200, with every probe family represented for every competency so the
#: repeat-family penalty always has an alternative to reach for.
FULL_PER_COMPETENCY = 4


def _questions_for(taxonomy: Taxonomy, competency_ids: list[str], per_competency: int):
    for cid in competency_ids:
        node = taxonomy.get(cid)
        for i in range(per_competency):
            yield build_question(
                competency_id=cid,
                label=node.label,
                concepts=node.concepts,
                family=node.probe_families[i % len(node.probe_families)],
                index=i,
            )


def starter_competencies(taxonomy: Taxonomy, jds=None) -> list[str]:
    """Which competencies the starter bank covers.

    The union of what the job descriptions actually ask for, not the first N
    taxonomy entries. Taking a prefix produced a bank that was entirely backend
    while one of the two JDs hired for data/ML, so those interviews compiled a
    rubric the bank had no items for and terminated at turn zero. A starter
    bank is authored for the roles being hired for; anything else is an
    instrument that cannot measure its own subjects.
    """
    from probe.jd import default_jds

    jds = jds if jds is not None else default_jds(taxonomy)
    ids: list[str] = []
    for jd in jds:
        for cid in jd.required:
            if cid not in ids:
                ids.append(cid)
    # Deterministic order, and stable if a JD's sample changes slightly.
    order = {cid: i for i, cid in enumerate(taxonomy.ids)}
    return sorted(ids, key=lambda cid: order[cid])[:STARTER_COMPETENCIES]


def starter_bank(
    taxonomy: Taxonomy,
    competency_ids: list[str] | None = None,
    version: str = "v1-starter",
    jds=None,
) -> QuestionBank:
    """60 items over 20 competencies — the Phase 1 instrument."""
    ids = competency_ids if competency_ids is not None else starter_competencies(taxonomy, jds)
    return QuestionBank(
        version=version,
        taxonomy_version=taxonomy.version,
        questions=list(_questions_for(taxonomy, ids, STARTER_PER_COMPETENCY)),
    )


def full_bank(taxonomy: Taxonomy, version: str = "v2-raw") -> QuestionBank:
    """200 items over the whole taxonomy — the Phase 3 instrument, before
    calibration replaces the authoring defaults."""
    return QuestionBank(
        version=version,
        taxonomy_version=taxonomy.version,
        questions=list(_questions_for(taxonomy, taxonomy.ids, FULL_PER_COMPETENCY)),
    )


def bank_summary(bank: QuestionBank) -> dict[str, object]:
    """Shape of a bank, for the CLI and for the results table's provenance."""
    by_family: dict[str, int] = {}
    for q in bank.questions:
        by_family[q.probe_family.value] = by_family.get(q.probe_family.value, 0) + 1
    competencies = {q.competency_id for q in bank.questions}
    calibrated = sum(1 for q in bank.questions if q.grm.calibrated)
    return {
        "version": bank.version,
        "n_questions": len(bank),
        "n_competencies": len(competencies),
        "by_family": dict(sorted(by_family.items())),
        "calibrated": calibrated,
        "quarantined": sum(1 for q in bank.questions if q.grm.quarantined),
        "mean_expected_seconds": round(
            sum(q.expected_seconds for q in bank.questions) / max(1, len(bank)), 1
        ),
    }
