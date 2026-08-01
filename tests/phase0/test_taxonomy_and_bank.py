"""The fixed taxonomy and the deterministic stub bank."""

from __future__ import annotations

from probe.bank.loader import LEVEL_THRESHOLDS, POOL_SIZE, load_bank, save_bank, stub_bank
from probe.rubric.taxonomy import HIGH_UNCERTAINTY_VAR, load_taxonomy


def test_taxonomy_is_structurally_valid(taxonomy):
    assert taxonomy.validate() == []


def test_taxonomy_size_and_coverage(taxonomy):
    assert 30 <= len(taxonomy) <= 50
    families = {n.family.value for n in taxonomy}
    assert families == {"backend", "data_ml"}, "spec calls for two role families"
    assert len(taxonomy.by_family("data_ml")) >= 8


def test_taxonomy_ids_are_unique_and_dotted(taxonomy):
    ids = taxonomy.ids
    assert len(ids) == len(set(ids))
    assert all("." in i for i in ids)


def test_taxonomy_is_cached(taxonomy):
    assert load_taxonomy() is taxonomy


def test_stub_rubric_produces_both_prior_widths(taxonomy):
    """Gap-probing is a consequence of prior width, so a fixture rubric that
    contains only one kind of prior cannot exercise it."""
    rubric = taxonomy.stub_rubric(n=6)
    variances = {c.prior_var for c in rubric.competencies}
    assert len(variances) == 2
    assert HIGH_UNCERTAINTY_VAR in variances


def test_resume_evidence_is_span_backed(taxonomy):
    rubric = taxonomy.stub_rubric(n=8)
    for comp in rubric.competencies:
        if comp.evidence_in_resume > 0:
            assert comp.resume_spans


def test_bank_items_are_well_formed(bank):
    for q in bank.questions:
        assert len(q.anchors) == 5
        assert [a.level for a in sorted(q.anchors, key=lambda a: a.level)] == [1, 2, 3, 4, 5]
        assert q.expected_seconds > 0
        assert q.grm.a > 0
        assert q.grm.calibrated is False, "authoring defaults must be marked uncalibrated"


def test_anchor_concept_requirements_are_cumulative(bank):
    for q in bank.questions:
        for level in range(1, 6):
            required = q.anchor(level).required_concepts
            assert len(required) == LEVEL_THRESHOLDS[level]
            if level > 1:
                # Each level extends the one below it rather than replacing it.
                lower = q.anchor(level - 1).required_concepts
                assert required[: len(lower)] == lower
        assert len(q.anchor(5).required_concepts) == POOL_SIZE


def test_bank_generation_is_deterministic(taxonomy):
    a = stub_bank(taxonomy, per_competency=2)
    b = stub_bank(taxonomy, per_competency=2)
    assert a.model_dump() == b.model_dump()


def test_bank_round_trips_through_disk(taxonomy, tmp_path):
    bank = stub_bank(taxonomy, per_competency=1)
    path = save_bank(bank, tmp_path / "bank.json")
    assert load_bank(path).model_dump() == bank.model_dump()


def test_quarantined_items_are_excluded_from_the_live_pool(taxonomy):
    bank = stub_bank(taxonomy, per_competency=2)
    bank.questions[0].grm.quarantined = True
    bank.questions[0].grm.quarantine_reason = "a < 0.3"

    assert len(bank.live()) == len(bank.questions) - 1
    assert bank.questions[0] not in bank.for_competency(bank.questions[0].competency_id)
