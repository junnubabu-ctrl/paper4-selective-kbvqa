from paper4_kbvqa.evaluation.corruption import (
    corruption_manifest,
    evidence_scarcity,
    inject_contradictions,
    inject_irrelevant,
    ranking_corruption,
    source_dropout,
    synthetic_contradiction,
)
from paper4_kbvqa.types import Evidence


def _e(i: int, source: str = "Wikipedia") -> Evidence:
    return Evidence(
        evidence_id=f"e{i}",
        text=f"Paris is a city number {i}.",
        source=source,
        retrieval_score=1.0 - i * 0.1,
    )


def test_irrelevant_injection_is_seeded_and_answer_free():
    base = [_e(0)]
    distractors = [_e(1, "ConceptNet"), _e(2, "Wikidata")]
    a = inject_irrelevant(base, distractors, count=1, seed=42)
    b = inject_irrelevant(base, distractors, count=1, seed=42)
    assert [x.evidence_id for x in a] == [x.evidence_id for x in b]
    assert a[-1].metadata["corruption"] == "irrelevant_injection"


def test_source_dropout_removes_only_target_source():
    rows = [_e(0, "Wikipedia"), _e(1, "Wikidata"), _e(2, "ConceptNet")]
    out = source_dropout(rows, source="wikidata")
    assert {x.source for x in out} == {"Wikipedia", "ConceptNet"}


def test_ranking_corruption_is_reproducible():
    rows = [_e(i) for i in range(6)]
    a = ranking_corruption(rows, severity=0.75, seed=123)
    b = ranking_corruption(rows, severity=0.75, seed=123)
    assert [x.evidence_id for x in a] == [x.evidence_id for x in b]
    assert [x.evidence_id for x in a] != [x.evidence_id for x in rows]


def test_evidence_scarcity_respects_k():
    rows = [_e(i) for i in range(5)]
    assert len(evidence_scarcity(rows, k=0)) == 0
    assert [x.evidence_id for x in evidence_scarcity(rows, k=2)] == ["e0", "e1"]


def test_synthetic_contradiction_is_tagged_and_traceable():
    original = _e(0)
    changed = synthetic_contradiction(original)
    assert changed.evidence_id != original.evidence_id
    assert changed.metadata["synthetic"] is True
    assert changed.metadata["parent_evidence_id"] == original.evidence_id
    assert changed.text != original.text


def test_contradiction_injection_and_manifest():
    rows = [_e(i) for i in range(3)]
    out = inject_contradictions(rows, count=2, seed=2026)
    manifest = corruption_manifest(out)
    assert len(out) == 5
    synthetic = [x for x in manifest if x["synthetic"]]
    assert len(synthetic) == 2
    assert all(x["parent_evidence_id"] for x in synthetic)
