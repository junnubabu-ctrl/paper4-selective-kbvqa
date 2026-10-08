"""Tests of distinct-ID accounting, not empirical evidence quality."""
from paper4_kbvqa.execution.controller import provenance_components, provenance_score
from paper4_kbvqa.types import Evidence


def test_repetition_cannot_hide_unknown_citation():
    evidence = [Evidence('good', 'passage', 'source', 'https://example.org/a')]
    raw_ids = ['good'] * 100 + ['unknown']
    assert provenance_score(evidence, raw_ids) == .5
    p = provenance_components(evidence, raw_ids)
    assert p['citation_validity'] == .5
    assert p['source_traceability'] == 1
    assert p['composite'] == .7


def test_repeated_traceable_id_cannot_hide_untraceable_valid_id():
    evidence = [Evidence('a', 'first', 's', 'https://example.org/a'),
                Evidence('b', 'second', 's', None)]
    p = provenance_components(evidence, ['a'] * 100 + ['b'])
    assert p['source_traceability'] == .5
    assert p['citation_validity'] == 1


def test_legacy_field_is_explicit_relative_coverage_alias():
    evidence = [Evidence('a', 'first', 's1', None), Evidence('b', 'second', 's2', None)]
    p = provenance_components(evidence, ['a'])
    assert p['relative_cited_source_coverage'] == p['source_diversity'] == .5
    assert provenance_components(evidence, [])['composite'] == 0
