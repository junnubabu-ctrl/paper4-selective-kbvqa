"""Bounded deterministic context construction; fixtures are not real evidence."""
import pytest
from paper4_kbvqa.types import Evidence
from paper4_kbvqa.knowledge.structured import StructuredKnowledgeProvider
from paper4_kbvqa.filtering.relevance import _cos, RelevanceFilter


class Source:
    name = 'conceptnet'
    def __init__(self, prefix): self.prefix = prefix
    def retrieve(self, query, limit):
        return [Evidence(self.prefix + query + str(i), 'fixture fact', self.name,
                         retrieval_score=1) for i in range(limit)]


def test_cap_is_global_across_sources_and_entity_queries():
    result = StructuredKnowledgeProvider([Source('b'), Source('a')]).retrieve_for_question(
        'Question fixture', ['cat', 'dog', 'bowl', 'mat'], 10)
    assert len(result) == 10
    assert [e.evidence_id for e in result] == sorted(e.evidence_id for e in result)


def test_source_order_cannot_change_equal_score_cutoff():
    a, b = Source('a'), Source('b')
    forward = StructuredKnowledgeProvider([a,b]).retrieve_for_question('?', ['cat'], 3)
    backward = StructuredKnowledgeProvider([b,a]).retrieve_for_question('?', ['cat'], 3)
    assert forward == backward


@pytest.mark.parametrize('limit', [0, -1, True, 1.5])
def test_invalid_global_limit_rejected(limit):
    with pytest.raises(ValueError):
        StructuredKnowledgeProvider([Source('a')]).retrieve_for_question('?', ['cat'], limit)


def test_stopword_only_context_does_not_crash_filtering():
    assert _cos('the is', 'and of') == 0
    selected, _ = RelevanceFilter().select([Evidence('a','and of','fixture')],
                                          'the is', ['the'], 5)
    assert selected == []
