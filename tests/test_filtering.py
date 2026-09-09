from paper4_kbvqa.types import Evidence
from paper4_kbvqa.filtering.relevance import RelevanceFilter

def test_filter_prefers_relevant_evidence():
    ev=[Evidence("a","Eiffel Tower Paris France","x",retrieval_score=.9),Evidence("b","banana fruit yellow","x",retrieval_score=.9)]
    sel,_=RelevanceFilter().select(ev,"Where is Eiffel Tower?",["Eiffel Tower"],1)
    assert sel[0].evidence_id=="a"


def test_filter_stops_before_low_relevance_padding():
    ev=[Evidence("a","Eiffel Tower Paris France","x",retrieval_score=.9),Evidence("b","Colosseum Rome Italy","x",retrieval_score=.1)]
    sel,_=RelevanceFilter(min_score=.15).select(ev,"Where is Eiffel Tower?",["Eiffel Tower"],5)
    assert [e.evidence_id for e in sel]==["a"]
