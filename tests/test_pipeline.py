from paper4_kbvqa.pipeline import KBVQAPipeline, build_retrieval_query
from paper4_kbvqa.generation.base import AnswerGenerator
from paper4_kbvqa.knowledge.base import KnowledgeProvider
from paper4_kbvqa.types import Evidence

class FakeProvider(KnowledgeProvider):
    name="fake"
    def __init__(self): self.last_query=None
    def retrieve(self, query, limit=10):
        self.last_query=query
        return [Evidence("e1","The Eiffel Tower is in Paris.","fake",retrieval_score=.9)]

class FakeGenerator(AnswerGenerator):
    def generate(self,image_path,question,evidence):
        return {"answer":"Paris","raw_confidence":.8,"supporting_evidence_ids":[e.evidence_id for e in evidence]}

def test_query_contains_no_answer_field():
    q=build_retrieval_query("Where is this tower?", ["Eiffel Tower"])
    assert q == "Where is this tower? Eiffel Tower"

def test_pipeline_full_cpu_fake():
    provider=FakeProvider()
    p=KBVQAPipeline(FakeGenerator(), provider=provider)
    r=p.run_one(question_id="q1", image_path="unused.jpg", question="Where is the Eiffel Tower?", visual_entities=["Eiffel Tower"], top_k=1)
    assert r["answer"] == "Paris"
    assert r["evidence"][0]["evidence_id"] == "e1"
    assert 0 <= r["raw_confidence"] <= 1
    assert "Paris" not in provider.last_query
