from __future__ import annotations
from paper4_kbvqa.knowledge.base import KnowledgeProvider
from paper4_kbvqa.types import Evidence
from paper4_kbvqa.retrieval.tfidf import TfidfRetriever

class LocalKnowledgeProvider(KnowledgeProvider):
    name = "local"
    def __init__(self, evidence: list[Evidence]):
        self.evidence = list(evidence)
        self.retriever = TfidfRetriever([e.text for e in self.evidence])
    def retrieve(self, query: str, limit: int = 10) -> list[Evidence]:
        scored=[]
        for idx, score in self.retriever.search(query, top_k=limit):
            e=self.evidence[idx]
            scored.append(Evidence(e.evidence_id,e.text,e.source,e.uri,e.entity_ids,float(score),e.metadata))
        return scored
