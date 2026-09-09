from __future__ import annotations
from abc import ABC, abstractmethod
from paper4_kbvqa.types import Evidence
from paper4_kbvqa.data.leakage import assert_prompt_no_ground_truth

class AnswerGenerator(ABC):
    @abstractmethod
    def generate(self, image_path: str, question: str, evidence: list[Evidence]) -> dict: ...

def build_prompt_payload(question: str,evidence: list[Evidence]) -> dict:
    payload={"question":question,"evidence":[{"id":e.evidence_id,"text":e.text,"source":e.source} for e in evidence]}
    assert_prompt_no_ground_truth(payload)
    return payload

class DeterministicDevGenerator(AnswerGenerator):
    """TEST-ONLY generator. Never use its outputs as benchmark results."""
    def generate(self,image_path,question,evidence):
        build_prompt_payload(question,evidence)
        answer=(evidence[0].text.split('.')[0] if evidence else "unknown")[:80]
        return {"answer":answer,"raw_confidence":0.5,"supporting_evidence_ids":[e.evidence_id for e in evidence[:2]]}
