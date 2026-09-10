from dataclasses import dataclass

from paper4_kbvqa.data.manifest import VQASample
from paper4_kbvqa.execution.controller import EviTrustExecutionController, ExecutionConfig
from paper4_kbvqa.types import Evidence


class Provider:
    def retrieve(self, query, limit=10):
        assert "gold-secret" not in query
        return [
            Evidence("e1", "A cat is an animal.", "wikipedia", retrieval_score=0.9),
            Evidence("e2", "A dog is an animal.", "wikidata", retrieval_score=0.7),
        ][:limit]


class Filter:
    def select(self, evidence, question, visual_entities, top_k=5):
        return list(evidence)[:top_k], {"e1": {"score": 0.9}}


class Generator:
    def generate(self, image_path, question, evidence):
        assert "gold-secret" not in question
        return {
            "answer": "cat",
            "raw_confidence": 0.8,
            "supporting_evidence_ids": ["e1"],
        }


@dataclass(frozen=True)
class CriticResult:
    supported: float = 0.9
    contradicted: float = 0.05
    insufficient: float = 0.05
    label: str = "SUPPORTED"
    model_id: str = "fake"
    prompt_version: str = "fake-v1"


class Critic:
    def verify(self, **kwargs):
        assert "gold-secret" not in str(kwargs)
        return CriticResult()


def test_controller_matches_repository_provider_filter_and_generator_interfaces():
    sample = VQASample(
        question_id="q1",
        image_path="image.jpg",
        question="What animal is shown?",
        answers=("gold-secret",),
        visual_entities=("cat",),
    )
    controller = EviTrustExecutionController(
        provider=Provider(),
        evidence_filter=Filter(),
        generator=Generator(),
        critic=Critic(),
        config=ExecutionConfig(retrieval_top_k=2, filtered_top_k=1),
    )
    rec = controller.run_one(sample)
    assert rec["answer"] == "cat"
    assert rec["supporting_evidence_ids"] == ["e1"]
    assert rec["filtered_evidence_ids"] == ["e1"]
    assert rec["raw_confidence"] == rec["raw_reliability"]
    assert "gold-secret" not in str(rec)
