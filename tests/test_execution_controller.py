from dataclasses import dataclass

from paper4_kbvqa.data.manifest import VQASample
from paper4_kbvqa.execution.controller import (
    EviTrustExecutionController,
    ExecutionConfig,
    leakage_safe_query,
    provenance_score,
)
from paper4_kbvqa.types import Evidence


class FakeRetriever:
    def retrieve(self, query, top_k=10):
        assert "forbidden_gold_answer" not in query
        return [
            Evidence("e1", "supporting evidence", "wikipedia"),
            Evidence("e2", "other evidence", "wikidata"),
        ][:top_k]


class FakeFilter:
    def filter(self, question, evidence, **kwargs):
        return list(evidence)


@dataclass
class FakeGeneration:
    answer: str = "answer"
    evidence_ids: tuple[str, ...] = ("e1",)
    confidence: float = 0.8
    visual_consistency: float = 0.7


class FakeGenerator:
    def generate(self, **kwargs):
        assert "forbidden_gold_answer" not in str(kwargs)
        return FakeGeneration()


@dataclass
class FakeCriticResult:
    supported: float = 0.9
    contradicted: float = 0.05
    insufficient: float = 0.05
    label: str = "SUPPORTED"
    model_id: str = "fake-critic"
    prompt_version: str = "test"


class FakeCritic:
    def verify(self, **kwargs):
        assert "forbidden_gold_answer" not in str(kwargs)
        return FakeCriticResult()


class IdentityCalibrator:
    def predict(self, scores):
        return list(scores)


def test_leakage_safe_query_excludes_answers():
    sample = VQASample(
        question_id="1",
        image_path="x.jpg",
        question="What is shown?",
        answers=("forbidden_gold_answer",),
        visual_entities=("tree",),
    )
    query = leakage_safe_query(sample)
    assert "forbidden_gold_answer" not in query
    assert "tree" in query


def test_controller_never_routes_ground_truth():
    sample = VQASample(
        question_id="1",
        image_path="x.jpg",
        question="What is shown?",
        answers=("forbidden_gold_answer",),
        visual_entities=("tree",),
    )
    controller = EviTrustExecutionController(
        retriever=FakeRetriever(),
        evidence_filter=FakeFilter(),
        generator=FakeGenerator(),
        critic=FakeCritic(),
        calibrator=IdentityCalibrator(),
        threshold=0.5,
        config=ExecutionConfig(retrieval_top_k=2, filtered_top_k=2),
    )
    rec = controller.run_one(sample)
    assert rec["answer"] == "answer"
    assert rec["abstain"] is False
    assert "forbidden_gold_answer" not in str(rec)


def test_provenance_is_fraction_of_valid_citations():
    evidence = [Evidence("e1", "a", "x"), Evidence("e2", "b", "y")]
    assert provenance_score(evidence, ["e1", "missing"]) == 0.5
