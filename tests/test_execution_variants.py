from dataclasses import dataclass

from paper4_kbvqa.data.manifest import VQASample
from paper4_kbvqa.execution.variants import MatchedVariantRunner
from paper4_kbvqa.execution.controller import ExecutionConfig
from paper4_kbvqa.types import Evidence


class Generator:
    def generate(self, image_path, question, evidence):
        return {"answer":"cat","raw_confidence":0.8,"supporting_evidence_ids":["e1"] if evidence else []}


class Provider:
    def retrieve(self, query, limit=10):
        return [Evidence("e1","cat fact","wikipedia","https://example.org/e1",retrieval_score=.9)]


class Filter:
    def select(self,evidence,question,visual_entities,top_k=5):
        return list(evidence)[:top_k], {"e1":{"score":.9}}


@dataclass
class CriticResult:
    supported: float=.9
    contradicted: float=.05
    insufficient: float=.05
    label: str="SUPPORTED"
    model_id: str="critic"
    prompt_version: str="v1"


class Critic:
    def verify(self,**kwargs):
        return CriticResult()


def sample():
    return VQASample(
        question_id="1", image_path="x.jpg", question="What animal?",
        answers=("cat",), visual_entities=("cat",)
    )


def test_b0_has_no_external_evidence_and_uses_generation_confidence():
    r=MatchedVariantRunner(variant="B0",generator=Generator())
    rec=r.run_one(sample())
    assert rec["retrieved_evidence_ids"]==[]
    assert rec["filtered_evidence_ids"]==[]
    assert rec["critic"] is None
    assert rec["raw_confidence"]==0.8


def test_b1_has_raw_evidence_without_filter_or_critic():
    r=MatchedVariantRunner(variant="B1",generator=Generator(),provider=Provider())
    rec=r.run_one(sample())
    assert rec["filtered_evidence_ids"]==["e1"]
    assert rec["filter_scores"]=={}
    assert rec["critic"] is None
    assert rec["raw_confidence"]==0.8


def test_b2_adds_filter_but_not_critic():
    r=MatchedVariantRunner(
        variant="B2",generator=Generator(),provider=Provider(),
        evidence_filter=Filter(),config=ExecutionConfig(retrieval_top_k=1,filtered_top_k=1)
    )
    rec=r.run_one(sample())
    assert rec["filter_scores"]["e1"]["score"]==.9
    assert rec["critic"] is None
    assert rec["raw_confidence"]==0.8


def test_b3_adds_critic_and_provenance_reliability():
    r=MatchedVariantRunner(
        variant="B3",generator=Generator(),provider=Provider(),
        evidence_filter=Filter(),critic=Critic(),
        config=ExecutionConfig(retrieval_top_k=1,filtered_top_k=1)
    )
    rec=r.run_one(sample())
    assert rec["critic"]["supported"]==.9
    assert rec["provenance_score"]>0
    assert 0 <= rec["raw_confidence"] <= 1
