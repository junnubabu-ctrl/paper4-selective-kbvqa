from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import time
from typing import Protocol, Sequence

from paper4_kbvqa.data.manifest import VQASample
from paper4_kbvqa.types import Evidence


class Retriever(Protocol):
    def retrieve(self, query: str, top_k: int = 10) -> Sequence[Evidence]: ...


class EvidenceFilter(Protocol):
    def filter(self, question: str, evidence: Sequence[Evidence], **kwargs) -> Sequence[Evidence]: ...


class AnswerGenerator(Protocol):
    def generate(self, *, image_path: str, question: str, evidence: Sequence[Evidence]): ...


class EvidenceCritic(Protocol):
    def verify(self, *, question: str, answer: str, visual_entities: Sequence[str], evidence: Sequence[Evidence], cited_evidence_ids: Sequence[str]): ...


class Calibrator(Protocol):
    def predict(self, scores: Sequence[float]) -> Sequence[float]: ...


@dataclass(frozen=True)
class ExecutionConfig:
    retrieval_top_k: int = 10
    filtered_top_k: int = 5
    target_risk: float = 0.05
    generator_weight: float = 0.30
    critic_support_weight: float = 0.35
    provenance_weight: float = 0.20
    visual_weight: float = 0.15
    contradiction_penalty: float = 0.35
    prompt_version: str = "evitrust-runtime-v1"

    def validate(self) -> None:
        if self.retrieval_top_k < 1 or self.filtered_top_k < 1:
            raise ValueError("top-k values must be positive")
        if self.filtered_top_k > self.retrieval_top_k:
            raise ValueError("filtered_top_k cannot exceed retrieval_top_k")
        if not 0.0 < self.target_risk < 1.0:
            raise ValueError("target_risk must be in (0,1)")
        for name in ("generator_weight", "critic_support_weight", "provenance_weight", "visual_weight", "contradiction_penalty"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")


def leakage_safe_query(sample: VQASample) -> str:
    parts = [sample.question.strip()]
    if sample.visual_entities:
        parts.append(" ".join(sample.visual_entities))
    return " ".join(p for p in parts if p).strip()


def provenance_score(evidence: Sequence[Evidence], cited_ids: Sequence[str]) -> float:
    if not cited_ids:
        return 0.0
    available = {ev.evidence_id for ev in evidence}
    valid = sum(1 for x in cited_ids if x in available)
    return valid / max(1, len(cited_ids))


def reliability_fusion(*, generation_confidence: float, support_probability: float, contradiction_probability: float, provenance: float, visual_consistency: float, config: ExecutionConfig) -> float:
    raw = (
        config.generator_weight * generation_confidence
        + config.critic_support_weight * support_probability
        + config.provenance_weight * provenance
        + config.visual_weight * visual_consistency
        - config.contradiction_penalty * contradiction_probability
    )
    return max(0.0, min(1.0, raw))


def _atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def record_checksum(obj: dict) -> str:
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return sha256(payload).hexdigest()


class EviTrustExecutionController:
    def __init__(self, *, retriever: Retriever, evidence_filter: EvidenceFilter, generator: AnswerGenerator, critic: EvidenceCritic, calibrator: Calibrator | None = None, threshold: float | None = None, config: ExecutionConfig | None = None):
        self.retriever = retriever
        self.evidence_filter = evidence_filter
        self.generator = generator
        self.critic = critic
        self.calibrator = calibrator
        self.threshold = threshold
        self.config = config or ExecutionConfig()
        self.config.validate()

    def run_one(self, sample: VQASample) -> dict:
        t0 = time.perf_counter()
        query = leakage_safe_query(sample)
        retrieved = list(self.retriever.retrieve(query, top_k=self.config.retrieval_top_k))
        filtered = list(self.evidence_filter.filter(sample.question, retrieved, visual_entities=sample.visual_entities, top_k=self.config.filtered_top_k))[: self.config.filtered_top_k]
        generated = self.generator.generate(image_path=sample.image_path, question=sample.question, evidence=filtered)
        answer = str(getattr(generated, "answer", ""))
        cited = list(getattr(generated, "evidence_ids", []))
        generation_confidence = float(getattr(generated, "confidence", getattr(generated, "raw_confidence", 0.0)))
        verified = self.critic.verify(question=sample.question, answer=answer, visual_entities=sample.visual_entities, evidence=filtered, cited_evidence_ids=cited)
        support = float(getattr(verified, "supported", 0.0))
        contradiction = float(getattr(verified, "contradicted", 0.0))
        insufficiency = float(getattr(verified, "insufficient", 1.0))
        prov = provenance_score(filtered, cited)
        visual_consistency = float(getattr(generated, "visual_consistency", 0.0))
        raw_reliability = reliability_fusion(generation_confidence=generation_confidence, support_probability=support, contradiction_probability=contradiction, provenance=prov, visual_consistency=visual_consistency, config=self.config)
        calibrated = None
        if self.calibrator is not None:
            calibrated = float(list(self.calibrator.predict([raw_reliability]))[0])
        abstain = None
        selective_answer = answer
        if calibrated is not None and self.threshold is not None:
            abstain = calibrated < float(self.threshold)
            if abstain:
                selective_answer = ""
        rec = {
            "question_id": str(sample.question_id),
            "query": query,
            "answer": answer,
            "selective_answer": selective_answer,
            "cited_evidence_ids": cited,
            "retrieved_evidence_ids": [e.evidence_id for e in retrieved],
            "filtered_evidence_ids": [e.evidence_id for e in filtered],
            "generation_confidence": generation_confidence,
            "critic": {
                "supported": support,
                "contradicted": contradiction,
                "insufficient": insufficiency,
                "label": str(getattr(getattr(verified, "label", ""), "value", getattr(verified, "label", ""))),
                "model_id": str(getattr(verified, "model_id", "")),
                "prompt_version": str(getattr(verified, "prompt_version", "")),
            },
            "provenance_score": prov,
            "visual_consistency": visual_consistency,
            "raw_reliability": raw_reliability,
            "calibrated_confidence": calibrated,
            "threshold": self.threshold,
            "abstain": abstain,
            "latency_seconds": time.perf_counter() - t0,
            "runtime_prompt_version": self.config.prompt_version,
        }
        rec["record_sha256"] = record_checksum(rec)
        return rec

    def run_manifest(self, samples: Sequence[VQASample], *, output_jsonl: str | Path, checkpoint_json: str | Path) -> None:
        output_jsonl = Path(output_jsonl)
        checkpoint_json = Path(checkpoint_json)
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        completed: set[str] = set()
        if checkpoint_json.exists():
            state = json.loads(checkpoint_json.read_text(encoding="utf-8"))
            completed = set(str(x) for x in state.get("completed_question_ids", []))
        with output_jsonl.open("a", encoding="utf-8") as out:
            for sample in samples:
                qid = str(sample.question_id)
                if qid in completed:
                    continue
                rec = self.run_one(sample)
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                completed.add(qid)
                _atomic_json(checkpoint_json, {"completed_question_ids": sorted(completed), "count": len(completed)})
