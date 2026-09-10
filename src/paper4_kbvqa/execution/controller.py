from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import time
from typing import Protocol, Sequence

from paper4_kbvqa.data.manifest import VQASample
from paper4_kbvqa.types import Evidence


class EvidenceCritic(Protocol):
    def verify(
        self,
        *,
        question: str,
        answer: str,
        visual_entities: Sequence[str],
        evidence: Sequence[Evidence],
        cited_evidence_ids: Sequence[str],
    ): ...


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
    prompt_version: str = "evitrust-runtime-v2"
    auto_extract_entities: bool = True

    def validate(self) -> None:
        if self.retrieval_top_k < 1 or self.filtered_top_k < 1:
            raise ValueError("top-k values must be positive")
        if self.filtered_top_k > self.retrieval_top_k:
            raise ValueError("filtered_top_k cannot exceed retrieval_top_k")
        if not 0.0 < self.target_risk < 1.0:
            raise ValueError("target_risk must be in (0,1)")
        for name in (
            "generator_weight",
            "critic_support_weight",
            "provenance_weight",
            "visual_weight",
            "contradiction_penalty",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")


def leakage_safe_query(question: str, visual_entities: Sequence[str]) -> str:
    parts = [str(question).strip()]
    if visual_entities:
        parts.append(" ".join(str(x).strip() for x in visual_entities if str(x).strip()))
    return " ".join(p for p in parts if p).strip()


def provenance_score(evidence: Sequence[Evidence], cited_ids: Sequence[str]) -> float:
    if not cited_ids:
        return 0.0
    available = {str(ev.evidence_id) for ev in evidence}
    valid = sum(1 for x in cited_ids if str(x) in available)
    return valid / max(1, len(cited_ids))


def reliability_fusion(
    *,
    generation_confidence: float,
    support_probability: float,
    contradiction_probability: float,
    provenance: float,
    visual_consistency: float,
    config: ExecutionConfig,
) -> float:
    values = [
        generation_confidence,
        support_probability,
        contradiction_probability,
        provenance,
        visual_consistency,
    ]
    clean = [0.0 if not math.isfinite(float(x)) else max(0.0, min(1.0, float(x))) for x in values]
    generation_confidence, support_probability, contradiction_probability, provenance, visual_consistency = clean
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


def _field(obj, *names, default=None):
    if isinstance(obj, dict):
        for name in names:
            if name in obj:
                return obj[name]
        return default
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


class EviTrustExecutionController:
    """Production-compatible LLM execution controller.

    Reference answers remain in VQASample only for downstream evaluation and are
    never passed to visual grounding, retrieval, generation, evidence filtering,
    the LLM critic, calibration, or selective decision code.
    """

    def __init__(
        self,
        *,
        provider,
        evidence_filter,
        generator,
        critic: EvidenceCritic,
        calibrator: Calibrator | None = None,
        threshold: float | None = None,
        config: ExecutionConfig | None = None,
    ):
        self.provider = provider
        self.evidence_filter = evidence_filter
        self.generator = generator
        self.critic = critic
        self.calibrator = calibrator
        self.threshold = threshold
        self.config = config or ExecutionConfig()
        self.config.validate()

    def _retrieve(self, query: str) -> list[Evidence]:
        try:
            return list(self.provider.retrieve(query, limit=self.config.retrieval_top_k))
        except TypeError:
            return list(self.provider.retrieve(query, top_k=self.config.retrieval_top_k))

    def _filter(
        self,
        evidence: Sequence[Evidence],
        question: str,
        visual_entities: Sequence[str],
    ) -> tuple[list[Evidence], dict]:
        if hasattr(self.evidence_filter, "select"):
            result = self.evidence_filter.select(
                list(evidence),
                question,
                list(visual_entities),
                top_k=self.config.filtered_top_k,
            )
            if isinstance(result, tuple) and len(result) == 2:
                selected, details = result
                return list(selected), dict(details)
            return list(result), {}
        result = self.evidence_filter.filter(
            question,
            list(evidence),
            visual_entities=list(visual_entities),
            top_k=self.config.filtered_top_k,
        )
        return list(result)[: self.config.filtered_top_k], {}

    def run_one(self, sample: VQASample) -> dict:
        t0 = time.perf_counter()

        visual_entities = list(sample.visual_entities)
        if (
            self.config.auto_extract_entities
            and not visual_entities
            and hasattr(self.generator, "extract_visual_entities")
        ):
            visual_entities = list(
                self.generator.extract_visual_entities(sample.image_path, sample.question)
            )

        query = leakage_safe_query(sample.question, visual_entities)
        retrieved = self._retrieve(query)
        filtered, filter_scores = self._filter(
            retrieved,
            sample.question,
            visual_entities,
        )

        generated = self.generator.generate(
            image_path=sample.image_path,
            question=sample.question,
            evidence=filtered,
        )
        answer = str(_field(generated, "answer", default="")).strip()
        cited = list(
            _field(
                generated,
                "supporting_evidence_ids",
                "evidence_ids",
                default=[],
            )
            or []
        )
        generation_confidence = float(
            _field(generated, "raw_confidence", "confidence", default=0.0)
        )
        visual_consistency = float(
            _field(generated, "visual_consistency", default=0.0)
        )

        verified = self.critic.verify(
            question=sample.question,
            answer=answer,
            visual_entities=visual_entities,
            evidence=filtered,
            cited_evidence_ids=cited,
        )
        support = float(_field(verified, "supported", default=0.0))
        contradiction = float(_field(verified, "contradicted", default=0.0))
        insufficiency = float(_field(verified, "insufficient", default=1.0))
        prov = provenance_score(filtered, cited)

        raw_reliability = reliability_fusion(
            generation_confidence=generation_confidence,
            support_probability=support,
            contradiction_probability=contradiction,
            provenance=prov,
            visual_consistency=visual_consistency,
            config=self.config,
        )

        calibrated = None
        if self.calibrator is not None:
            calibrated = float(list(self.calibrator.predict([raw_reliability]))[0])

        abstain = None
        selective_answer = answer
        if calibrated is not None and self.threshold is not None:
            abstain = calibrated < float(self.threshold)
            if abstain:
                selective_answer = "<ABSTAIN>"

        rec = {
            "question_id": str(sample.question_id),
            "question": sample.question,
            "retrieval_query": query,
            "visual_entities": visual_entities,
            "answer": answer,
            "selective_answer": selective_answer,
            "supporting_evidence_ids": cited,
            "evidence": [asdict(e) for e in filtered],
            "retrieved_evidence_ids": [str(e.evidence_id) for e in retrieved],
            "filtered_evidence_ids": [str(e.evidence_id) for e in filtered],
            "filter_scores": filter_scores,
            "generation_confidence": generation_confidence,
            "critic": {
                "supported": support,
                "contradicted": contradiction,
                "insufficient": insufficiency,
                "label": str(_field(_field(verified, "label", default=""), "value", default=_field(verified, "label", default=""))),
                "model_id": str(_field(verified, "model_id", default="")),
                "prompt_version": str(_field(verified, "prompt_version", default="")),
            },
            "provenance_score": prov,
            "visual_consistency": visual_consistency,
            "raw_reliability": raw_reliability,
            # Compatibility with existing calibration/evaluation scripts.
            "raw_confidence": raw_reliability,
            "calibrated_confidence": calibrated,
            "threshold": self.threshold,
            "abstain": abstain,
            "latency_s": time.perf_counter() - t0,
            "runtime_prompt_version": self.config.prompt_version,
        }
        rec["record_sha256"] = record_checksum(rec)
        return rec

    def run_manifest(
        self,
        samples: Sequence[VQASample],
        *,
        output_jsonl: str | Path,
        checkpoint_json: str | Path,
    ) -> dict:
        output_jsonl = Path(output_jsonl)
        checkpoint_json = Path(checkpoint_json)
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)

        completed: set[str] = set()
        if checkpoint_json.exists():
            state = json.loads(checkpoint_json.read_text(encoding="utf-8"))
            completed = set(str(x) for x in state.get("completed_question_ids", []))

        errors: list[dict] = []
        with output_jsonl.open("a", encoding="utf-8") as out:
            for sample in samples:
                qid = str(sample.question_id)
                if qid in completed:
                    continue
                try:
                    rec = self.run_one(sample)
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out.flush()
                    completed.add(qid)
                except Exception as exc:
                    errors.append({"question_id": qid, "error": repr(exc)})
                _atomic_json(
                    checkpoint_json,
                    {
                        "completed_question_ids": sorted(completed),
                        "count": len(completed),
                        "errors": errors,
                    },
                )
        return {
            "completed": len(completed),
            "errors": len(errors),
            "completed_question_ids": sorted(completed),
        }
