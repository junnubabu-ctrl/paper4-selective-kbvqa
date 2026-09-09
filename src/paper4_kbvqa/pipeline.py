from __future__ import annotations
import math
from dataclasses import asdict
from paper4_kbvqa.filtering.relevance import RelevanceFilter
from paper4_kbvqa.generation.base import AnswerGenerator
from paper4_kbvqa.knowledge.base import KnowledgeProvider
from paper4_kbvqa.types import Evidence
from paper4_kbvqa.verification.verifier import EvidenceVerifier


def build_retrieval_query(question: str, visual_entities: list[str] | tuple[str, ...]) -> str:
    """Leakage-safe query: it can use question text and visual entities, never annotated answers."""
    entities = " ".join(str(x).strip() for x in visual_entities if str(x).strip())
    return f"{question.strip()} {entities}".strip()


def fuse_uncalibrated_confidence(generation_confidence: float, verification_score: float) -> float:
    """Transparent pre-calibration score; not a probability until a held-out calibrator is fitted."""
    g = float(generation_confidence)
    v = max(0.0, min(1.0, float(verification_score)))
    if not math.isfinite(g):
        return v
    g = max(0.0, min(1.0, g))
    return 0.5 * g + 0.5 * v


class KBVQAPipeline:
    def __init__(
        self,
        generator: AnswerGenerator,
        provider: KnowledgeProvider | None = None,
        relevance_filter: RelevanceFilter | None = None,
        verifier: EvidenceVerifier | None = None,
    ):
        self.generator = generator
        self.provider = provider
        self.filter = relevance_filter or RelevanceFilter()
        self.verifier = verifier or EvidenceVerifier()

    def run_one(
        self,
        *,
        question_id: str,
        image_path: str,
        question: str,
        visual_entities: list[str] | tuple[str, ...] = (),
        mode: str = "full",
        retrieval_limit: int = 10,
        top_k: int = 5,
        visual_consistency: float = 0.5,
        auto_extract_entities: bool = True,
    ) -> dict:
        if mode not in {"no_knowledge", "raw_knowledge", "full"}:
            raise ValueError("mode must be one of: no_knowledge, raw_knowledge, full")
        visual_entities=list(visual_entities)
        if auto_extract_entities and not visual_entities and hasattr(self.generator, "extract_visual_entities"):
            visual_entities=list(self.generator.extract_visual_entities(image_path, question))
        query = build_retrieval_query(question, visual_entities)
        evidence: list[Evidence] = []
        scoring = {}
        if mode != "no_knowledge":
            if self.provider is None:
                raise RuntimeError("Knowledge provider required for knowledge-enabled modes")
            evidence = self.provider.retrieve(query, limit=retrieval_limit)
            if mode == "full":
                evidence, scoring = self.filter.select(evidence, question, visual_entities, top_k=top_k)
            else:
                evidence = evidence[:top_k]
        generation = self.generator.generate(image_path, question, evidence)
        answer = str(generation.get("answer", "")).strip()
        verification = self.verifier.verify(answer, evidence, visual_consistency=visual_consistency)
        raw_conf = fuse_uncalibrated_confidence(
            float(generation.get("raw_confidence", float("nan"))),
            float(verification["verification_score"]),
        )
        return {
            "question_id": str(question_id),
            "question": question,
            "retrieval_query": query,
            "visual_entities": visual_entities,
            "mode": mode,
            "answer": answer,
            "supporting_evidence_ids": list(generation.get("supporting_evidence_ids", [])),
            "evidence": [asdict(e) for e in evidence],
            "filter_scores": scoring,
            "generation_confidence": generation.get("raw_confidence"),
            "verification": verification,
            "raw_confidence": raw_conf,
        }
