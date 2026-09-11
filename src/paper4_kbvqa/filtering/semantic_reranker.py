from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from paper4_kbvqa.types import Evidence


class SentenceTransformerReranker:
    """Optional stronger semantic evidence reranker for reviewer-facing ablation.

    The baseline is lazy-loaded and leaves the primary transparent relevance
    filter unchanged. It is intended to test whether the EviTrust reliability
    layer remains useful when evidence ranking is stronger.
    """

    def __init__(
        self,
        model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        semantic_weight: float = 0.85,
        retrieval_weight: float = 0.15,
        revision: str = "main",
    ):
        self.model_id = model_id
        self.revision = revision
        self.semantic_weight = float(semantic_weight)
        self.retrieval_weight = float(retrieval_weight)
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_id, revision=self.revision)

    def select(
        self,
        evidence: Sequence[Evidence],
        question: str,
        visual_entities: Sequence[str],
        top_k: int = 5,
    ):
        self._load()
        if not evidence:
            return [], {}

        query = " ".join(
            [question] + [str(x) for x in visual_entities if str(x).strip()]
        ).strip()
        texts = [e.text for e in evidence]
        qvec = self._model.encode([query], normalize_embeddings=True)[0]
        evec = self._model.encode(texts, normalize_embeddings=True)
        semantic = evec @ qvec

        scored = []
        details = {}
        for ev, sem in zip(evidence, semantic):
            retrieval = max(0.0, min(1.0, float(ev.retrieval_score)))
            score = self.semantic_weight * float(sem) + self.retrieval_weight * retrieval
            details[str(ev.evidence_id)] = {
                "score": float(score),
                "semantic": float(sem),
                "retrieval": retrieval,
                "model_id": self.model_id,
            }
            scored.append((score, ev))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [ev for _, ev in scored[:top_k]], details
