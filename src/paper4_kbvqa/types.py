from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    text: str
    source: str
    uri: str | None = None
    entity_ids: tuple[str, ...] = ()
    retrieval_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class AnswerRecord:
    question_id: str
    question: str
    answer: str
    evidence_ids: list[str]
    raw_confidence: float
    verification_score: float
    calibrated_confidence: float | None = None
    abstain: bool | None = None
