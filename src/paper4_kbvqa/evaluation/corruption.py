from __future__ import annotations

from dataclasses import replace
import random
import re
from typing import Iterable, Sequence

from paper4_kbvqa.types import Evidence

_NEGATION_RE = re.compile(r"\b(is|are|was|were|has|have|can|does|do)\b", re.IGNORECASE)


def _clone_with_metadata(e: Evidence, **updates) -> Evidence:
    meta = dict(e.metadata)
    meta.update(updates)
    return replace(e, metadata=meta)


def inject_irrelevant(
    evidence: Sequence[Evidence],
    distractors: Sequence[Evidence],
    *,
    count: int,
    seed: int,
) -> list[Evidence]:
    """Append deterministic distractors without consulting answer annotations."""
    if count < 0:
        raise ValueError("count must be >= 0")
    rng = random.Random(seed)
    pool = list(distractors)
    rng.shuffle(pool)
    chosen = pool[: min(count, len(pool))]
    out = [_clone_with_metadata(e, corruption="clean") for e in evidence]
    out.extend(_clone_with_metadata(e, corruption="irrelevant_injection") for e in chosen)
    return out


def source_dropout(
    evidence: Sequence[Evidence],
    *,
    source: str,
) -> list[Evidence]:
    """Drop one named source; source matching is case-insensitive."""
    target = source.casefold()
    return [
        _clone_with_metadata(e, corruption="source_dropout", dropped_source=source)
        for e in evidence
        if e.source.casefold() != target
    ]


def ranking_corruption(
    evidence: Sequence[Evidence],
    *,
    severity: float,
    seed: int,
) -> list[Evidence]:
    """Promote lower-ranked items using a reproducible sequence perturbation.

    severity=0 preserves order; severity=1 fully shuffles the order. Intermediate
    values perform a deterministic number of random swaps. Retrieval scores are
    intentionally preserved so the perturbation can be audited separately from
    the underlying retriever output.
    """
    if not 0.0 <= severity <= 1.0:
        raise ValueError("severity must be in [0, 1]")
    out = list(evidence)
    if len(out) < 2 or severity == 0:
        return [_clone_with_metadata(e, corruption="ranking", severity=severity) for e in out]
    rng = random.Random(seed)
    swaps = max(1, round(severity * len(out)))
    for _ in range(swaps):
        i, j = rng.sample(range(len(out)), 2)
        out[i], out[j] = out[j], out[i]
    return [_clone_with_metadata(e, corruption="ranking", severity=severity) for e in out]


def evidence_scarcity(evidence: Sequence[Evidence], *, k: int) -> list[Evidence]:
    if k < 0:
        raise ValueError("k must be >= 0")
    return [
        _clone_with_metadata(e, corruption="evidence_scarcity", retained_k=k)
        for e in list(evidence)[:k]
    ]


def synthetic_contradiction(evidence: Evidence) -> Evidence:
    """Create a deterministic linguistic contradiction for stress testing.

    This transformation never reads a question's ground-truth answer. It is a
    controlled robustness perturbation, not a claim of naturally occurring
    contradiction. The generated item is explicitly tagged as synthetic.
    """
    text = evidence.text.strip()
    match = _NEGATION_RE.search(text)
    if match:
        verb = match.group(0)
        text = text[: match.end()] + " not" + text[match.end() :]
    else:
        text = "It is false that " + text
    return Evidence(
        evidence_id=f"{evidence.evidence_id}::synthetic-contradiction",
        text=text,
        source=evidence.source,
        uri=evidence.uri,
        entity_ids=evidence.entity_ids,
        retrieval_score=evidence.retrieval_score,
        metadata={
            **dict(evidence.metadata),
            "corruption": "synthetic_contradiction",
            "synthetic": True,
            "parent_evidence_id": evidence.evidence_id,
        },
    )


def inject_contradictions(
    evidence: Sequence[Evidence],
    *,
    count: int,
    seed: int,
) -> list[Evidence]:
    if count < 0:
        raise ValueError("count must be >= 0")
    base = list(evidence)
    rng = random.Random(seed)
    candidates = list(base)
    rng.shuffle(candidates)
    corrupted = [synthetic_contradiction(e) for e in candidates[: min(count, len(candidates))]]
    return [_clone_with_metadata(e, corruption="clean") for e in base] + corrupted


def corruption_manifest(rows: Iterable[Evidence]) -> list[dict]:
    """Return an auditable, JSON-serializable record of corrupted evidence."""
    return [
        {
            "evidence_id": e.evidence_id,
            "source": e.source,
            "retrieval_score": e.retrieval_score,
            "corruption": e.metadata.get("corruption", "clean"),
            "synthetic": bool(e.metadata.get("synthetic", False)),
            "parent_evidence_id": e.metadata.get("parent_evidence_id"),
        }
        for e in rows
    ]
