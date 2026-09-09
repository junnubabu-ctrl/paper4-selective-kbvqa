from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import hashlib

FORBIDDEN_TEST_FIELDS = {"answer", "answers", "ground_truth", "gt_answer", "annotation_answers"}

@dataclass(frozen=True)
class LeakageReport:
    duplicate_ids: set[str]
    duplicate_content_hashes: set[str]
    forbidden_prompt_fields: set[str]

    @property
    def clean(self) -> bool:
        return not (self.duplicate_ids or self.duplicate_content_hashes or self.forbidden_prompt_fields)

def _content_hash(row: dict) -> str:
    payload = f"{row.get('image_id','')}|{row.get('question','')}".strip().lower()
    return hashlib.sha256(payload.encode()).hexdigest()

def check_split_overlap(train: Iterable[dict], val: Iterable[dict], test: Iterable[dict]) -> LeakageReport:
    splits=[list(train), list(val), list(test)]
    id_sets=[{str(r.get('question_id')) for r in s if r.get('question_id') is not None} for s in splits]
    hash_sets=[{_content_hash(r) for r in s} for s in splits]
    dup_ids=(id_sets[0]&id_sets[1]) | (id_sets[0]&id_sets[2]) | (id_sets[1]&id_sets[2])
    dup_hash=(hash_sets[0]&hash_sets[1]) | (hash_sets[0]&hash_sets[2]) | (hash_sets[1]&hash_sets[2])
    return LeakageReport(dup_ids, dup_hash, set())

def assert_prompt_no_ground_truth(prompt_payload: dict) -> None:
    keys={k.lower() for k in prompt_payload}
    bad=keys & FORBIDDEN_TEST_FIELDS
    if bad:
        raise ValueError(f"Ground-truth leakage: forbidden prompt fields {sorted(bad)}")
