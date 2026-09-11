"""Study integrity and controlled perturbations; never generates answers."""
from __future__ import annotations
import hashlib
import json
import math
import random
from pathlib import Path
from paper4_kbvqa.types import Evidence
from paper4_kbvqa.evaluation.corruption import (
    source_dropout, evidence_scarcity, ranking_corruption, synthetic_contradiction,
)


def write_json(path, obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    def clean(x):
        if isinstance(x,float) and not math.isfinite(x): return None
        if isinstance(x,dict): return {k:clean(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)): return [clean(v) for v in x]
        return x
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(clean(obj),indent=2,allow_nan=False),encoding='utf-8')
    tmp.replace(path)


def freeze_json(path, obj):
    path=Path(path)
    if path.exists():
        if json.loads(path.read_text()) != obj:
            raise ValueError(f'Run identity changed: {path}. Use a new run directory.')
    else: write_json(path,obj)


def read_rows(path):
    rows=[json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    ids=[str(x['question_id']) for x in rows]
    if len(ids)!=len(set(ids)): raise ValueError(f'Duplicate question IDs in {path}')
    return rows


def validate_predictions(manifest, predictions):
    expected={str(x['question_id']) for x in read_rows(manifest)}
    rows=read_rows(predictions)
    actual={str(x['question_id']) for x in rows}
    if actual!=expected:
        raise ValueError(f'Prediction ID mismatch: missing={len(expected-actual)}, extra={len(actual-expected)}')
    for row in rows:
        if not isinstance(row.get('answer'),str) or not row['answer'].strip():
            raise ValueError('Empty or invalid model answer')
        c=row.get('raw_confidence')
        if not isinstance(c,(int,float)) or not math.isfinite(c) or not 0<=c<=1:
            raise ValueError('Invalid prediction confidence')
    return rows


class CorruptionTransform:
    """Question-seeded perturbations applied before filtering/generation.

    Contradictions are synthetic linguistic stress tests, not labelled factual
    contradictions. Cross-question distractors are not guaranteed irrelevant.
    """
    def __init__(self,spec='clean',seed=2026,distractor_predictions=None):
        self.spec=spec; self.seed=seed; self.pool=[]
        if spec.startswith('cross_question:'):
            if not distractor_predictions: raise ValueError('Calibration evidence is required')
            for row in read_rows(distractor_predictions):
                self.pool.extend(Evidence(**e) for e in row.get('retrieved_evidence',[]))
            if not self.pool: raise ValueError('No calibration evidence for distractors')

    def __call__(self,evidence,qid):
        if self.spec=='clean': return list(evidence)
        kind,value=self.spec.split(':',1)
        seed=int(hashlib.sha256(f'{self.seed}:{qid}:{self.spec}'.encode()).hexdigest()[:16],16)
        if kind=='drop': return source_dropout(evidence,source=value)
        if kind=='scarcity': return evidence_scarcity(evidence,k=int(value))
        if kind=='ranking': return ranking_corruption(evidence,severity=float(value),seed=seed)
        rng=random.Random(seed)
        if kind=='contradiction':
            candidates=list(evidence); rng.shuffle(candidates)
            return [synthetic_contradiction(e) for e in candidates[:int(value)]]+list(evidence)
        if kind=='cross_question':
            ids={e.evidence_id for e in evidence}
            pool=[e for e in self.pool if e.evidence_id not in ids]
            rng.shuffle(pool)
            from dataclasses import replace
            extra=[replace(e,evidence_id=e.evidence_id+'::distractor',metadata={**e.metadata,
                'corruption':'cross_question_injection','relevance_verified':False}) for e in pool[:int(value)]]
            return extra+list(evidence)
        raise ValueError('Unknown corruption: '+self.spec)
