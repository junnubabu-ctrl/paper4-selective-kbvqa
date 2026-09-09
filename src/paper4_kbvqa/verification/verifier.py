from __future__ import annotations
from dataclasses import dataclass
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from paper4_kbvqa.types import Evidence

def _cos(a,b):
    if not a.strip() or not b.strip(): return 0.0
    x=TfidfVectorizer(stop_words="english").fit_transform([a,b]); return float(cosine_similarity(x[0],x[1])[0,0])

def _entity_overlap(answer: str, evidence: str) -> float:
    toks=lambda s:{x.lower() for x in re.findall(r"[A-Za-z0-9]+",s) if len(x)>2}
    a,e=toks(answer),toks(evidence); return len(a&e)/len(a) if a else 0.0

def _contradiction_heuristic(answer: str,evidence: str) -> float:
    neg={"no","not","never","none","without","cannot","can't"}
    a=neg & set(answer.lower().split()); e=neg & set(evidence.lower().split())
    return 1.0 if bool(a) != bool(e) and _cos(answer,evidence)>.15 else 0.0

@dataclass(frozen=True)
class VerifierWeights:
    semantic: float=.34; entity: float=.18; visual: float=.14; retrieval: float=.18; provenance: float=.12; contradiction: float=.04

class EvidenceVerifier:
    def __init__(self, weights: VerifierWeights=VerifierWeights()): self.w=weights
    def verify(self, answer: str, evidence: list[Evidence], visual_consistency: float=0.5) -> dict:
        joined=" ".join(e.text for e in evidence)
        sem=_cos(answer,joined)
        ent=_entity_overlap(answer,joined)
        ret=(sum(max(0,min(1,e.retrieval_score)) for e in evidence)/len(evidence)) if evidence else 0.0
        prov=(len({e.source for e in evidence})/3.0) if evidence else 0.0
        prov=min(1.0,prov)
        con=max((_contradiction_heuristic(answer,e.text) for e in evidence),default=0.0)
        vis=max(0.0,min(1.0,float(visual_consistency)))
        raw=self.w.semantic*sem+self.w.entity*ent+self.w.visual*vis+self.w.retrieval*ret+self.w.provenance*prov-self.w.contradiction*con
        score=max(0.0,min(1.0,raw))
        return {"verification_score":score,"semantic":sem,"entity":ent,"visual":vis,"retrieval":ret,"provenance":prov,"contradiction":con}
