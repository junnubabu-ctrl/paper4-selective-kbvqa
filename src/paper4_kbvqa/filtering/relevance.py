from __future__ import annotations
from dataclasses import dataclass
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from paper4_kbvqa.types import Evidence

TOKEN=re.compile(r"[A-Za-z0-9]+")
def _tokens(x: str) -> set[str]: return {t.lower() for t in TOKEN.findall(x)}
def _jaccard(a: str,b: str) -> float:
    x,y=_tokens(a),_tokens(b)
    return len(x&y)/len(x|y) if x|y else 0.0

def _cos(a: str,b: str) -> float:
    if not a.strip() or not b.strip(): return 0.0
    v=TfidfVectorizer(stop_words="english").fit_transform([a,b])
    return float(cosine_similarity(v[0],v[1])[0,0])

@dataclass(frozen=True)
class RelevanceWeights:
    semantic: float=.30; question: float=.20; entity: float=.18; visual: float=.14; source: float=.12; redundancy: float=.06

class RelevanceFilter:
    def __init__(self, weights: RelevanceWeights=RelevanceWeights(), min_score: float = 0.15):
        self.w=weights
        self.min_score=float(min_score)
    def score(self, evidence: Evidence, question: str, visual_entities: list[str], already_selected: list[Evidence]) -> dict:
        sem=_cos(question,evidence.text)
        qrel=_jaccard(question,evidence.text)
        ents=" ".join(visual_entities)
        ent=_jaccard(ents,evidence.text) if visual_entities else 0.0
        vis=_cos(ents,evidence.text) if visual_entities else 0.0
        source=max(0.0,min(1.0,float(evidence.retrieval_score)))
        red=max((_cos(evidence.text,e.text) for e in already_selected),default=0.0)
        score=self.w.semantic*sem+self.w.question*qrel+self.w.entity*ent+self.w.visual*vis+self.w.source*source-self.w.redundancy*red
        return {"score":float(score),"semantic":sem,"question":qrel,"entity":ent,"visual":vis,"source":source,"redundancy":red}
    def select(self, evidence: list[Evidence], question: str, visual_entities: list[str], top_k: int=5):
        remaining=list(evidence); selected=[]; details={}
        while remaining and len(selected)<top_k:
            scored=[(self.score(e,question,visual_entities,selected),e) for e in remaining]
            d,best=max(scored,key=lambda x:x[0]["score"])
            if d["score"] < self.min_score:
                break
            details[best.evidence_id]=d; selected.append(best); remaining.remove(best)
        return selected, details
