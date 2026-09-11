"""Use entity names for entity-indexed APIs; no answer annotations are accepted."""
from __future__ import annotations
import re
from paper4_kbvqa.knowledge.combined import CombinedKnowledgeProvider


class StructuredKnowledgeProvider:
    def __init__(self,providers): self.providers=list(providers)

    def retrieve_for_question(self,question,visual_entities,limit=10):
        # ConceptNet concept URIs and Wikidata entity search do not accept a full
        # visual question as an entity identifier. Prefer model-extracted entities.
        terms=[str(x).strip() for x in visual_entities if str(x).strip()]
        if not terms:
            stop={'what','which','where','when','why','how','this','that','these','those',
                  'does','would','could','image','picture','the','are','is','of','in','a','an','to'}
            terms=[x for x in re.findall(r'[A-Za-z][A-Za-z-]+',question.lower()) if x not in stop and len(x)>2]
        pooled=[]; seen=set()
        for provider in self.providers:
            queries=[question+' '+' '.join(terms[:4])] if provider.name=='wikipedia' else terms[:4]
            for query in queries:
                for evidence in provider.retrieve(query.strip(),limit=limit):
                    if evidence.evidence_id not in seen:
                        seen.add(evidence.evidence_id); pooled.append(evidence)
        return sorted(pooled,key=lambda e:e.retrieval_score,reverse=True)
