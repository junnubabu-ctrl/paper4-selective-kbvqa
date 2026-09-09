from __future__ import annotations
import re
from paper4_kbvqa.knowledge.base import KnowledgeProvider
from paper4_kbvqa.types import Evidence

_WS = re.compile(r"\s+")

def _norm(text: str) -> str:
    return _WS.sub(" ", text.strip().lower())

class CombinedKnowledgeProvider(KnowledgeProvider):
    name = "combined"
    def __init__(self, providers: list[KnowledgeProvider]):
        if not providers:
            raise ValueError("At least one knowledge provider is required")
        self.providers = list(providers)

    def retrieve(self, query: str, limit: int = 10) -> list[Evidence]:
        pooled: list[Evidence] = []
        seen: set[str] = set()
        per_source = max(1, limit)
        for provider in self.providers:
            for ev in provider.retrieve(query, limit=per_source):
                key = _norm(ev.text)
                if not key or key in seen:
                    continue
                seen.add(key)
                pooled.append(ev)
        pooled.sort(key=lambda e: float(e.retrieval_score), reverse=True)
        return pooled[: limit * len(self.providers)]
