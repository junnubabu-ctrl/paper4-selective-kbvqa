from __future__ import annotations
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TfidfRetriever:
    """Deterministic CPU fallback for tests/dev; not the final semantic retriever baseline."""
    def __init__(self, corpus: list[str]):
        self.corpus=list(corpus)
        self.vectorizer=TfidfVectorizer(stop_words="english")
        self.matrix=self.vectorizer.fit_transform(self.corpus) if self.corpus else None
    def search(self, query: str, top_k: int = 5) -> list[tuple[int,float]]:
        if self.matrix is None: return []
        q=self.vectorizer.transform([query])
        scores=cosine_similarity(q,self.matrix)[0]
        order=np.argsort(-scores)[:top_k]
        return [(int(i),float(scores[i])) for i in order]
