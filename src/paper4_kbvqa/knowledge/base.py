from __future__ import annotations
from abc import ABC, abstractmethod
from paper4_kbvqa.types import Evidence

class KnowledgeProvider(ABC):
    name: str
    @abstractmethod
    def retrieve(self, query: str, limit: int = 10) -> list[Evidence]: ...
