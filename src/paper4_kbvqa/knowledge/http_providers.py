from __future__ import annotations
import json, hashlib
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen
from paper4_kbvqa.knowledge.base import KnowledgeProvider
from paper4_kbvqa.types import Evidence

class CachedHTTPProvider(KnowledgeProvider):
    def __init__(self, cache_dir: str | Path = "cache/knowledge", timeout: int = 15):
        self.cache_dir=Path(cache_dir); self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout=timeout
    def _json(self, url: str) -> dict:
        key=hashlib.sha256(url.encode()).hexdigest()+".json"; p=self.cache_dir/key
        if p.exists(): return json.loads(p.read_text(encoding="utf-8"))
        req=Request(url, headers={"User-Agent":"paper4-kbvqa-research/0.1"})
        with urlopen(req, timeout=self.timeout) as r: data=json.loads(r.read().decode())
        p.write_text(json.dumps(data), encoding="utf-8"); return data

class WikipediaProvider(CachedHTTPProvider):
    name="wikipedia"
    def retrieve(self, query: str, limit: int = 10) -> list[Evidence]:
        url=f"https://en.wikipedia.org/w/api.php?action=query&list=search&format=json&utf8=1&srlimit={limit}&srsearch={quote(query)}"
        data=self._json(url); out=[]
        for rank,x in enumerate(data.get("query",{}).get("search",[]), 1):
            title=x.get("title",""); snippet=x.get("snippet","").replace('<span class="searchmatch">','').replace('</span>','')
            out.append(Evidence(f"wikipedia:{x.get('pageid')}", f"{title}. {snippet}", self.name, f"https://en.wikipedia.org/?curid={x.get('pageid')}", retrieval_score=1.0/rank))
        return out

class WikidataProvider(CachedHTTPProvider):
    name="wikidata"
    def retrieve(self, query: str, limit: int = 10) -> list[Evidence]:
        url=f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={quote(query)}&language=en&format=json&limit={limit}"
        data=self._json(url); out=[]
        for rank,x in enumerate(data.get("search",[]), 1):
            text=". ".join(t for t in [x.get("label",""), x.get("description","")] if t)
            out.append(Evidence(f"wikidata:{x.get('id')}", text, self.name, x.get("concepturi"), (x.get("id"),) if x.get("id") else (), retrieval_score=1.0/rank))
        return out

class ConceptNetProvider(CachedHTTPProvider):
    name="conceptnet"
    def retrieve(self, query: str, limit: int = 10) -> list[Evidence]:
        term=quote(query.lower().replace(" ","_"))
        url=f"https://api.conceptnet.io/c/en/{term}?offset=0&limit={limit}"
        data=self._json(url); out=[]
        for i,x in enumerate(data.get("edges",[])[:limit]):
            text=x.get("surfaceText") or f"{x.get('start',{}).get('label','')} {x.get('rel',{}).get('label','')} {x.get('end',{}).get('label','')}"
            out.append(Evidence(f"conceptnet:{i}:{hashlib.md5(text.encode()).hexdigest()[:8]}", text, self.name, x.get("@id"), retrieval_score=1.0/(i+1)))
        return out
