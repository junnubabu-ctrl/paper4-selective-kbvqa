from __future__ import annotations
import json, hashlib
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
import threading
import time
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen
from paper4_kbvqa.knowledge.base import KnowledgeProvider
from paper4_kbvqa.types import Evidence

HTTP_USER_AGENT = "paper4-kbvqa-research/0.2 (https://github.com/junnubabu-ctrl/paper4-selective-kbvqa)"
WIKIMEDIA_RATE_POLICY = {
    "version": "wikimedia-respectful-development-v1",
    "scope": "shared_in_process_wikipedia_and_wikidata",
    "minimum_request_interval_seconds": 6,
    "missing_or_invalid_retry_after_seconds": 60,
    "maximum_cooldown_wait_seconds": 30,
    "automatic_retry_attempts": 0,
    "long_cooldown_behavior": "uncached_requests_fail_closed_for_remainder_of_invocation",
    "cache_hits_make_no_network_request": True,
    "user_agent": HTTP_USER_AGENT,
}


def retry_after_seconds(value: str | None, *, wall_time: float) -> tuple[float, str]:
    """Honor both standard Retry-After forms; malformed headers fail conservatively."""
    if value is not None:
        text = str(value).strip()
        if text.isascii() and text.isdecimal():
            return float(int(text)), "delta_seconds"
        try:
            date = parsedate_to_datetime(text)
            if date.tzinfo is None:
                date = date.replace(tzinfo=timezone.utc)
            return max(0.0, date.timestamp() - wall_time), "http_date"
        except (TypeError, ValueError, OverflowError):
            pass
    return float(WIKIMEDIA_RATE_POLICY["missing_or_invalid_retry_after_seconds"]), "conservative_fallback"


class WikimediaRateLimitError(RuntimeError):
    """A real rate-limit response or a local cooldown block, distinguished in audit."""
    def __init__(self, url: str, decision: dict, *, http_status: int | None = None):
        self.url = url
        self.decision = decision
        self.http_status = http_status
        remaining = float(decision.get("cooldown_remaining_seconds", 0))
        message = (f"Wikimedia rate-limit protection: {decision['action']}; "
                   f"cooldown remaining {remaining:.3f}s. No automatic retry or provider omission. "
                   "Review the request audit and wait until Retry-After has elapsed before a new invocation.")
        super().__init__(message)


class WikimediaRequestScheduler:
    """Shared request pacing and cooldown; not coordination with other processes/IP users."""
    def __init__(self, *, clock=time.monotonic, wall_time=time.time, sleep=time.sleep):
        self.clock = clock
        self.wall_time = wall_time
        self.sleep = sleep
        self.next_request_at = 0.0
        self.cooldown_until = 0.0
        self.unavailable = False
        self._lock = threading.Lock()

    def before_request(self, url: str) -> dict:
        with self._lock:
            now = self.clock()
            remaining = max(0.0, self.cooldown_until - now)
            if self.unavailable:
                raise WikimediaRateLimitError(url, {
                    "action": "blocked_remainder_of_invocation", "network_request_started": False,
                    "cooldown_remaining_seconds": remaining})
            wait = max(0.0, self.next_request_at - now, remaining)
            if wait > WIKIMEDIA_RATE_POLICY["maximum_cooldown_wait_seconds"]:
                self.unavailable = True
                raise WikimediaRateLimitError(url, {
                    "action": "cooldown_exceeds_wait_budget", "network_request_started": False,
                    "cooldown_remaining_seconds": remaining, "required_wait_seconds": wait})
            started_wait = now
            due = now + wait
            while True:
                remaining_wait = due - self.clock()
                if remaining_wait <= 0:
                    break
                self.sleep(remaining_wait)
            now = self.clock()
            self.next_request_at = now + WIKIMEDIA_RATE_POLICY["minimum_request_interval_seconds"]
            return {"action": "request_allowed", "network_request_started": True,
                    "waited_seconds": now - started_wait,
                    "cooldown_remaining_seconds": max(0.0, self.cooldown_until - now)}

    def rate_limited(self, retry_after: str | None) -> dict:
        with self._lock:
            seconds, parsed_as = retry_after_seconds(retry_after, wall_time=self.wall_time())
            now = self.clock()
            self.cooldown_until = max(self.cooldown_until, now + seconds)
            remaining = max(0.0, self.cooldown_until - now)
            if remaining > WIKIMEDIA_RATE_POLICY["maximum_cooldown_wait_seconds"]:
                self.unavailable = True
            return {"action": "long_cooldown_abort_invocation" if self.unavailable else "cooldown_set_no_retry",
                    "network_request_started": True, "retry_after_raw": retry_after,
                    "retry_after_parsed_as": parsed_as, "retry_after_seconds": seconds,
                    "cooldown_remaining_seconds": remaining,
                    "retry_not_before_utc": datetime.fromtimestamp(
                        self.wall_time() + remaining, timezone.utc).isoformat(),
                    "automatic_retry_attempts": 0}


_WIKIMEDIA_SCHEDULER = WikimediaRequestScheduler()


class CachedHTTPProvider(KnowledgeProvider):
    def __init__(self, cache_dir: str | Path = "cache/knowledge", timeout: int = 15,
                 rate_scheduler: WikimediaRequestScheduler | None = None):
        self.cache_dir=Path(cache_dir); self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout=timeout
        self.rate_scheduler = ((rate_scheduler or _WIKIMEDIA_SCHEDULER)
                               if self.name in {"wikipedia", "wikidata"} else None)

    def _audit(self, url: str, event: str, **details) -> None:
        row = {"schema": "paper4-knowledge-http-request-v1", "provider": self.name,
               "url": url, "event": event,
               "recorded_utc": datetime.now(timezone.utc).isoformat(),
               "rate_policy": WIKIMEDIA_RATE_POLICY if self.rate_scheduler is not None else None,
               **details}
        with (self.cache_dir / "request_audit.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()

    def _json(self, url: str) -> dict:
        key=hashlib.sha256(url.encode()).hexdigest()+".json"; p=self.cache_dir/key
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            self._audit(url, "cache_hit", http_status=None, network_request_started=False,
                        cache_file=p.name, cache_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
            return data
        decision = None
        if self.rate_scheduler is not None:
            try:
                decision = self.rate_scheduler.before_request(url)
            except WikimediaRateLimitError as exc:
                self._audit(url, "cooldown_blocked", http_status=None,
                            network_request_started=False, cooldown_decision=exc.decision)
                raise
        req=Request(url, headers={"User-Agent": HTTP_USER_AGENT})
        started = datetime.now(timezone.utc).isoformat()
        self._audit(url, "request_started", http_status=None, network_request_started=True,
                    request_started_utc=started, scheduler_decision=decision)
        try:
            with urlopen(req, timeout=self.timeout) as r:
                body = r.read()
                data = json.loads(body.decode())
                status = getattr(r, "status", 200)
        except HTTPError as exc:
            if exc.code == 429 and self.rate_scheduler is not None:
                cooldown = self.rate_scheduler.rate_limited(exc.headers.get("Retry-After") if exc.headers else None)
                self._audit(url, "http_rate_limited", http_status=exc.code,
                            request_started_utc=started, network_request_started=True,
                            cooldown_decision=cooldown)
                raise WikimediaRateLimitError(url, cooldown, http_status=exc.code) from exc
            self._audit(url, "http_error", http_status=exc.code, network_request_started=True,
                        request_started_utc=started, error_type=type(exc).__name__, error=str(exc))
            raise
        except Exception as exc:
            self._audit(url, "request_error", http_status=None, network_request_started=True,
                        request_started_utc=started, error_type=type(exc).__name__, error=str(exc))
            raise
        p.write_text(json.dumps(data), encoding="utf-8")
        self._audit(url, "response_cached", http_status=status, network_request_started=True,
                    request_started_utc=started, response_sha256=hashlib.sha256(body).hexdigest(),
                    cache_file=p.name)
        return data

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
