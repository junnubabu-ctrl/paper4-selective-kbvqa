"""Fake-clock/network contracts only; these are not live-provider or GPU evidence."""
import json
from datetime import datetime, timezone
from email.utils import format_datetime
from io import BytesIO
from urllib.error import HTTPError

import pytest

from paper4_kbvqa.knowledge import http_providers as http


class Clock:
    def __init__(self):
        self.now = 0.0
        self.epoch = 1791498600.0
        self.waits = []

    def monotonic(self):
        return self.now

    def wall_time(self):
        return self.epoch + self.now

    def sleep(self, seconds):
        assert 0 < seconds <= 30
        self.waits.append(seconds)
        self.now += seconds

    def scheduler(self):
        return http.WikimediaRequestScheduler(clock=self.monotonic,
                                             wall_time=self.wall_time, sleep=self.sleep)


class Response(BytesIO):
    status = 200

    def __init__(self):
        super().__init__(b'{"TEST_ONLY": true}')


def audit(provider):
    return [json.loads(line) for line in
            (provider.cache_dir / "request_audit.jsonl").read_text().splitlines()]


def test_shared_scheduler_paces_wikipedia_and_wikidata_and_cache_avoids_network(monkeypatch, tmp_path):
    clock = Clock()
    scheduler = clock.scheduler()
    wikipedia = http.WikipediaProvider(tmp_path / "wikipedia", rate_scheduler=scheduler)
    wikidata = http.WikidataProvider(tmp_path / "wikidata", rate_scheduler=scheduler)
    calls = []
    def open_fixture(request, timeout):
        calls.append((clock.now, request.full_url, request.get_header("User-agent")))
        return Response()
    monkeypatch.setattr(http, "urlopen", open_fixture)
    wikipedia._json("https://en.wikipedia.org/w/api.php?TEST_ONLY=one")
    wikidata._json("https://www.wikidata.org/w/api.php?TEST_ONLY=two")
    wikipedia._json("https://en.wikipedia.org/w/api.php?TEST_ONLY=one")
    assert [row[0] for row in calls] == [0.0, 6.0]
    assert clock.waits == [6.0]
    assert all(row[2] == http.HTTP_USER_AGENT for row in calls)
    assert "https://github.com/junnubabu-ctrl/paper4-selective-kbvqa" in http.HTTP_USER_AGENT
    assert audit(wikipedia)[-1]["event"] == "cache_hit"
    assert audit(wikipedia)[-1]["network_request_started"] is False
    assert audit(wikidata)[0]["scheduler_decision"]["waited_seconds"] == 6
    assert audit(wikidata)[-1]["rate_policy"] == http.WIKIMEDIA_RATE_POLICY


@pytest.mark.parametrize("retry_after,expected_wait", [("12", 12), ("30", 30), ("HTTP_DATE", 15)])
def test_short_retry_after_is_honored_before_next_request_with_no_automatic_retry(monkeypatch, tmp_path, retry_after, expected_wait):
    clock = Clock()
    if retry_after == "HTTP_DATE":
        retry_after = format_datetime(datetime.fromtimestamp(clock.wall_time() + expected_wait,
                                                             timezone.utc), usegmt=True)
    scheduler = clock.scheduler()
    first = http.WikipediaProvider(tmp_path / "wikipedia", rate_scheduler=scheduler)
    second = http.WikidataProvider(tmp_path / "wikidata", rate_scheduler=scheduler)
    calls = []
    def open_fixture(request, timeout):
        calls.append(clock.now)
        if len(calls) == 1:
            raise HTTPError(request.full_url, 429, "TEST_ONLY Too Many Requests",
                            {"Retry-After": retry_after}, None)
        return Response()
    monkeypatch.setattr(http, "urlopen", open_fixture)
    with pytest.raises(http.WikimediaRateLimitError) as error:
        first._json("https://en.wikipedia.org/w/api.php?TEST_ONLY=limited")
    assert calls == [0.0] and clock.waits == []
    assert error.value.http_status == 429
    second._json("https://www.wikidata.org/w/api.php?TEST_ONLY=next")
    assert calls == [0.0, float(expected_wait)]
    assert clock.waits == [float(expected_wait)]
    limited = audit(first)[-1]
    assert limited["event"] == "http_rate_limited" and limited["http_status"] == 429
    assert limited["cooldown_decision"]["automatic_retry_attempts"] == 0
    assert limited["cooldown_decision"]["retry_after_seconds"] == expected_wait


@pytest.mark.parametrize("retry_after,expected_wait", [("45", 45), (None, 60), ("invalid", 60)])
def test_long_cooldown_blocks_all_later_uncached_wikimedia_calls_even_after_elapsed_time(monkeypatch, tmp_path, retry_after, expected_wait):
    clock = Clock()
    scheduler = clock.scheduler()
    first = http.WikipediaProvider(tmp_path / "wikipedia", rate_scheduler=scheduler)
    second = http.WikidataProvider(tmp_path / "wikidata", rate_scheduler=scheduler)
    calls = []
    def open_fixture(request, timeout):
        calls.append((clock.now, request.full_url))
        headers = {} if retry_after is None else {"Retry-After": retry_after}
        raise HTTPError(request.full_url, 429, "TEST_ONLY Too Many Requests", headers, None)
    monkeypatch.setattr(http, "urlopen", open_fixture)
    with pytest.raises(http.WikimediaRateLimitError):
        first._json("https://en.wikipedia.org/w/api.php?TEST_ONLY=limited")
    assert len(calls) == 1 and scheduler.unavailable is True
    for elapsed in [0, expected_wait + 1]:
        clock.now = float(elapsed)
        with pytest.raises(http.WikimediaRateLimitError, match="blocked_remainder_of_invocation"):
            second._json("https://www.wikidata.org/w/api.php?TEST_ONLY=blocked")
    assert len(calls) == 1 and clock.waits == []
    blocked = audit(second)
    assert len(blocked) == 2
    assert all(row["event"] == "cooldown_blocked" and row["http_status"] is None
               and row["network_request_started"] is False for row in blocked)
    assert audit(first)[-1]["cooldown_decision"]["retry_after_seconds"] == expected_wait


def test_retry_after_date_in_past_and_invalid_header_handling():
    now = 1791498600.0
    past = format_datetime(datetime.fromtimestamp(now - 10, timezone.utc), usegmt=True)
    assert http.retry_after_seconds(past, wall_time=now) == (0, "http_date")
    assert http.retry_after_seconds("0", wall_time=now) == (0, "delta_seconds")
    assert http.retry_after_seconds("-5", wall_time=now) == (60, "conservative_fallback")


def test_advancing_clock_never_passes_negative_duration_to_sleep():
    # The old condition/sleep pair read 5.9 then 6.1, requesting sleep(-0.1).
    reads = iter([0.0, 5.9, 6.1, 6.2])
    waits = []
    def sleep_fixture(seconds):
        assert seconds > 0
        waits.append(seconds)
    scheduler = http.WikimediaRequestScheduler(clock=lambda: next(reads),
                                             wall_time=lambda: 0, sleep=sleep_fixture)
    scheduler.next_request_at = 6.0
    decision = scheduler.before_request("https://www.wikidata.org/w/api.php?TEST_ONLY=clock_race")
    assert waits == pytest.approx([0.1])
    assert decision["action"] == "request_allowed"
    assert scheduler.next_request_at == pytest.approx(12.2)


def test_cli_plan_explicitly_identifies_new_rate_policy(tmp_path, capsys):
    import importlib.util
    from pathlib import Path
    script = Path(__file__).resolve().parents[1] / "scripts/run_fixed_candidate_pilot.py"
    spec = importlib.util.spec_from_file_location("rate_policy_cli_test", script)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    assert cli.main(["--pilot-dir", str(tmp_path / "TEST_ONLY_prepared"),
                     "--out-dir", str(tmp_path / "output"), "--plan"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["wikimedia_rate_policy"] == http.WIKIMEDIA_RATE_POLICY
    assert plan["wikimedia_rate_policy"]["maximum_cooldown_wait_seconds"] == 30
    assert plan["continue_on_invalid"] is False
