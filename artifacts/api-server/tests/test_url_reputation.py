import asyncio

from app.fusion import fuse
from app.normalizer import normalize_message
from app.parser import parse_message
from app.rules import behavior_signals
from app.url_reputation import UrlReputationClient


def test_no_api_key_keeps_reputation_unchecked_without_network_calls():
    client = UrlReputationClient(api_key="")
    result = asyncio.run(client.check_urls(["https://example.com/path"]))

    assert result[0]["reputation"] == "not_checked"
    assert "not configured" in result[0]["detail"].lower()
    assert result[0]["provider"] == "Google Safe Browsing"


def test_url_normalizer_handles_bare_domains_and_rejects_credentials():
    assert UrlReputationClient._absolute_http_url("example.com/path") == "https://example.com/path"
    assert UrlReputationClient._absolute_http_url("http://example.com/a") == "http://example.com/a"
    assert UrlReputationClient._absolute_http_url("javascript:alert(1)") is None
    assert UrlReputationClient._absolute_http_url("https://user:secret@example.com/") is None
    assert UrlReputationClient._absolute_http_url("https://example.com:invalid/") is None


def test_google_api_key_is_sent_in_header_not_query_string(monkeypatch):
    captured = {}

    class FakeResponse:
        is_success = True
        status_code = 200

        def json(self):
            return {"matches": []}

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, **kwargs):
            captured["url"] = url
            captured.update(kwargs)
            return FakeResponse()

    monkeypatch.setattr("app.url_reputation.httpx.AsyncClient", lambda timeout: FakeClient())
    client = UrlReputationClient(api_key="test-key-do-not-log")
    result = asyncio.run(client._lookup_one("https://example.com/path"))

    assert result["reputation"] == "no_match"
    assert captured["url"] == client.ENDPOINT
    assert captured["headers"]["x-goog-api-key"] == "test-key-do-not-log"
    assert "params" not in captured
    assert "test-key-do-not-log" not in captured["url"]


def test_lookup_results_are_cached_and_only_unique_urls_are_looked_up(monkeypatch):
    client = UrlReputationClient(api_key="test-key")
    calls = []

    async def fake_lookup(url):
        calls.append(url)
        if "bad.example" in url:
            return {
                "reputation": "malicious",
                "provider": client.PROVIDER,
                "threat_types": ["SOCIAL_ENGINEERING"],
                "detail": "test match",
            }
        return {
            "reputation": "no_match",
            "provider": client.PROVIDER,
            "threat_types": [],
            "detail": "test no match",
        }

    monkeypatch.setattr(client, "_lookup_one", fake_lookup)
    urls = ["https://good.example/", "https://bad.example/", "https://bad.example/"]
    first = asyncio.run(client.check_urls(urls))
    second = asyncio.run(client.check_urls(urls))

    assert len(calls) == 2
    assert [item["reputation"] for item in first] == ["no_match", "malicious", "malicious"]
    assert [item["reputation"] for item in second] == ["no_match", "malicious", "malicious"]
    assert all(item["url"] for item in second)


def test_reputation_match_contributes_bounded_url_evidence():
    parsed = parse_message("Please review https://bad.example/login", None, None)
    enriched_urls = {
        "count": 1,
        "evidence_basis": "test reputation result",
        "urls": [
            {
                "url": "https://bad.example/login",
                "domain": "bad.example",
                "shortened": False,
                "ip_address_url": False,
                "suspicious_structure": False,
                "brand_domain_mismatch": False,
                "reputation": "malicious",
            }
        ],
    }
    result = fuse(
        parsed=parsed,
        normalized=normalize_message(parsed.raw),
        behavior=behavior_signals(parsed.raw),
        content_signal=0,
        similarity_signal=0,
        url_analysis=enriched_urls,
    )

    assert result.evidence["url"] == 12.0
    assert 0 <= result.score <= 100


def test_no_match_and_not_checked_do_not_add_reputation_score():
    parsed = parse_message("Please review https://example.com/path", None, None)
    for reputation in ("no_match", "not_checked"):
        result = fuse(
            parsed=parsed,
            normalized=normalize_message(parsed.raw),
            behavior=behavior_signals(parsed.raw),
            content_signal=0,
            similarity_signal=0,
            url_analysis={
                "count": 1,
                "urls": [
                    {
                        "url": parsed.urls[0],
                        "domain": parsed.url_domains[0],
                        "shortened": False,
                        "ip_address_url": False,
                        "suspicious_structure": False,
                        "brand_domain_mismatch": False,
                        "reputation": reputation,
                    }
                ],
            },
        )
        assert result.evidence["url"] == 0.0



def test_url_lookup_limit_is_enforced_and_skipped_urls_stay_unchecked(monkeypatch):
    client = UrlReputationClient(api_key="test-key")
    client.MAX_URLS_PER_MESSAGE = 2
    calls = []

    async def fake_lookup(url):
        calls.append(url)
        return {
            "reputation": "no_match",
            "provider": client.PROVIDER,
            "threat_types": [],
            "detail": "test no match",
        }

    monkeypatch.setattr(client, "_lookup_one", fake_lookup)
    urls = [f"https://site{i}.example/path" for i in range(3)]
    results = asyncio.run(client.check_urls(urls))

    assert calls == urls[:2]
    assert [result["reputation"] for result in results] == [
        "no_match",
        "no_match",
        "not_checked",
    ]
    assert "limit" in results[2]["detail"].lower()
