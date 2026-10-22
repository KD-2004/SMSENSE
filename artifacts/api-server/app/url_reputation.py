"""Optional Google Safe Browsing URL lookups.

Only URL strings are sent to Google; SMS bodies and sender values are never sent.
When no API key is configured, URL reputation remains not_checked and the
main analysis pipeline continues with local structural evidence only.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

from .config import settings

logger = logging.getLogger(__name__)


class UrlReputationClient:
    """Bounded, opt-in Google Safe Browsing Lookup API client."""

    ENDPOINT = "https://safebrowsing.googleapis.com/v4/threatMatches:find"
    PROVIDER = "Google Safe Browsing"
    MAX_URLS_PER_MESSAGE = 10
    MAX_CONCURRENT_LOOKUPS = 3
    CACHE_TTL_SECONDS = 3600
    MAX_CACHE_ENTRIES = 2048
    REQUEST_TIMEOUT_SECONDS = 5.0

    def __init__(self, api_key: str | None = None) -> None:
        configured_key = (
            settings.google_safe_browsing_api_key if api_key is None else api_key
        )
        self.api_key = configured_key.strip()
        self._cache: dict[str, tuple[float, dict[str, Any]]] = {}

    @staticmethod
    def _absolute_http_url(raw_url: str) -> str | None:
        """Make a parsed bare domain explicit and reject unsupported schemes."""
        value = (raw_url or "").strip()
        if not value:
            return None
        if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", value):
            if not value.lower().startswith(("http://", "https://")):
                return None
        else:
            value = "https://" + value
        try:
            parts = urlsplit(value)
            if parts.scheme not in {"http", "https"} or not parts.hostname:
                return None
            # Force port parsing so malformed ports are rejected.
            if parts.port is not None and not 1 <= parts.port <= 65535:
                return None
            if parts.username is not None or parts.password is not None:
                return None
            return urlunsplit(parts)
        except (ValueError, UnicodeError):
            return None

    @staticmethod
    def _not_checked(detail: str) -> dict[str, Any]:
        return {
            "reputation": "not_checked",
            "provider": UrlReputationClient.PROVIDER,
            "threat_types": [],
            "detail": detail,
        }

    async def _lookup_one(self, raw_url: str) -> dict[str, Any]:
        """Perform one lookup so a provider match is never misattributed to another URL."""
        if not self.api_key:
            return self._not_checked("API key not configured; only structural checks ran")

        url = self._absolute_http_url(raw_url)
        if url is None:
            return self._not_checked("URL could not be normalized for reputation lookup")

        request_body = {
            "client": {
                "clientId": "smsense",
                "clientVersion": "1.0",
            },
            "threatInfo": {
                "threatTypes": [
                    "MALWARE",
                    "SOCIAL_ENGINEERING",
                    "UNWANTED_SOFTWARE",
                    "POTENTIALLY_HARMFUL_APPLICATION",
                ],
                "platformTypes": ["ANY_PLATFORM"],
                "threatEntryTypes": ["URL"],
                "threatEntries": [{"url": url}],
            },
        }
        try:
            async with httpx.AsyncClient(timeout=self.REQUEST_TIMEOUT_SECONDS) as client:
                response = await client.post(
                    self.ENDPOINT,
                    params={"key": self.api_key},
                    json=request_body,
                )
            if not response.is_success:
                logger.warning("URL reputation provider returned HTTP %s", response.status_code)
                return self._not_checked(f"Provider request failed (HTTP {response.status_code})")
            payload = response.json()
            if not isinstance(payload, dict):
                return self._not_checked("Provider returned an unexpected response")
            matches = payload.get("matches", [])
            if not isinstance(matches, list):
                return self._not_checked("Provider returned an unexpected matches field")
            threat_types = sorted(
                {
                    str(match.get("threatType"))
                    for match in matches
                    if isinstance(match, dict) and match.get("threatType")
                }
            )
            if matches:
                return {
                    "reputation": "malicious",
                    "provider": self.PROVIDER,
                    "threat_types": threat_types,
                    "detail": "URL matched a Google Safe Browsing threat list",
                }
            return {
                "reputation": "no_match",
                "provider": self.PROVIDER,
                "threat_types": [],
                "detail": "No threat-list match returned; this is not a guarantee the URL is safe",
            }
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            logger.warning("URL reputation lookup failed (%s)", type(exc).__name__)
            return self._not_checked(f"Provider lookup failed ({type(exc).__name__})")
        except Exception as exc:
            # Reputation is an optional enrichment. It must never take down SMS analysis.
            logger.exception("Unexpected URL reputation lookup failure")
            return self._not_checked(f"Provider lookup failed ({type(exc).__name__})")

    async def check_urls(self, urls: list[str]) -> list[dict[str, Any]]:
        """Return one result per input URL, bounded by count, time and concurrency."""
        if not urls:
            return []

        unique_urls = list(dict.fromkeys(urls))
        results_by_url: dict[str, dict[str, Any]] = {}
        if not self.api_key:
            for url in unique_urls:
                results_by_url[url] = self._not_checked(
                    "API key not configured; only structural checks ran"
                )
        else:
            now = time.monotonic()
            uncached: list[str] = []
            for url in unique_urls:
                cached = self._cache.get(url)
                if cached and now - cached[0] < self.CACHE_TTL_SECONDS:
                    results_by_url[url] = dict(cached[1])
                else:
                    uncached.append(url)

            to_check = uncached[: self.MAX_URLS_PER_MESSAGE]
            skipped = uncached[self.MAX_URLS_PER_MESSAGE :]
            semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_LOOKUPS)

            async def bounded_lookup(url: str) -> tuple[str, dict[str, Any]]:
                async with semaphore:
                    return url, await self._lookup_one(url)

            looked_up = await asyncio.gather(*(bounded_lookup(url) for url in to_check))
            checked_at = time.monotonic()
            for url, result in looked_up:
                results_by_url[url] = result
                # Cache only successful malicious/no_match answers, not provider failures.
                if result.get("reputation") in {"malicious", "no_match"}:
                    self._cache[url] = (checked_at, dict(result))

            if len(self._cache) > self.MAX_CACHE_ENTRIES:
                expired = [
                    key
                    for key, (created_at, _value) in self._cache.items()
                    if checked_at - created_at >= self.CACHE_TTL_SECONDS
                ]
                for key in expired:
                    self._cache.pop(key, None)
                while len(self._cache) > self.MAX_CACHE_ENTRIES:
                    self._cache.pop(next(iter(self._cache)))

            for url in skipped:
                results_by_url[url] = self._not_checked(
                    f"Per-message limit reached; at most {self.MAX_URLS_PER_MESSAGE} URLs are checked per analysis"
                )

        return [
            {"url": url, **results_by_url[url]}
            for url in urls
        ]
