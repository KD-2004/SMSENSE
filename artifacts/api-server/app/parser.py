import ipaddress
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse

# Candidate match includes explicit URLs, www links, and bare domain-like tokens.
# URLs are parsed only; this module never resolves or fetches them.
URL_RE = re.compile(
    r"(?ix)"
    r"(?<![@\w])"
    r"(?:"
    r"(?:https?://|www\.)[^\s<>\"']+"
    r"|"
    r"(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+"
    r"(?:[a-z]{2,63}|xn--[a-z0-9-]{2,59})"
    r"(?::\d{1,5})?"
    r"(?:/[^\s<>\"']*)?"
    r")"
)
TRAILING_PUNCTUATION = ".,!?;:)]}»”’"
LEADING_PUNCTUATION = "([{<“‘"


@dataclass(frozen=True)
class ParsedMessage:
    raw: str
    sender: str | None
    timestamp: datetime
    urls: list[str]
    domains: list[str]
    length: int


def _clean_candidate(value: str) -> str:
    return value.strip().lstrip(LEADING_PUNCTUATION).rstrip(TRAILING_PUNCTUATION)


def _extract_host(value: str) -> str | None:
    candidate = value.strip()
    if candidate.lower().startswith("www."):
        candidate = "http://" + candidate
    elif not re.match(r"(?i)^https?://", candidate):
        candidate = "http://" + candidate

    try:
        parsed = urlparse(candidate)
        host = parsed.hostname
        if not host:
            return None
        host = host.rstrip(".").lower()
        if parsed.port is not None and not (1 <= parsed.port <= 65535):
            return None
        try:
            host = ipaddress.ip_address(host).compressed
        except ValueError:
            pass
        return host or None
    except (ValueError, UnicodeError):
        return None


def parse_message(raw: str, sender: str | None, timestamp: datetime | None) -> ParsedMessage:
    urls: list[str] = []
    domains: list[str] = []
    seen_urls: set[str] = set()
    seen_domains: set[str] = set()

    for match in URL_RE.finditer(raw or ""):
        url = _clean_candidate(match.group(0))
        if not url:
            continue
        host = _extract_host(url)
        if host is None:
            continue

        url_key = url.casefold()
        if url_key not in seen_urls:
            seen_urls.add(url_key)
            urls.append(url)

        if host not in seen_domains:
            seen_domains.add(host)
            domains.append(host)

    return ParsedMessage(
        raw=raw,
        sender=sender,
        timestamp=timestamp or datetime.now(timezone.utc),
        urls=urls,
        domains=domains,
        length=len(raw),
    )
