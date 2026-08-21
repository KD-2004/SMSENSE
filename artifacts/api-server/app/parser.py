import re
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse

URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>()]+")


@dataclass(frozen=True)
class ParsedMessage:
    raw: str
    sender: str | None
    timestamp: datetime
    urls: list[str]
    domains: list[str]
    length: int


def parse_message(raw: str, sender: str | None, timestamp: datetime | None) -> ParsedMessage:
    urls = [url.rstrip(".,!?;:)]}") for url in URL_RE.findall(raw)]
    domains = []
    for value in urls:
        candidate = value if "://" in value else f"http://{value}"
        host = urlparse(candidate).hostname
        if host and host.lower() not in domains:
            domains.append(host.lower())
    return ParsedMessage(
        raw=raw,
        sender=sender,
        timestamp=timestamp or datetime.now(timezone.utc),
        urls=urls,
        domains=domains,
        length=len(raw),
    )