from dataclasses import dataclass
from .normalizer import NormalizationResult
from .parser import ParsedMessage
from .rules import BehaviorSignals, sender_signals, url_signals
from .config import settings


@dataclass(frozen=True)
class FusionResult:
    score: int
    label: str
    evidence: dict[str, float]
    top: list[str]
    category: str


def fuse(parsed: ParsedMessage, normalized: NormalizationResult, behavior: BehaviorSignals, content_signal: float, similarity_signal: float) -> FusionResult:
    sender = sender_signals(parsed.sender)
    urls = url_signals(parsed)
    content = min(30.0, max(0.0, content_signal))
    url = min(20.0, sum(5 for item in urls["urls"] if item["shortened"] or item["suspicious_structure"] or item["brand_domain_mismatch"]))
    behavior_score = min(20.0, len(behavior.combination_hits) * 7 + sum([behavior.urgency, behavior.threats, behavior.requests_secret]) * 2)
    sender_score = min(15.0, (8 if sender["suspicious"] else 0) + (4 if sender["type"] == "missing" else 0))
    obfuscation_score = min(10.0, sum([normalized.transliteration, bool(normalized.misspellings), normalized.leetspeak, normalized.punctuation_manipulation, normalized.spacing_manipulation]) * 2)
    similarity = min(5.0, max(0.0, similarity_signal))
    evidence = {"content": content, "sender": sender_score, "url": url, "behavior": behavior_score, "obfuscation": obfuscation_score, "similarity": similarity}
    score = round(sum(evidence.values()))
    if parsed.length < 25 and not parsed.sender:
        score = min(score, 70)
    label = "LOW" if score < settings.low_threshold else "SUSPICIOUS" if score < settings.suspicious_threshold else "HIGH RISK" if score < settings.high_risk_threshold else "CRITICAL"
    top = [name for name, value in sorted(evidence.items(), key=lambda item: item[1], reverse=True) if value > 0][:3]
    category = "credential_theft" if behavior.requests_secret else "phishing" if urls["count"] or behavior.combination_hits else "spam_or_scam"
    return FusionResult(score, label, evidence, top, category)