import re
from dataclasses import dataclass
from .normalizer import NormalizationResult
from .parser import ParsedMessage


@dataclass(frozen=True)
class BehaviorSignals:
    urgency: bool
    threats: bool
    kyc: bool
    otp: bool
    upi: bool
    pin: bool
    cvv: bool
    password: bool
    reward: bool
    authority: bool
    requests_secret: bool
    combination_hits: list[str]


def behavior_signals(text: str) -> BehaviorSignals:
    value = text.lower()
    has = lambda pattern: bool(re.search(pattern, value))
    urgency = has(r"\b(?:urgent|immediately|within\s+\d+|today|last\s+warning|जल्दी)\b")
    threats = has(r"\b(?:blocked|suspend|legal action|arrest|penalty|deactivate| बंद)\b")
    kyc = has(r"\bkyc\b|know your customer|re-?verify")
    otp = has(r"\botp\b|one[- ]time password")
    upi = has(r"\bupi\b|vpa")
    pin = has(r"\bpin\b")
    cvv = has(r"\bcvv\b")
    password = has(r"\bpass(?:word|code)\b")
    reward = has(r"\b(?:prize|reward|lottery|winner|cashback|बधाई)\b")
    authority = has(r"\b(?:bank|rbi|police|government|income tax|customs|courier)\b")
    requests_secret = has(r"\b(?:share|send|provide|tell|enter|confirm|submit)\b.{0,45}\b(?:otp|pin|cvv|password|passcode)\b")
    combinations = []
    if kyc and threats: combinations.append("KYC plus account threat")
    if kyc and ("http://" in value or "https://" in value or "www." in value): combinations.append("KYC plus URL")
    if otp and requests_secret: combinations.append("OTP sharing request")
    if reward and has(r"\b(?:pay|fee|charge|deposit|upi|send)\b"): combinations.append("reward plus payment request")
    if has(r"\b(?:job|work from home|hiring)\b") and has(r"\b(?:fee|registration|pay)\b"): combinations.append("job offer plus fee")
    return BehaviorSignals(urgency, threats, kyc, otp, upi, pin, cvv, password, reward, authority, requests_secret, combinations)


def sender_signals(sender: str | None) -> dict[str, object]:
    if not sender:
        return {"type": "missing", "is_phone": False, "suspicious": True, "anomalies": ["sender unavailable"]}
    is_phone = bool(re.fullmatch(r"\+?[\d\s()-]{7,}", sender))
    is_header = bool(re.fullmatch(r"[A-Za-z0-9-]{2,15}", sender)) and not is_phone
    suspicious = bool(re.search(r"(?:free|prize|support|verify|alert|bank|admin)", sender, re.I))
    return {"type": "phone" if is_phone else ("header" if is_header else "other"), "is_phone": is_phone, "suspicious": suspicious, "anomalies": ["unusual sender pattern"] if suspicious else []}


def url_signals(parsed: ParsedMessage) -> dict[str, object]:
    results = []
    for raw, domain in zip(parsed.urls, parsed.domains):
        lowered = raw.lower()
        shortened = domain in {"bit.ly", "tinyurl.com", "t.co", "is.gd", "cutt.ly", "rb.gy"}
        suspicious_structure = bool(re.search(r"@", lowered) or re.search(r"[^\x00-\x7f]", domain) or domain.count("-") >= 3 or re.search(r"\d{5,}", domain))
        mismatch = bool(re.search(r"\b(?:bank|rbi|paytm|phonepe|sbi|upi|aadhaar)\b", parsed.raw, re.I)) and not any(name in domain for name in ("bank", "rbi", "paytm", "phonepe", "sbi", "upi"))
        results.append({"url": raw, "domain": domain, "shortened": shortened, "suspicious_structure": suspicious_structure, "brand_domain_mismatch": mismatch, "reputation": "not_checked"})
    return {"urls": results, "count": len(results), "evidence_basis": "structure-only; reputation and redirects were not checked"}


def language_hint(text: str) -> str:
    if re.search(r"[\u0900-\u097F]", text):
        return "hi"
    if re.search(r"[^\x00-\x7F]", text):
        return "mixed"
    return "en"