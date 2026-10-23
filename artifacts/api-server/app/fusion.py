from __future__ import annotations

from dataclasses import dataclass

from .config import settings
from .normalizer import NormalizationResult
from .parser import ParsedMessage
from .rules import (
    BehaviorSignals,
    sender_signals,
    url_signals,
)


# ============================================================
# RESULT
# ============================================================

@dataclass(frozen=True)
class FusionResult:
    score: int
    label: str
    evidence: dict[str, float]
    top: list[str]
    category: str


# ============================================================
# HELPERS
# ============================================================

def _clamp(
    value: float,
    minimum: float,
    maximum: float,
) -> float:
    return max(
        minimum,
        min(maximum, float(value)),
    )


def _sender_risk(
    sender: dict[str, object],
) -> float:
    """
    Convert sender-analysis findings into risk evidence.

    IMPORTANT:
    A trusted-brand claim such as "SBI" is NOT by itself
    evidence that the sender is malicious.

    We only score:
      - promotional sender identifiers
      - unusual sender characters

    A normal phone number/header contributes 0.
    """

    anomalies = sender.get(
        "anomalies",
        [],
    )

    if not isinstance(anomalies, list):
        return 0.0

    score = 0.0

    for anomaly in anomalies:
        text = str(anomaly).lower()

        if "promotional sender" in text:
            score += 8.0

        elif "unusual sender characters" in text:
            score += 6.0

        # Do NOT score:
        # "trusted-brand/authority claim in sender"
        #
        # A sender claiming "SBI" is not proof of spoofing.

    return _clamp(
        score,
        0.0,
        15.0,
    )


def _behavior_risk(
    behavior: BehaviorSignals,
) -> float:
    """
    Convert behavior signals into risk evidence.

    Important design rules:

      OTP alone             = 0
      PIN alone             = 0
      CVV alone             = 0
      UPI alone             = 0
      Authority alone       = 0

    Suspicion comes from combinations and social-engineering
    behavior, not from merely mentioning a security term.
    """

    score = 0.0

    # --------------------------------------------------------
    # Strong social-engineering signals
    # --------------------------------------------------------

    if behavior.urgency:
        score += 4.0

    if behavior.threats:
        score += 5.0

    # KYC alone is not necessarily malicious.
    # Example:
    # "Please complete your KYC before 31 December."
    #
    # Therefore only a small contextual contribution.
    if behavior.kyc:
        score += 1.0

    # Actual credential request is strong evidence.
    if behavior.requests_secret:
        score += 6.0

    # Reward/prize language is contextual.
    if behavior.reward:
        score += 2.0

    # --------------------------------------------------------
    # Combination evidence
    # --------------------------------------------------------

    combinations = set(
        behavior.combination_hits
    )

    if "KYC plus account threat" in combinations:
        score += 4.0

    if "KYC plus urgency" in combinations:
        score += 2.0

    if "KYC plus URL" in combinations:
        score += 3.0

    if "OTP sharing request" in combinations:
        score += 5.0

    if "credential sharing request" in combinations:
        score += 5.0

    if "reward plus payment request" in combinations:
        score += 4.0

    if "job offer plus fee" in combinations:
        score += 5.0

    if "authority plus KYC" in combinations:
        score += 1.0

    if "authority plus threat" in combinations:
        score += 2.0

    return _clamp(
        score,
        0.0,
        20.0,
    )


def _url_risk(
    urls: dict[str, object],
) -> float:
    """
    Convert URL structural findings into evidence.

    This does NOT treat "reputation = not_checked" as malicious.
    """

    score = 0.0

    items = urls.get(
        "urls",
        [],
    )

    if not isinstance(items, list):
        return 0.0

    for item in items:
        if not isinstance(item, dict):
            continue

        if item.get("shortened"):
            score += 4.0

        if item.get("suspicious_structure"):
            score += 4.0

        if item.get("ip_address_url"):
            score += 5.0

        if item.get("brand_domain_mismatch"):
            score += 6.0

        # A reputation provider match is stronger than URL structure alone.
        # A no_match or not_checked result contributes no maliciousness points.
        if item.get("reputation") == "malicious":
            score += 12.0

    return _clamp(
        score,
        0.0,
        20.0,
    )


def _obfuscation_risk(
    normalized: NormalizationResult,
) -> float:
    """
    Convert obfuscation findings into evidence.

    Transliteration itself is NOT malicious.
    It is a contextual feature.

    Deliberate spelling/leet manipulation is stronger evidence.
    """

    score = 0.0

    # Roman-Hinglish/transliteration alone should not create
    # a large phishing score.
    if normalized.transliteration:
        score += 0.5

    # Treat the primary mutation indicators as alternative descriptions of
    # a text change, not automatically independent evidence. A token corrected
    # by a known mapping can set both spelling_corrected and leetspeak; counting
    # both used to award 7 points for the same mutation.
    if normalized.spelling_corrected:
        score += 4.0
    elif normalized.leetspeak:
        score += 3.0
    elif getattr(normalized, "repeated_chars", False):
        score += 1.0

    if normalized.punctuation_manipulation:
        score += 1.0

    if normalized.spacing_manipulation:
        score += 1.0

    # Newer normalizer fields, when available. Repeated characters are
    # already accounted for above when no stronger mutation flag applies.
    if getattr(
        normalized,
        "concatenated_split",
        False,
    ):
        score += 1.0

    return _clamp(
        score,
        0.0,
        10.0,
    )


def _similarity_risk(
    similarity_signal: float,
    *,
    content: float,
    behavior: float,
    url: float,
    obfuscation: float,
) -> float:
    """
    Convert FAISS similarity into bounded evidence.

    CRITICAL RULE:

    Similarity by itself is NOT enough to create a phishing
    score because the current local index is a small demo
    knowledge base.

    Similarity is only allowed to contribute when there is
    already independent evidence from another detector.
    """

    similarity = _clamp(
        similarity_signal,
        0.0,
        5.0,
    )

    independent_evidence = (
        content >= 8.0
        or behavior >= 5.0
        or url > 0.0
        or obfuscation >= 4.0
    )

    if not independent_evidence:
        return 0.0

    return similarity


def _category(
    behavior: BehaviorSignals,
    urls: dict[str, object],
) -> str:
    """
    Determine an explainable category.

    This is rule-based classification, not an ML probability.
    """

    combinations = set(
        behavior.combination_hits
    )

    # Strong specific categories first.
    if (
        behavior.kyc
        and behavior.threats
    ):
        return "KYC_FRAUD"

    if (
        behavior.otp
        and behavior.requests_secret
    ):
        return "OTP_THEFT"

    if (
        behavior.requests_secret
        and (
            behavior.upi
            or behavior.pin
            or behavior.cvv
            or behavior.password
        )
    ):
        return "PAYMENT_CREDENTIAL_FRAUD"

    if "job offer plus fee" in combinations:
        return "JOB_SCAM"

    if "reward plus payment request" in combinations:
        return "LOTTERY_REWARD_SCAM"

    if (
        urls.get("count", 0)
        and (
            behavior.kyc
            or behavior.threats
            or behavior.requests_secret
        )
    ):
        return "LINK_PHISHING"

    if behavior.otp and not behavior.requests_secret:
        if not (
            behavior.threats
            or behavior.urgency
            or urls.get("count", 0)
        ):
            return "OTP_NOTICE"

    if combinations:
        return "PHISHING"

    if urls.get("count", 0):
        return "URL_REVIEW"

    return "GENERAL_MESSAGE"


# ============================================================
# MAIN FUSION
# ============================================================

def fuse(
    parsed: ParsedMessage,
    normalized: NormalizationResult,
    behavior: BehaviorSignals,
    content_signal: float,
    similarity_signal: float,
    url_analysis: dict[str, object] | None = None,
) -> FusionResult:
    """
    Final transparent evidence fusion.

    Architecture:

        pretrained model evidence
                  +
        behavior rules
                  +
        URL evidence
                  +
        sender evidence
                  +
        obfuscation evidence
                  +
        similarity evidence
                  |
                  v
        transparent weighted sum
                  |
                  v
        0-100 risk score
                  |
                  v
        LOW / SUSPICIOUS / HIGH RISK / CRITICAL

    No trained meta-model is used.
    Qwen3 does not participate in this calculation.
    """

    # --------------------------------------------------------
    # Independent evidence sources
    # --------------------------------------------------------

    sender = sender_signals(
        parsed.sender
    )

    # Use enriched reputation results when available. Callers that do not
    # provide URL analysis retain the original deterministic structural checks.
    urls = url_analysis if url_analysis is not None else url_signals(parsed)

    # --------------------------------------------------------
    # 1. CONTENT
    # --------------------------------------------------------

    content = _clamp(
        content_signal,
        0.0,
        30.0,
    )

    # --------------------------------------------------------
    # 2. URL
    # --------------------------------------------------------

    url_score = _url_risk(
        urls
    )

    # --------------------------------------------------------
    # 3. BEHAVIOR
    # --------------------------------------------------------

    behavior_score = _behavior_risk(
        behavior
    )

    # --------------------------------------------------------
    # 4. SENDER
    # --------------------------------------------------------

    sender_score = _sender_risk(
        sender
    )

    # --------------------------------------------------------
    # 5. OBFUSCATION
    # --------------------------------------------------------

    obfuscation_score = _obfuscation_risk(
        normalized
    )

    # --------------------------------------------------------
    # 6. SIMILARITY
    # --------------------------------------------------------

    similarity = _similarity_risk(
        similarity_signal,
        content=content,
        behavior=behavior_score,
        url=url_score,
        obfuscation=obfuscation_score,
    )

    # --------------------------------------------------------
    # FINAL EVIDENCE VECTOR
    # --------------------------------------------------------

    evidence: dict[str, float] = {
        "content": round(
            content,
            4,
        ),
        "sender": round(
            sender_score,
            4,
        ),
        "url": round(
            url_score,
            4,
        ),
        "behavior": round(
            behavior_score,
            4,
        ),
        "obfuscation": round(
            obfuscation_score,
            4,
        ),
        "similarity": round(
            similarity,
            4,
        ),
    }

    # --------------------------------------------------------
    # TOTAL SCORE
    # --------------------------------------------------------

    score = round(
        sum(evidence.values())
    )

    score = max(
        0,
        min(100, score),
    )

    # --------------------------------------------------------
    # SHORT-MESSAGE SAFETY CAP
    # --------------------------------------------------------

    if (
        parsed.length < 25
        and not parsed.sender
    ):
        score = min(
            score,
            70,
        )

    # --------------------------------------------------------
    # LABEL
    # --------------------------------------------------------

    if score < settings.low_threshold:
        label = "LOW"

    elif score < settings.suspicious_threshold:
        label = "SUSPICIOUS"

    elif score < settings.high_risk_threshold:
        label = "HIGH RISK"

    else:
        label = "CRITICAL"

    # --------------------------------------------------------
    # TOP EVIDENCE
    # --------------------------------------------------------

    top = [
        name
        for name, value in sorted(
            evidence.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        if value > 0
    ][:3]

    # --------------------------------------------------------
    # CATEGORY
    # --------------------------------------------------------

    category = _category(
        behavior,
        urls,
    )

    return FusionResult(
        score=score,
        label=label,
        evidence=evidence,
        top=top,
        category=category,
    )