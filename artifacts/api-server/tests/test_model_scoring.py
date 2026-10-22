import asyncio
import json

from app.models import (
    HuggingFaceModels,
    ModelUnavailableError,
    SmsShieldClient,
)
from app.schemas import AnalysisResponse
from app.service import AnalysisService


def _assessment(
    risk=15,
    verdict="safe",
    category="personal",
    red_flags=None,
    segments=1,
):
    return {
        "model": "SMS-Shield",
        "model_id": "hf.co/ganmoor-ai-labs/sms-shield:Q4_K_M",
        "risk": risk,
        "verdict": verdict,
        "model_verdict": verdict,
        "verdict_consistent": True,
        "category": category,
        "scam_type": None,
        "red_flags": red_flags or [],
        "explanation": "Test assessment",
        "segments_analyzed": segments,
        "aggregation": "single_message" if segments == 1 else "maximum_risk_across_overlapping_segments",
        "segment_risks": [risk] * segments,
    }


def test_sms_shield_parses_structured_indian_sms_verdict():
    result = SmsShieldClient._parse_result(
        json.dumps(
            {
                "verdict": "dangerous",
                "category": "scam",
                "scam_type": "kyc_fraud",
                "risk": 88,
                "red_flags": ["account threat", "suspicious link"],
                "explanation": "The sender pressures the recipient to open an unfamiliar link.",
            }
        )
    )
    assert result["model"] == "SMS-Shield"
    assert result["risk"] == 88
    assert result["verdict"] == "dangerous"
    assert result["scam_type"] == "kyc_fraud"
    assert "suspicious link" in result["red_flags"]


def test_sms_shield_canonicalizes_verdict_when_risk_band_conflicts():
    result = SmsShieldClient._parse_result(
        json.dumps(
            {
                "verdict": "safe",
                "category": "personal",
                "scam_type": None,
                "risk": 76,
                "red_flags": [],
                "explanation": "Inconsistent model output.",
            }
        )
    )
    assert result["verdict"] == "dangerous"
    assert result["model_verdict"] == "safe"
    assert result["verdict_consistent"] is False


def test_sms_shield_rejects_missing_or_invalid_risk_scores():
    for payload in (
        {"verdict": "dangerous", "risk": "unknown"},
        {"verdict": "safe", "risk": 101},
        {"verdict": "safe", "risk": float("nan")},
        {"verdict": "maybe", "risk": 20},
    ):
        try:
            SmsShieldClient._parse_result(json.dumps(payload))
        except ModelUnavailableError:
            continue
        raise AssertionError(f"Invalid model output should be rejected: {payload}")


def test_sms_shield_accepts_json_wrapped_after_a_short_preamble():
    result = SmsShieldClient._parse_result(
        'Result follows: {"verdict":"suspicious","risk":55,"category":"banking",'
        '"scam_type":null,"red_flags":["unusual request"],"explanation":"Check sender."}'
    )
    assert result["risk"] == 55
    assert result["verdict"] == "suspicious"


def test_long_message_is_split_into_overlapping_segments_with_full_coverage():
    message = "A" * 1300
    segments = SmsShieldClient._segments(message)
    assert len(segments) >= 3
    assert all(0 < len(segment) <= SmsShieldClient.MAX_SEGMENT_CHARS for segment in segments)
    assert segments[0] == message[:512]
    assert segments[-1][-1] == message[-1]
    # Each segment overlaps the following segment, so no boundary is skipped.
    assert all(
        segments[index][-SmsShieldClient.SEGMENT_OVERLAP_CHARS :]
        == segments[index + 1][: SmsShieldClient.SEGMENT_OVERLAP_CHARS]
        for index in range(len(segments) - 1)
    )


def test_long_message_aggregation_uses_highest_risk_segment(monkeypatch):
    detector = SmsShieldClient()
    risks = iter([10, 84, 25])

    async def fake_classify_segment(segment, sender):
        risk = next(risks)
        verdict = "safe" if risk < 40 else "suspicious" if risk < 70 else "dangerous"
        return _assessment(
            risk=risk,
            verdict=verdict,
            red_flags=[f"flag-{risk}"],
        )

    monkeypatch.setattr(detector, "_classify_segment", fake_classify_segment)
    result = asyncio.run(detector.classify("A" * 1100, "BANKALERT"))
    assert result["risk"] == 84
    assert result["verdict"] == "dangerous"
    assert result["segments_analyzed"] == 3
    assert result["segment_risks"] == [10, 84, 25]
    assert result["red_flags"] == ["flag-10", "flag-84", "flag-25"]


def test_indicbert_masking_replaces_links_and_indian_phone_numbers():
    raw = "Pay at https://example.com/path to 9876543210"
    prepared = HuggingFaceModels._prepare_indicbert_text(raw)
    assert "https://example.com" not in prepared
    assert "9876543210" not in prepared
    assert "[URL]" in prepared
    assert "[PHONE]" in prepared


def test_optional_indicbert_is_diagnostic_only(monkeypatch):
    from dataclasses import replace
    import app.models as models

    monkeypatch.setattr(
        models,
        "settings",
        replace(models.settings, indicbert_enabled=True),
    )
    detector = HuggingFaceModels()
    monkeypatch.setattr(detector, "package_status", lambda: (True, "test runtime"))

    captured = {}

    def fake_indicbert(text, truncation=True, max_length=None, top_k=None):
        captured["text"] = text
        captured["max_length"] = max_length
        return [
            {"label": "ham", "score": 0.05},
            {"label": "scam", "score": 0.95},
        ]

    monkeypatch.setattr(detector, "_load", lambda name, model_id: fake_indicbert)
    signal = asyncio.run(
        detector.challenger_signal("Pay at https://example.com to 9876543210")
    )
    assert signal is not None
    assert signal.available is True
    assert abs(signal.score - 14.25) < 1e-9
    assert captured["max_length"] == 128
    assert "[URL]" in captured["text"]
    assert "[PHONE]" in captured["text"]
    assert "excluded from fused scoring" in signal.detail


def test_optional_explainer_failure_does_not_discard_analysis_result(monkeypatch):
    service = AnalysisService()

    async def fake_classify(message, sender):
        return _assessment()

    async def no_challenger(raw):
        return None

    async def fake_similarity(raw, normalized, ollama):
        return {
            "available": False,
            "matches": [],
            "top_distance": None,
            "similarity_signal": 0.0,
            "detail": "disabled in test",
        }

    async def unavailable_explanation(payload):
        raise ModelUnavailableError("Ollama is offline")

    monkeypatch.setattr(service.sms_shield, "classify", fake_classify)
    monkeypatch.setattr(service.hf, "challenger_signal", no_challenger)
    monkeypatch.setattr(service.similarity, "search", fake_similarity)
    monkeypatch.setattr(service.ollama, "explain", unavailable_explanation)

    result = asyncio.run(
        service.analyze("Hey, are we meeting at six?", None, post_hoc=True)
    )
    assert result["risk_score"] >= 0
    assert result["label"] in {"LOW", "SUSPICIOUS", "HIGH RISK", "CRITICAL"}
    assert result["model_assessment"]["model"] == "SMS-Shield"
    assert result["explanation_status"]["requested"] is True
    assert result["explanation_status"]["available"] is False
    assert result["post_hoc"]["recommended_action"]


def test_api_schema_keeps_model_assessment_similarity_and_sender(monkeypatch):
    service = AnalysisService()

    async def fake_classify(message, sender):
        return _assessment()

    async def no_challenger(raw):
        return None

    async def fake_similarity(raw, normalized, ollama):
        return {"available": False, "matches": [], "similarity_signal": 0.0}

    monkeypatch.setattr(service.sms_shield, "classify", fake_classify)
    monkeypatch.setattr(service.hf, "challenger_signal", no_challenger)
    monkeypatch.setattr(service.similarity, "search", fake_similarity)

    result = asyncio.run(
        service.analyze("A normal check-in.", "PERSONAL", post_hoc=False)
    )
    response = AnalysisResponse(**result)
    assert response.sender == "PERSONAL"
    assert response.model_assessment["model"] == "SMS-Shield"
    assert response.similarity["available"] is False
    assert response.explanation_status["requested"] is False


def test_sms_shield_canonicalizes_unsupported_verdict_if_risk_is_valid():
    result = SmsShieldClient._parse_result(
        json.dumps({
            "verdict": "neutral", "category": "general", "scam_type": None,
            "risk": 55, "red_flags": [], "explanation": "Non-enum verdict."
        })
    )
    assert result["verdict"] == "suspicious"
    assert result["model_verdict"] == "neutral"
    assert result["verdict_supported"] is False
    assert result["verdict_consistent"] is False


def test_sms_shield_does_not_accept_unsupported_verdict_without_valid_risk():
    try:
        SmsShieldClient._parse_result(
            json.dumps({"verdict": "unknown", "category": "unknown", "red_flags": []})
        )
    except ModelUnavailableError:
        pass
    else:
        raise AssertionError("Missing/invalid risk must remain an error")
