import asyncio

from app.models import HuggingFaceModels, ModelSignal, ModelUnavailableError
from app.schemas import AnalysisResponse
from app.service import AnalysisService


HINSPAM = "Keshav0av/HinSpam-spam-detection"


def test_analyze_request_rejects_blank_but_preserves_original_message():
    from pydantic import ValidationError
    from app.schemas import AnalyzeRequest

    original = "  Please review this message.  "
    request = AnalyzeRequest(message=original)
    assert request.message == original

    try:
        AnalyzeRequest(message="   \n \t")
    except ValidationError:
        return
    raise AssertionError("Whitespace-only messages should be rejected")


def test_indicbert_masking_replaces_links_and_indian_phone_numbers():
    raw = "Pay at https://example.com/path to 9876543210"
    prepared = HuggingFaceModels._prepare_indicbert_text(raw)
    assert "https://example.com" not in prepared
    assert "9876543210" not in prepared
    assert "[URL]" in prepared
    assert "[PHONE]" in prepared


def test_optional_indicbert_is_exposed_as_diagnostic_challenger(monkeypatch):
    from dataclasses import replace
    import app.models as models

    monkeypatch.setattr(
        models,
        "settings",
        replace(models.settings, indicbert_enabled=True),
    )
    detector = HuggingFaceModels()
    monkeypatch.setattr(detector, "package_status", lambda: (True, "test runtime"))

    def fake_hinspam(text, truncation=True, top_k=None):
        return [
            {"label": "LABEL_0", "score": 0.1},
            {"label": "LABEL_1", "score": 0.9},
        ]

    class FakeSilverGuard:
        def predict(self, sender, raw):
            return {"threat_score": 0.2}

    captured = {}

    def fake_indicbert(text, truncation=True, max_length=None, top_k=None):
        captured["text"] = text
        captured["max_length"] = max_length
        return [
            {"label": "ham", "score": 0.05},
            {"label": "scam", "score": 0.95},
        ]

    def fake_load(name, model_id):
        return {
            "hinspam": fake_hinspam,
            "silverguard": FakeSilverGuard(),
            "indicbert": fake_indicbert,
        }[name]

    monkeypatch.setattr(detector, "_load", fake_load)
    signals = asyncio.run(
        detector.signals("Pay at https://example.com to 9876543210", None)
    )

    challenger = next(signal for signal in signals if signal.name == "IndicBERT")
    assert challenger.available is True
    assert abs(challenger.score - 14.25) < 1e-9
    assert captured["max_length"] == 128
    assert "[URL]" in captured["text"]
    assert "[PHONE]" in captured["text"]


def test_hinspam_uses_documented_label_id_mapping():
    probability = HuggingFaceModels._risk_probability(
        [
            {"label": "LABEL_0", "score": 0.08},
            {"label": "LABEL_1", "score": 0.92},
        ],
        HINSPAM,
    )
    assert abs(probability - 0.92) < 1e-9


def test_semantic_labels_use_scam_probability_not_top_label_confidence():
    probability = HuggingFaceModels._risk_probability(
        [
            {"label": "ham", "score": 0.73},
            {"label": "spam", "score": 0.27},
        ],
        "custom/model-with-semantic-labels",
    )
    assert abs(probability - 0.27) < 1e-9


def test_single_benign_top_class_is_inverted_to_risk_probability():
    probability = HuggingFaceModels._risk_probability(
        {"label": "legitimate", "score": 0.81},
        "custom/model-with-semantic-labels",
    )
    assert abs(probability - 0.19) < 1e-9


def test_unknown_numeric_labels_are_not_guessed_for_custom_models():
    try:
        HuggingFaceModels._risk_probability(
            [{"label": "LABEL_0", "score": 0.2}, {"label": "LABEL_1", "score": 0.8}],
            "someone/custom-model",
        )
    except ModelUnavailableError:
        return
    raise AssertionError("Unknown label ids must fail safely instead of guessing")


def test_non_probability_scores_are_rejected():
    try:
        HuggingFaceModels._risk_probability(
            {"label": "spam", "score": 1.2},
            HINSPAM,
        )
    except ModelUnavailableError:
        return
    raise AssertionError("Out-of-range class scores must fail validation")


def test_optional_explainer_failure_does_not_discard_analysis_result():
    service = AnalysisService()

    async def fake_signals(raw, sender):
        return [
            ModelSignal("HinSpam", True, 2.0, "test signal"),
            ModelSignal("SilverGuard", True, 1.0, "test signal"),
        ]

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

    service.hf.signals = fake_signals
    service.similarity.search = fake_similarity
    service.ollama.explain = unavailable_explanation

    result = asyncio.run(
        service.analyze("Hey, are we meeting at six?", None, post_hoc=True)
    )

    assert result["risk_score"] >= 0
    assert result["label"] in {"LOW", "SUSPICIOUS", "HIGH RISK", "CRITICAL"}
    assert result["explanation_status"]["requested"] is True
    assert result["explanation_status"]["available"] is False
    assert result["post_hoc"]["recommended_action"]


def test_api_schema_keeps_similarity_sender_and_explanation_status():
    service = AnalysisService()

    async def fake_signals(raw, sender):
        return [
            ModelSignal("HinSpam", True, 2.0, "test signal"),
            ModelSignal("SilverGuard", True, 1.0, "test signal"),
        ]

    async def fake_similarity(raw, normalized, ollama):
        return {"available": False, "matches": [], "similarity_signal": 0.0}

    service.hf.signals = fake_signals
    service.similarity.search = fake_similarity

    result = asyncio.run(
        service.analyze("A normal check-in.", "PERSONAL", post_hoc=False)
    )
    response = AnalysisResponse(**result)

    assert response.sender == "PERSONAL"
    assert response.similarity["available"] is False
    assert response.explanation_status["requested"] is False
