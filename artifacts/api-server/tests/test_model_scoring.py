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
        AnalyzeRequest(message="   \\n \\t")
    except ValidationError:
        return
    raise AssertionError("Whitespace-only messages should be rejected")


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
