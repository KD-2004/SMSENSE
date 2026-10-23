import asyncio
import math
import os

import pytest

from app.config import settings
from app.models import OllamaClient


REAL_RUNTIME = os.getenv("RUN_REAL_MODEL_TESTS", "").strip().lower() == "true"
pytestmark = pytest.mark.skipif(
    not REAL_RUNTIME,
    reason="set RUN_REAL_MODEL_TESTS=true to run live model tests",
)


@pytest.mark.integration
def test_live_analyze_endpoint_returns_frontend_renderable_sms_shield_result(monkeypatch):
    """Exercise real inference through the same API route used by the browser."""
    import httpx
    import app.main as api

    # Keep this opt-in integration test from writing sample messages to the user's
    # real local history database.
    monkeypatch.setattr(api.storage, "save_analysis", lambda _result: 12345)

    async def request():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api.app),
            base_url="http://testserver",
            timeout=700,
        ) as client:
            response = await client.post(
                "/api/analyze",
                json={
                    "message": "Your bank account will be blocked. Update KYC now at https://example.com/verify.",
                    "sender": "VM-SBIHELP",
                },
            )
        assert response.status_code == 200, response.text
        result = response.json()

        # These are the fields the visible frontend needs to render a complete result.
        for field in (
            "risk_score", "label", "category", "language", "raw_message",
            "normalized_message", "model_assessment", "decision_diagnostics",
            "evidence", "suspicious_phrases", "url_analysis", "obfuscation",
            "explanation", "storage_status",
        ):
            assert field in result, f"API response is missing frontend field: {field}"

        assert result["id"] == 12345
        assert 0 <= result["risk_score"] <= 100
        assert result["label"] in {"LOW", "SUSPICIOUS", "HIGH RISK", "CRITICAL"}
        assert result["model_assessment"]["model"] == "SMS-Shield"
        assert result["model_assessment"]["model_id"] == settings.sms_shield_model
        assert 0 <= result["model_assessment"]["risk"] <= 100
        assert isinstance(result["evidence"], dict)
        assert isinstance(result["explanation"]["model_signals"], list)
        assert any(
            signal.get("name") == "SMS-Shield"
            for signal in result["explanation"]["model_signals"]
        )
        return result

    asyncio.run(request())

@pytest.mark.integration
def test_optional_qwen_explainer_returns_usable_content_when_installed():
    """Qwen is an enhancement; its absence must not fail primary-detector testing."""
    client = OllamaClient()
    available, detail = asyncio.run(client.model_status(settings.ollama_chat_model))
    if not available:
        pytest.skip(detail)

    result = asyncio.run(client.explain({
        "final_score": 85,
        "final_label": "CRITICAL",
        "category": "KYC_FRAUD",
        "top_evidence": ["content", "behavior"],
        "raw_message": "Your account will be blocked. Send your PIN.",
        "normalized_message": "your account will be blocked. send your pin.",
        "suspicious_phrases": ["account will be blocked", "send your pin"],
    }))
    assert isinstance(result.get("summary"), str) and result["summary"].strip()
    assert isinstance(result.get("recommended_action"), str) and result["recommended_action"].strip()
    assert isinstance(result.get("suspicious_phrases"), list)


@pytest.mark.integration
def test_optional_embedding_model_returns_finite_vector_when_installed():
    """Embedding/FAISS is optional and is not required for a core SMS verdict."""
    client = OllamaClient()
    available, detail = asyncio.run(client.model_status(settings.ollama_embedding_model))
    if not available:
        pytest.skip(detail)

    vector = asyncio.run(client.embed("SMSense runtime smoke test"))
    assert vector
    assert all(math.isfinite(value) for value in vector)
