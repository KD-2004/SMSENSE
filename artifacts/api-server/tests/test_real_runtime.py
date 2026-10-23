import asyncio
import math
import os

import pytest

from app.config import settings
from app.models import OllamaClient, SmsShieldClient


REAL_RUNTIME = os.getenv("RUN_REAL_MODEL_TESTS", "").strip().lower() == "true"
pytestmark = pytest.mark.skipif(
    not REAL_RUNTIME,
    reason="set RUN_REAL_MODEL_TESTS=true to run live model tests",
)


@pytest.mark.integration
def test_primary_sms_shield_model_returns_structured_result():
    """Exercise the detector that actually powers SMSense, not retired challengers."""

    async def run():
        client = SmsShieldClient()
        available, detail = await client.status()
        assert available, detail

        result = await client.classify(
            "Your bank account will be blocked. Update KYC now at https://example.com/verify.",
            "VM-SBIHELP",
        )
        assert result["model"] == "SMS-Shield"
        assert result["model_id"] == settings.sms_shield_model
        assert isinstance(result["risk"], int)
        assert 0 <= result["risk"] <= 100
        assert result["verdict"] in {"safe", "suspicious", "dangerous"}
        assert isinstance(result.get("red_flags"), list)
        assert result.get("segments_analyzed", 0) >= 1
        assert len(result.get("segment_risks", [])) >= 1

    asyncio.run(run())


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
