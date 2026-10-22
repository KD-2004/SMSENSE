import asyncio

import httpx

import app.main as api


def _analysis_result(message: str, sender: str | None = None) -> dict:
    return {
        "risk_score": 12,
        "label": "LOW",
        "category": "GENERAL_MESSAGE",
        "language": "English",
        "raw_message": message,
        "normalized_message": message.lower(),
        "sender": sender,
        "similarity": {
            "available": False,
            "matches": [],
            "top_distance": None,
            "similarity_signal": 0.0,
            "detail": "disabled in API test",
        },
        "explanation_status": {
            "requested": False,
            "available": False,
            "detail": "not requested",
        },
        "evidence": {
            "content": 2.0,
            "sender": 0.0,
            "url": 0.0,
            "behavior": 5.0,
            "obfuscation": 0.0,
            "similarity": 0.0,
        },
        "suspicious_phrases": [],
        "url_analysis": {"count": 0, "urls": []},
        "obfuscation": {},
        "explanation": {"top_contributing_evidence": ["behavior"], "model_signals": []},
    }


def test_healthz_is_lightweight_and_reachable():
    async def request():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api.app),
            base_url="http://testserver",
        ) as client:
            return await client.get("/api/healthz")

    response = asyncio.run(request())
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_analyze_endpoint_preserves_schema_fields(monkeypatch):
    async def fake_analyze(message, sender, timestamp, post_hoc):
        return _analysis_result(message, sender)

    monkeypatch.setattr(api.service, "analyze", fake_analyze)
    monkeypatch.setattr(api.storage, "save_analysis", lambda result: 321)

    async def request():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api.app),
            base_url="http://testserver",
        ) as client:
            return await client.post(
                "/api/analyze",
                json={"message": "  Please review this text.  ", "sender": "PERSONAL"},
            )

    response = asyncio.run(request())

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["id"] == 321
    assert body["raw_message"] == "  Please review this text.  "
    assert body["sender"] == "PERSONAL"
    assert body["similarity"]["available"] is False
    assert body["explanation_status"]["requested"] is False


def test_analyze_endpoint_rejects_whitespace_only_message():
    async def request():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api.app),
            base_url="http://testserver",
        ) as client:
            return await client.post("/api/analyze", json={"message": "   \n\t"})

    response = asyncio.run(request())

    assert response.status_code == 422
