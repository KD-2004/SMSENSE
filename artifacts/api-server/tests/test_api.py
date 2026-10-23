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
    post_hoc_flags = []

    async def fake_analyze(message, sender, timestamp, post_hoc):
        post_hoc_flags.append(post_hoc)
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
    # Core scoring returns without waiting for the optional explanation model.
    assert post_hoc_flags == [False]


def test_explain_endpoint_returns_optional_model_result(monkeypatch):
    async def fake_explain(payload):
        assert payload["final_score"] == 82
        assert payload["final_label"] == "CRITICAL"
        return {
            "summary": "The message threatens account closure and requests credentials.",
            "suspicious_phrases": ["account will be blocked"],
            "recommended_action": "Verify with the official bank app.",
        }

    monkeypatch.setattr(api.service.ollama, "explain", fake_explain)

    async def request():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api.app),
            base_url="http://testserver",
        ) as client:
            return await client.post("/api/explain", json={
                "risk_score": 82,
                "label": "CRITICAL",
                "category": "KYC_FRAUD",
                "raw_message": "Your account will be blocked. Send your PIN.",
                "normalized_message": "your account will be blocked. send your pin.",
                "top_evidence": ["content", "behavior"],
                "suspicious_phrases": ["account will be blocked"],
            })

    response = asyncio.run(request())
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["available"] is True
    assert body["summary"].startswith("The message threatens")
    assert body["recommended_action"] == "Verify with the official bank app."


def test_explain_endpoint_falls_back_without_breaking_primary_result(monkeypatch):
    async def unavailable(_payload):
        raise RuntimeError("optional Qwen model is not installed")

    monkeypatch.setattr(api.service.ollama, "explain", unavailable)

    async def request():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api.app),
            base_url="http://testserver",
        ) as client:
            return await client.post("/api/explain", json={
                "risk_score": 74,
                "label": "HIGH RISK",
                "category": "URL_REVIEW",
                "raw_message": "Open this link to verify your account.",
                "normalized_message": "open this link to verify your account.",
                "top_evidence": ["content"],
                "suspicious_phrases": [],
            })

    response = asyncio.run(request())
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["available"] is False
    assert "primary detection pipeline" in body["summary"]
    assert body["recommended_action"]


def test_analyze_endpoint_rejects_whitespace_only_message():
    async def request():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api.app),
            base_url="http://testserver",
        ) as client:
            return await client.post("/api/analyze", json={"message": "   \n\t"})

    response = asyncio.run(request())

    assert response.status_code == 422



def test_cors_allows_documented_local_frontend_and_rejects_unlisted_origin():
    async def request(origin):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api.app),
            base_url="http://testserver",
        ) as client:
            return await client.get("/api/healthz", headers={"Origin": origin})

    local_response = asyncio.run(request("http://127.0.0.1:5500"))
    unlisted_response = asyncio.run(request("https://untrusted.example"))

    assert local_response.status_code == 200
    assert local_response.headers.get("access-control-allow-origin") == "http://127.0.0.1:5500"
    assert unlisted_response.status_code == 200
    assert "access-control-allow-origin" not in unlisted_response.headers


def test_cors_preflight_accepts_browser_requested_headers_for_allowed_origin():
    async def request():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api.app),
            base_url="http://testserver",
        ) as client:
            return await client.options(
                "/api/analyze",
                headers={
                    "Origin": "http://127.0.0.1:5500",
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "content-type,x-requested-with",
                },
            )

    response = asyncio.run(request())

    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://127.0.0.1:5500"
    assert "post" in response.headers.get("access-control-allow-methods", "").lower()
    assert "x-requested-with" in response.headers.get("access-control-allow-headers", "").lower()


def test_api_responses_use_no_store_and_basic_security_headers():
    async def request():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api.app),
            base_url="http://testserver",
        ) as client:
            return await client.get("/api/healthz")

    response = asyncio.run(request())

    assert response.headers.get("cache-control") == "no-store"
    assert response.headers.get("x-content-type-options") == "nosniff"
    assert response.headers.get("x-frame-options") == "DENY"
    assert response.headers.get("referrer-policy") == "no-referrer"
